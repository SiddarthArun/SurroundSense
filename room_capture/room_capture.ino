// room_capture: XIAO ESP32S3 Sense with 4 PDM mics. Streams audio to the laptop over USB,
// or classifies it on the device (MODE_INFER).
//
// Tools menu: Board "XIAO_ESP32S3" | USB CDC On Boot "Enabled" | PSRAM "OPI PSRAM"
//
// Uses the ESP-IDF PDM driver directly: the Arduino I2S library ignores the slot mask
// in PDM mode, so it can only ever read data line 0 (mics 1 and 2).
//
// Bring-up order:
//   1. MODE_LEVELS. Upload, open Serial Monitor, snap fingers next to each mic in turn:
//      only its meter should jump. Rate should read about 128000 B/s (4 ch).
//   2. If a snap shows on the wrong meter, fix SLOT_FOR_MIC below.
//   3. MODE_STREAM. Upload, CLOSE the Serial Monitor, run host/recorder.py.
//   4. Still dropping blocks? STREAM_CHANNELS 1 sends mic 1 only (recorder.py --channels 1).
//   5. MODE_INFER. Train, then `python export.py --firmware-dir room_capture` (writes model.h
//      here). Upload, open Serial Monitor: probabilities every 0.5 s, Event lines on a
//      detection with the direction it came from, and each Event broadcast over ESP-NOW to the
//      wristband (wristband/). Settings at the top of yam_infer.h (classifier) and doa.h
//      (direction, including which mic is worn where).
//
// Wiring (all mics: 3V -> 3V3, GND -> GND), and where each is worn on the harness:
//   Mic 1: CLK D0, DAT D1, SEL GND  chest, right    Mic 2: CLK D0, DAT D1, SEL 3V3  chest, left
//   Mic 3: CLK D0, DAT D3, SEL GND  back, right     Mic 4: CLK D0, DAT D3, SEL 3V3  back, left

#include "driver/i2s_pdm.h"
#include <protocol.h>

// ---------- settings ----------
#define MODE_STREAM     0   // binary audio stream for host/recorder.py
#define MODE_LEVELS     1   // readable level meters in Serial Monitor
#define MODE_INFER      2   // on-device classifier, text results in Serial Monitor
#define MODE            MODE_INFER
#define PDM_LINES       2   // 1: mics 1-2 (D1 only). 2: all four mics (D1 + D3)
#define STREAM_CHANNELS 4   // binary mode: 4 = every mic, 1 = mic 1 only (a quarter of the USB
                            // traffic; enough for training clips, recorder.py --channels 1)
#define GAIN_SHIFT      1   // digital gain, each step doubles. Raise (1-4) if levels stay tiny

#define PIN_PDM_CLK   1   // D0 (GPIO1): shared clock to every mic
#define PIN_PDM_DAT0  2   // D1 (GPIO2): mics 1 and 2
#define PIN_PDM_DAT1  4   // D3 (GPIO4): mics 3 and 4

constexpr int    CH          = PDM_LINES * 2;
constexpr int    FRAMES      = AUDIO_FRAMES_PER_BLOCK;          // 256 frames = 16 ms
constexpr size_t BLOCK_BYTES = FRAMES * CH * sizeof(int16_t);

// Which slot in the driver's output holds each mic. The driver puts the right-hand mic
// (SEL = 3V3) of each pair first, so mic 1 is slot 1. Confirm with the tap test.
#if PDM_LINES == 2
const uint8_t SLOT_FOR_MIC[CH] = {1, 0, 3, 2};
#else
const uint8_t SLOT_FOR_MIC[CH] = {1, 0};
#endif

#if STREAM_CHANNELS != 1 && STREAM_CHANNELS != (PDM_LINES * 2)
#error "STREAM_CHANNELS must be 1 or equal to the number of mics (PDM_LINES * 2)"
#endif
constexpr size_t PAYLOAD_BYTES = FRAMES * STREAM_CHANNELS * sizeof(int16_t);

#if MODE == MODE_INFER
#ifndef BOARD_HAS_PSRAM
#error "MODE_INFER needs Tools > PSRAM: OPI PSRAM. Wi-Fi takes internal RAM, so the tensor arena lives in PSRAM."
#endif
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include "yam_infer.h"   // classifier; MODEL_CLASSES and ALERT_ON at its top
#endif

i2s_chan_handle_t rxChan = nullptr;
int16_t  raw[FRAMES * CH];      // as read from the driver
int16_t  block[FRAMES * CH];    // processed, mic order, what gets sent
float    dcPrevIn[CH];
float    dcPrevOut[CH];
uint16_t seq = 0;

// Fast blink forever: the PDM driver refused to start (check settings and pins).
void fail() {
  pinMode(LED_BUILTIN, OUTPUT);
  while (true) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(100);
  }
}

bool startMics() {
  i2s_chan_config_t chanCfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);  // PDM RX is I2S0 only
  // Deeper DMA buffering (8 x 480 frames = 240 ms) so a short USB stall doesn't lose audio.
  // Each DMA buffer must stay under 4092 bytes: 480 frames x 4 slots x 2 bytes = 3840.
  chanCfg.dma_desc_num = 8;
  chanCfg.dma_frame_num = 480;
  if (i2s_new_channel(&chanCfg, nullptr, &rxChan) != ESP_OK) return false;

  i2s_pdm_rx_config_t cfg = {
    .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
    .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .clk = (gpio_num_t)PIN_PDM_CLK,
      .dins = {
        (gpio_num_t)PIN_PDM_DAT0,
#if PDM_LINES == 2
        (gpio_num_t)PIN_PDM_DAT1,
#else
        GPIO_NUM_NC,
#endif
        GPIO_NUM_NC,
        GPIO_NUM_NC,
      },
      .invert_flags = {
        .clk_inv = false,
      },
    },
  };

#if PDM_LINES == 2
  cfg.slot_cfg.slot_mask = (i2s_pdm_slot_mask_t)(I2S_PDM_RX_LINE0_SLOT_LEFT | I2S_PDM_RX_LINE0_SLOT_RIGHT |
                                                 I2S_PDM_RX_LINE1_SLOT_LEFT | I2S_PDM_RX_LINE1_SLOT_RIGHT);
#else
  cfg.slot_cfg.slot_mask = I2S_PDM_SLOT_BOTH;   // line 0, left + right
#endif

  if (i2s_channel_init_pdm_rx_mode(rxChan, &cfg) != ESP_OK) return false;
  if (i2s_channel_enable(rxChan) != ESP_OK) return false;
  return true;
}

// ---------- radio: ESP-NOW alerts to the wristband (MODE_INFER only) ----------
#if MODE == MODE_INFER
#define HEARTBEAT_MS 5000

// Broadcast on purpose: the band learns this board's MAC from each message and sends its ACK
// straight back, so there is no MAC to paste. Put the band's MAC here (bringup/esp01_hello)
// to send unicast instead; the band answers either way.
static const uint8_t WRIST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

uint16_t radioSeq = 0;
esp_err_t lastSendErr = ESP_OK;
uint32_t sentEvents = 0;
uint32_t lastHeartbeat = 0;
volatile bool     ackPending = false;
volatile uint16_t ackSeq = 0;
volatile uint32_t ackRttMs = 0;

// The band echoes seq and t_ms, so the RTT is millis() minus its echo: one clock, no sync.
void onRadioRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (len != (int)sizeof(msg_t)) return;
  msg_t m;
  memcpy(&m, data, sizeof(m));
  if (m.type != MSG_ACK) return;
  ackSeq = m.seq;
  ackRttMs = millis() - m.t_ms;
  ackPending = true;
}

// Puts the radio on ESPNOW_CHANNEL; returns the channel it actually ended up on.
// The promiscuous toggle is what makes the change stick on an idle station.
uint8_t setHomeChannel() {
  esp_wifi_set_promiscuous(true);
  const esp_err_t err = esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
  uint8_t primary = 0;
  wifi_second_chan_t second;
  esp_wifi_get_channel(&primary, &second);
  if (err != ESP_OK || primary != ESPNOW_CHANNEL) {
    Serial.printf("[yamDetector] radio: set channel %d failed (%s), radio is on %u\n",
                  ESPNOW_CHANNEL, esp_err_to_name(err), (unsigned)primary);
  }
  return primary;
}

bool radioSetup() {
  // Keep the station idle: a saved SSID or an auto-reconnect makes it scan, which hops channels
  // and makes ESP-NOW refuse every send ("Peer channel is not equal to the home channel").
  WiFi.persistent(false);          // ignore any SSID an earlier sketch saved to flash
  if (!WiFi.mode(WIFI_STA)) {
    // Wi-Fi buffers must come from internal RAM; this says whether that is what ran out.
    Serial.printf("[yamDetector] radio: WiFi.mode failed, internal heap free %u B, largest block %u B\n",
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    return false;
  }
  WiFi.setAutoReconnect(false);
  WiFi.disconnect();
  const uint8_t onChannel = setHomeChannel();
  esp_err_t err = esp_now_init();
  if (err != ESP_OK) {
    Serial.printf("[yamDetector] radio: esp_now_init failed (%s)\n", esp_err_to_name(err));
    return false;
  }
  esp_now_register_recv_cb(onRadioRecv);   // the ACK is the delivery report

  // ESP-IDF only sends to addresses in its peer list, the broadcast address included.
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, WRIST_ADDR, 6);
  peer.channel = ESPNOW_CHANNEL;
  peer.ifidx   = WIFI_IF_STA;
  peer.encrypt = false;
  err = esp_now_add_peer(&peer);
  if (err != ESP_OK && err != ESP_ERR_ESPNOW_EXIST) {
    Serial.printf("[yamDetector] radio: add_peer failed (%s)\n", esp_err_to_name(err));
    return false;
  }
  Serial.printf("[yamDetector] radio: ESP-NOW channel %d (radio on %u), alerts broadcast\n",
                ESPNOW_CHANNEL, (unsigned)onChannel);
  return true;
}

void radioSend(uint8_t type, uint8_t cls, uint8_t conf, uint8_t dir) {
  msg_t m = {};
  m.type     = type;
  m.class_id = cls;
  m.conf     = conf;
  m.dir      = dir;
  m.seq      = ++radioSeq;
  m.t_ms     = millis();
  // Print only when the status changes, so a failure shows once instead of every heartbeat.
  esp_err_t err = esp_now_send(WRIST_ADDR, (const uint8_t *)&m, sizeof(m));
  if (err == ESP_ERR_ESPNOW_CHAN) {   // something moved the radio: put it back and retry once
    setHomeChannel();
    err = esp_now_send(WRIST_ADDR, (const uint8_t *)&m, sizeof(m));
  }
  if (err != lastSendErr) {
    lastSendErr = err;
    if (err == ESP_OK) Serial.println("[yamDetector] radio: sending again");
    else Serial.printf("[yamDetector] radio: send failed (%s)\n", esp_err_to_name(err));
  }
}

// Runs after every audio block: forwards a new Event, keeps the link alive, reports ACKs.
void radioLoop() {
  if (eventCount != sentEvents) {
    sentEvents = eventCount;
    radioSend(MSG_ALERT, lastEventClass, lastEventConf, lastEventDir);
  }
  const uint32_t now = millis();
  if (now - lastHeartbeat >= HEARTBEAT_MS) {
    lastHeartbeat = now;
    radioSend(MSG_HEARTBEAT, CLS_BACKGROUND, 0, DIR_UNKNOWN);
  }
  if (ackPending) {
    ackPending = false;
    Serial.printf("[yamDetector] wristband ACK seq %u, %lu ms\n",
                  (unsigned)ackSeq, (unsigned long)ackRttMs);
  }
}
#endif

void setup() {
  Serial.setTxBufferSize(8192);   // room for a few blocks
  Serial.setTxTimeoutMs(50);      // ride out short USB stalls; the 240 ms DMA buffer covers the wait
  Serial.begin(115200);           // baud is ignored on native USB

#if MODE == MODE_INFER
  const uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);   // give the Serial Monitor a moment
  // Radio first: Wi-Fi buffers must be internal RAM, while the tensor arena can fall back to PSRAM.
  const bool radioOk = radioSetup();
  inferSetup();
  // One MSG_TEST at boot: the band shows it as TEST and buzzes, which checks the radio path
  // without waiting for a real siren.
  if (radioOk) radioSend(MSG_TEST, CLS_SIREN, 0, DIR_UNKNOWN);
  else Serial.println("[yamDetector] radio: ESP-NOW init failed, alerts will not be sent");
#endif
  if (!startMics()) fail();
}

// Reorder to mic order, DC blocker on every channel, then fixed gain with clipping.
// The laptop records exactly what the on-device model will later see.
void processBlock() {
  for (int f = 0; f < FRAMES; f++) {
    for (int m = 0; m < CH; m++) {
      const float x = raw[f * CH + SLOT_FOR_MIC[m]];
      const float y = x - dcPrevIn[m] + 0.995f * dcPrevOut[m];
      dcPrevIn[m] = x;
      dcPrevOut[m] = y;
      int32_t v = (int32_t)(y * (float)(1 << GAIN_SHIFT));
      if (v > 32767) v = 32767;
      if (v < -32768) v = -32768;
      block[f * CH + m] = (int16_t)v;
    }
  }
}

#if MODE == MODE_LEVELS
double   sumSq[CH];
int32_t  peak[CH];
uint32_t framesSince = 0;
uint32_t bytesSince = 0;
uint32_t lastPrint = 0;

// Prints twice a second: measured data rate (tells you how many channels really arrive)
// and a level meter per mic.
void textReport() {
  for (int f = 0; f < FRAMES; f++) {
    for (int c = 0; c < CH; c++) {
      const int32_t s = block[f * CH + c];
      sumSq[c] += (double)s * s;
      const int32_t a = s < 0 ? -s : s;
      if (a > peak[c]) peak[c] = a;
    }
  }
  framesSince += FRAMES;
  bytesSince += BLOCK_BYTES;

  const uint32_t now = millis();
  if (now - lastPrint < 500) return;

  const float secs = (now - lastPrint) / 1000.0f;
  const float bytesPerSec = bytesSince / secs;
  Serial.printf("%6.0f B/s (~%.1f ch) |", bytesPerSec, bytesPerSec / (AUDIO_SAMPLE_RATE * 2.0f));

  for (int c = 0; c < CH; c++) {
    const float rms = sqrt(sumSq[c] / framesSince) + 1e-3f;
    const float dbfs = 20.0f * log10f(rms / 32768.0f);
    int bars = (int)((dbfs + 80.0f) / 4.0f);    // -80 dBFS = empty, 0 dBFS = 20 bars
    if (bars < 0) bars = 0;
    if (bars > 20) bars = 20;
    char meter[21];
    for (int b = 0; b < 20; b++) meter[b] = b < bars ? '#' : '.';
    meter[20] = '\0';
    Serial.printf(" m%d %s %6.1f dB pk %5ld |", c + 1, meter, dbfs, (long)peak[c]);
    sumSq[c] = 0;
    peak[c] = 0;
  }
  Serial.println();
  framesSince = 0;
  bytesSince = 0;
  lastPrint = now;
}
#elif MODE == MODE_STREAM
// Header from protocol.h: A5 5A, type, seq (uint16 LE), payload length (uint16 LE), then samples.
// Header and samples go out in one write, so a frame is never split between two writes.
uint8_t frameBuf[7 + PAYLOAD_BYTES];

void sendBlock() {
  if (!Serial) return;   // no laptop connected: skip, keep capturing
  frameBuf[0] = FRAME_MAGIC_0;
  frameBuf[1] = FRAME_MAGIC_1;
  frameBuf[2] = FRAME_TYPE_AUDIO;
  frameBuf[3] = (uint8_t)(seq & 0xFF);
  frameBuf[4] = (uint8_t)(seq >> 8);
  frameBuf[5] = (uint8_t)(PAYLOAD_BYTES & 0xFF);
  frameBuf[6] = (uint8_t)(PAYLOAD_BYTES >> 8);
#if STREAM_CHANNELS == 1
  int16_t *out = (int16_t *)(frameBuf + 7);
  for (int f = 0; f < FRAMES; f++) out[f] = block[f * CH];   // mic 1 only
#else
  memcpy(frameBuf + 7, block, PAYLOAD_BYTES);
#endif
  Serial.write(frameBuf, sizeof(frameBuf));
}
#endif

void loop() {
  size_t got = 0;
  if (i2s_channel_read(rxChan, raw, BLOCK_BYTES, &got, 1000) != ESP_OK || got != BLOCK_BYTES) {
    return;   // timeout or partial read: drop it
  }
  processBlock();
#if MODE == MODE_INFER
  inferPushBlock(block, FRAMES, CH);
  radioLoop();
#elif MODE == MODE_LEVELS
  textReport();
#else
  sendBlock();
#endif
  seq++;   // counts every block, so the recorder sees any that were skipped
}
