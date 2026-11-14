// wristband: ESP-01 (ESP8266) alert receiver. Gets msg_t from the room unit over ESP-NOW,
// buzzes the motor and names the sound on the 128x64 I2C OLED. Answers every message with
// MSG_ACK (echoing seq and t_ms), which the room unit prints as "wristband ACK seq N, M ms".
//
// Tools menu: Board "Generic ESP8266 Module" | Upload Speed 115200 | Reset Method "no dtr"
//             Flash Size "1MB" | CPU Frequency 80 MHz
// Libraries: Adafruit SSD1306 + Adafruit GFX Library (Library Manager)
//
// The board has no buttons: no reset, no boot button, nothing to press. Flash mode comes from
// holding GPIO0 low when power arrives, so flash the module out of the board or wire GPIO0 to
// GND, apply power, Upload, then let GPIO0 go and power-cycle. See
// bringup/esp01_hello/esp01_hello.ino. The motor twitches during an upload because its driver
// sits on GPIO3/RX, which carries the bootloader's data. Because nothing can be pressed, nothing
// here may need a reset: setup() never halts, and loop() retries the radio and the OLED if
// either failed to start.
//
// Wiring (the ESP-01 is 3.3 V only: 2x AA, no regulator, never 5 V):
//   VCC -> battery +, 100 uF + 100 nF to GND        CH_PD -> 3V3 (or the chip stays off)
//   SDA GPIO0 -> OLED SDA (module pull-up),  SCL GPIO2 -> OLED SCL (module pull-up)
//   Motor: GPIO3/RX -> 1k -> BC337 base, emitter GND, motor between 3V3 and collector,
//          1N4148 across the motor (cathode to 3V3), 10k from base to GND so the motor stays
//          still while GPIO3 floats at power-up
//   GPIO1/TX: 115200 debug out through the DevKit; GPIO3 is the UART's RX pin, so the sketch
//             takes it back with pinMode() and the band has no serial input.

#include <ESP8266WiFi.h>
#include <user_interface.h>
#include <espnow.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <protocol.h>

// ---------- settings ----------
#define PIN_SDA         0    // GPIO0, boot strap: the OLED's pull-ups hold it high
#define PIN_SCL         2    // GPIO2, boot strap: must be high at boot too
#define PIN_MOTOR       3    // GPIO3 / RX, motor through an NPN transistor
#define MOTOR_MS        400  // one buzz per alert
#define ALERT_HOLD_MS   6000 // how long the big class name stays up
#define LINK_TIMEOUT_MS 12000 // silence from the room unit (heartbeat is every 5 s)
#define LOW_BATT_MV     2900 // replace the AAs below this
#define OLED_ADDR       0x3C // 0x3D on some modules
#define OLED_W          128
#define OLED_H          64

ADC_MODE(ADC_VCC);   // lets ESP.getVcc() read the supply; the ESP-01 has no ADC pin

Adafruit_SSD1306 display(OLED_W, OLED_H, &Wire, -1);
bool     displayOk = false;
uint32_t lastDisplayTry = 0;

// Filled by the ESP-NOW callback (WiFi task), consumed in loop().
msg_t   inbox;
uint8_t inboxMac[6];
bool    inboxReady = false;

msg_t    lastMsg;              // last message of any type
bool     haveAlert = false;
uint32_t rxCount = 0;
uint32_t alertCount = 0;
uint32_t lastRxMs = 0;         // millis() of the last message
uint32_t lastAlertMs = 0;
uint32_t motorUntil = 0;
uint32_t lastRender = 0;
bool     dirty = true;

// ---------- radio ----------
void onRecv(u8 *mac, u8 *data, u8 len) {
  if (len < sizeof(msg_t)) return;
  memcpy(&inbox, data, sizeof(msg_t));
  memcpy(inboxMac, mac, 6);
  inboxReady = true;
}

void onSent(u8 *, u8 status) {
  if (status != 0) Serial.println("[wristband] ACK was not delivered");
}

bool     radioOk = false;
uint32_t lastRadioTry = 0;

// Kept as a function so loop() can retry it: with no reset button, a failed esp_now_init()
// must not leave the band silent until someone pulls a cell.
bool radioUp() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  WiFi.setSleepMode(WIFI_NONE_SLEEP);   // an alert must not wait for a beacon (costs ~70 mA)
  if (esp_now_init() != 0) return false;
  esp_now_set_self_role(ESP_NOW_ROLE_COMBO);   // receive alerts and send ACKs
  esp_now_register_recv_cb(onRecv);
  esp_now_register_send_cb(onSent);
  wifi_set_channel(ESPNOW_CHANNEL);            // must match the room unit
  return true;
}

// Ack to the address the alert came from, so neither board needs the other's MAC in code.
void sendAck(const uint8_t *mac, const msg_t *m) {
  if (esp_now_is_peer_exist((u8 *)mac) == 0) {
    esp_now_add_peer((u8 *)mac, ESP_NOW_ROLE_COMBO, ESPNOW_CHANNEL, nullptr, 0);
  }
  msg_t ack = {};
  ack.type     = MSG_ACK;
  ack.class_id = m->class_id;
  ack.conf     = m->conf;
  ack.zone     = m->zone;
  ack.seq      = m->seq;
  ack.t_ms     = m->t_ms;   // the room unit subtracts this from its own clock
  esp_now_send((u8 *)mac, (u8 *)&ack, sizeof(ack));
}

// ---------- display helpers ----------
uint8_t confPct(uint8_t conf) {
  return (uint8_t)((conf * 100u + 127u) / 255u);
}

const char *nameFor(uint8_t id) {
  static char custom[12];
  if (id < CLS_COUNT) return CLASS_NAMES[id];
  if (id >= 16) return "unknown";
  snprintf(custom, sizeof(custom), "custom %u", (unsigned)id);
  return custom;
}

// OLED only shows ASCII; CLASS_NAMES uses underscores, the screen reads better with spaces.
void upper(char *dst, const char *src, size_t n) {
  size_t i = 0;
  for (; src[i] && i < n - 1; i++) {
    char c = src[i] == '_' ? ' ' : src[i];
    dst[i] = (c >= 'a' && c <= 'z') ? (char)(c - 'a' + 'A') : c;
  }
  dst[i] = '\0';
}

int16_t centerX(const char *s, uint8_t size) {
  return (int16_t)((OLED_W - (int)strlen(s) * 6 * size) / 2);
}

bool alertActive() {
  return haveAlert && (uint32_t)(millis() - lastAlertMs) < ALERT_HOLD_MS;
}

void drawIdle() {
  char buf[28], name[12];
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print("yamDetector");
  display.setCursor(0, 10);
  display.print("wristband / listening");

  const uint32_t age = millis() - lastRxMs;
  if (rxCount == 0)                    snprintf(buf, sizeof(buf), "no signal yet");
  else if (age >= LINK_TIMEOUT_MS)     snprintf(buf, sizeof(buf), "NO SIGNAL %.1fs", age / 1000.0f);
  else                                 snprintf(buf, sizeof(buf), "link %.1fs ago", age / 1000.0f);
  display.setCursor(0, 24);
  display.print(buf);

  snprintf(buf, sizeof(buf), "rx %lu  alerts %lu", (unsigned long)rxCount, (unsigned long)alertCount);
  display.setCursor(0, 34);
  display.print(buf);

  if (haveAlert) {
    upper(name, nameFor(lastMsg.class_id), sizeof(name));
    snprintf(buf, sizeof(buf), "last %s %u%%", name, (unsigned)confPct(lastMsg.conf));
    display.setCursor(0, 44);
    display.print(buf);
  }

  const uint16_t mv = ESP.getVcc();
  snprintf(buf, sizeof(buf), "%.2f V%s", mv / 1000.0f, mv < LOW_BATT_MV ? "  LOW BAT" : "");
  display.setCursor(0, 54);
  display.print(buf);
}

void drawAlert() {
  char buf[28], name[12];
  upper(name, nameFor(lastMsg.class_id), sizeof(name));

  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(centerX(name, 2), 0);
  display.print(name);

  const int16_t w = (int16_t)((uint16_t)lastMsg.conf * (OLED_W - 2) / 255u);
  display.drawRect(0, 19, OLED_W, 9, SSD1306_WHITE);
  if (w > 0) display.fillRect(1, 20, w, 7, SSD1306_WHITE);

  display.setTextSize(1);
  snprintf(buf, sizeof(buf), "%u%%  seq %u", (unsigned)confPct(lastMsg.conf), (unsigned)lastMsg.seq);
  display.setCursor(0, 32);
  display.print(buf);

  snprintf(buf, sizeof(buf), "%s  %.1f s", lastMsg.type == MSG_TEST ? "TEST" : "alert",
           (millis() - lastAlertMs) / 1000.0f);
  display.setCursor(0, 42);
  display.print(buf);

  snprintf(buf, sizeof(buf), "%.2f V", ESP.getVcc() / 1000.0f);
  display.setCursor(0, 54);
  display.print(buf);
}

// ---------- message handling ----------
void handleMsg() {
  msg_t m;
  uint8_t mac[6];
  memcpy(&m, &inbox, sizeof(m));
  memcpy(mac, inboxMac, 6);
  inboxReady = false;

  rxCount++;
  lastRxMs = millis();
  lastMsg = m;
  dirty = true;

  Serial.printf("[wristband] %s class %u (%s) conf %u%% seq %u\n",
                m.type == MSG_ALERT ? "alert" : m.type == MSG_TEST ? "test" : "heartbeat",
                (unsigned)m.class_id, nameFor(m.class_id), (unsigned)confPct(m.conf),
                (unsigned)m.seq);

  if (m.type == MSG_ALERT || m.type == MSG_TEST) {
    alertCount++;
    lastAlertMs = millis();
    haveAlert = true;
    digitalWrite(PIN_MOTOR, HIGH);
    motorUntil = millis() + MOTOR_MS;
    sendAck(mac, &m);
  } else if (m.type == MSG_HEARTBEAT) {
    sendAck(mac, &m);   // no buzz, but the room unit sees the band is alive
  }
}

// A sagging supply can drop the OLED without resetting the board, and no button can bring it
// back, so this is retried from loop() as well.
bool displayUp() {
  displayOk = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!displayOk) {
    Serial.println("[wristband] no OLED at 0x3C: check SDA GPIO0, SCL GPIO2 and the pull-ups");
    return false;
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 24);
  display.print("yamDetector band");
  display.display();
  dirty = true;
  return true;
}

void setup() {
  pinMode(PIN_MOTOR, OUTPUT);
  digitalWrite(PIN_MOTOR, LOW);
  Serial.begin(115200);
  pinMode(PIN_MOTOR, OUTPUT);   // GPIO3 is UART RX: take it back after Serial.begin()

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  displayUp();

  radioOk = radioUp();
  if (!radioOk) Serial.println("[wristband] ESP-NOW init failed, retrying every 2 s");

  uint8_t mac[6];
  WiFi.macAddress(mac);
  Serial.printf("\n[wristband] MAC %02X:%02X:%02X:%02X:%02X:%02X | channel %d | %u mV\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], ESPNOW_CHANNEL,
                (unsigned)ESP.getVcc());

  digitalWrite(PIN_MOTOR, HIGH);   // self-test: one short buzz tells you the motor wiring works
  delay(150);
  digitalWrite(PIN_MOTOR, LOW);
}

void loop() {
  if (inboxReady) handleMsg();

  if (motorUntil && (int32_t)(millis() - motorUntil) >= 0) {
    motorUntil = 0;
    digitalWrite(PIN_MOTOR, LOW);
  }

  const uint32_t now = millis();

  // No reset button: whatever failed to start gets another chance instead of a dead board.
  if (!radioOk && now - lastRadioTry >= 2000) {
    lastRadioTry = now;
    radioOk = radioUp();
    if (radioOk) Serial.println("[wristband] radio up");
  }
  if (!displayOk && now - lastDisplayTry >= 5000) {
    lastDisplayTry = now;
    displayUp();
  }

  const uint32_t interval = alertActive() ? 250 : 1000;   // the age line ticks on the alert screen
  if (dirty || now - lastRender >= interval) {
    dirty = false;
    lastRender = now;
    if (displayOk) {
      display.clearDisplay();
      if (alertActive()) drawAlert();
      else drawIdle();
      display.display();
    }
  }
}
