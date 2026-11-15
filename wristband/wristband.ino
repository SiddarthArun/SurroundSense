// wristband: ESP-01 (ESP8266) alert receiver. Gets msg_t from the room unit over ESP-NOW,
// buzzes the motor and names the sound on the Adafruit 1.5" 128x128 OLED. Between alerts the
// screen shows what the room unit hears right now (from its once-a-second heartbeat), so the demo
// needs no Serial Monitor. Answers every message with MSG_ACK (echoing seq and t_ms), which the
// room unit prints as "wristband ACK seq N, M ms".
//
// Tools menu: Board "Generic ESP8266 Module" | Upload Speed 115200 | Reset Method "no dtr"
//             Flash Size "1MB" | CPU Frequency 80 MHz
// Libraries: Adafruit SSD1327 (Library Manager; it brings Adafruit GFX and Adafruit BusIO)
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
//   OLED: Adafruit 1.5" 128x128 grayscale (SSD1327, product 4741) over I2C, four wires:
//         Vin -> 3V3, GND -> GND, SDA -> GPIO0, SCL -> GPIO2. Its other pins stay unconnected.
//   Boot straps: GPIO0 and GPIO2 must be high at power-up. The OLED board's pull-ups do that;
//         also fit 10k from GPIO0 to 3V3 and 10k from GPIO2 to 3V3, so the band still boots if
//         the screen is missing or broken. GPIO0 low at power-up = flash mode: the sketch never
//         runs, the screen stays blank and nothing holds the motor off.
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
#include <Adafruit_SSD1327.h>
#include <protocol.h>

// ---------- settings ----------
#define PIN_SDA         0    // GPIO0, boot strap: pull-ups hold it high
#define PIN_SCL         2    // GPIO2, boot strap: must be high at boot too
#define PIN_MOTOR       3    // GPIO3 / RX, motor through an NPN transistor
#define MOTOR_MS        400  // one buzz per alert
#define ALERT_HOLD_MS   6000 // how long the big class name stays up
#define LINK_TIMEOUT_MS 12000 // silence from the room unit (heartbeat is every 1 s)
#define LIVE_STALE_MS   3000  // the "hearing" block blanks when heartbeats stop for this long
#define LOW_BATT_MV     2900 // replace the AAs below this
#define OLED_SIZE       128  // 128 x 128 pixels
#define REDRAW_MS       5000 // idle refresh; otherwise the screen redraws only when something changes
#define STATUS_MS       5000 // serial "alive" line, so a late-opened monitor sees the band; 0 = off

// The SSD1327 has 16 grey levels: 0 is off, 15 is full brightness.
#define GRAY_FULL       SSD1327_WHITE
#define GRAY_DIM        8

// The Adafruit board answers at 0x3D, or at 0x3C with its address jumper changed.
const uint8_t OLED_ADDRS[] = {0x3D, 0x3C};

ADC_MODE(ADC_VCC);   // lets ESP.getVcc() read the supply; the ESP-01 has no ADC pin

// 400 kHz I2C. A full frame is 8 KB, about 0.2 s, so loop() redraws only on a change.
Adafruit_SSD1327 display(OLED_SIZE, OLED_SIZE, &Wire, -1, 400000, 400000);
bool     displayOk = false;
uint32_t lastDisplayTry = 0;

// Filled by the ESP-NOW callback (WiFi task), consumed in loop().
msg_t   inbox;
uint8_t inboxMac[6];
bool    inboxReady = false;

msg_t    lastAlert;            // last alert or test (heartbeats must not overwrite what's shown)
bool     haveAlert = false;
bool     alertShown = false;   // what the screen shows now, so it redraws when the alert expires
msg_t    live;                 // last heartbeat: what the room unit hears right now
bool     haveLive = false;
uint32_t liveMs = 0;
uint32_t rxCount = 0;
uint32_t alertCount = 0;
uint32_t lastRxMs = 0;         // millis() of the last message
uint32_t lastAlertMs = 0;
uint32_t motorUntil = 0;
uint32_t lastRender = 0;
uint32_t lastStatus = 0;
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
  ack.dir      = m->dir;
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
  if (id < CLS_CUSTOM_FIRST || id >= 32) return "unknown";
  snprintf(custom, sizeof(custom), "custom %u", (unsigned)id);
  return custom;
}

const char *dirName(uint8_t dir) {
  return dir < DIR_COUNT ? DIR_NAMES[dir] : DIR_NAMES[DIR_UNKNOWN];
}

bool dirKnown(uint8_t dir) {
  return dir != DIR_UNKNOWN && dir < DIR_COUNT;
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

// Text size 1 is 6 x 8 pixels per character, size 2 is 12 x 16, size 3 is 18 x 24.
void printCentered(const char *s, uint8_t size, int16_t y, uint8_t gray) {
  const int16_t x = (int16_t)((OLED_SIZE - (int)strlen(s) * 6 * size) / 2);
  display.setTextSize(size);
  display.setTextColor(gray);
  display.setCursor(x < 0 ? 0 : x, y);
  display.print(s);
}

bool alertActive() {
  return haveAlert && (uint32_t)(millis() - lastAlertMs) < ALERT_HOLD_MS;
}

void printBattery(int16_t y) {
  char buf[20];
  const uint16_t mv = ESP.getVcc();
  const bool low = mv < LOW_BATT_MV;
  snprintf(buf, sizeof(buf), "%.2f V%s", mv / 1000.0f, low ? "  LOW BAT" : "");
  printCentered(buf, 1, y, low ? GRAY_FULL : GRAY_DIM);
}

// "SIREN 91% BEHIND-R", or without the direction when the room unit couldn't tell.
void classLine(char *buf, size_t n, const msg_t &m) {
  char name[12], dir[12];
  upper(name, nameFor(m.class_id), sizeof(name));
  if (dirKnown(m.dir)) {
    upper(dir, dirName(m.dir), sizeof(dir));
    snprintf(buf, n, "%s %u%% %s", name, (unsigned)confPct(m.conf), dir);
  } else {
    snprintf(buf, n, "%s %u%%", name, (unsigned)confPct(m.conf));
  }
}

// Idle:
//   y   0  Surround Sense
//   y  16  hearing
//   y  28  SPEECH         (size 3, size 2 if long: the room unit's top class right now, any class)
//   y  58  84%  FRONT-R   (direction only when the room unit worked it out)
//   y  80  last alert
//   y  92  SIREN 91% BEHIND-R
//   y 118  3.05 V
// Without fresh heartbeats the hearing block shows WAITING, NO SIGNAL or LISTENING instead.
void drawIdle() {
  char buf[28], name[12], dir[12];
  printCentered("Surround Sense", 1, 0, GRAY_DIM);

  const uint32_t now = millis();
  if (haveLive && now - liveMs < LIVE_STALE_MS) {
    printCentered("hearing", 1, 16, GRAY_DIM);
    upper(name, nameFor(live.class_id), sizeof(name));
    printCentered(name, strlen(name) * 18 <= OLED_SIZE ? 3 : 2, 28, GRAY_FULL);
    if (dirKnown(live.dir)) {
      upper(dir, dirName(live.dir), sizeof(dir));
      snprintf(buf, sizeof(buf), "%u%%  %s", (unsigned)confPct(live.conf), dir);
    } else {
      snprintf(buf, sizeof(buf), "%u%%", (unsigned)confPct(live.conf));
    }
    printCentered(buf, 1, 58, GRAY_FULL);
  } else {
    const uint32_t age = now - lastRxMs;
    const bool lost = rxCount > 0 && age >= LINK_TIMEOUT_MS;
    printCentered(rxCount == 0 ? "WAITING" : lost ? "NO SIGNAL" : "LISTENING", 2, 28, GRAY_FULL);
    if (rxCount == 0) snprintf(buf, sizeof(buf), "turn on the vest");
    else              snprintf(buf, sizeof(buf), "heard %lu s ago", (unsigned long)(age / 1000));
    printCentered(buf, 1, 50, GRAY_DIM);
  }

  if (haveAlert) {
    printCentered("last alert", 1, 80, GRAY_DIM);
    classLine(buf, sizeof(buf), lastAlert);
    printCentered(buf, 1, 92, GRAY_FULL);
  }

  printBattery(118);
}

// Alert, held for ALERT_HOLD_MS:
//   y   8  SIREN          (size 3; size 2 for names longer than 7 letters)
//   y  44  BEHIND-R       (size 2, or "direction unknown")
//   y  72  confidence bar
//   y  90  91%            (size 2; "91% TEST" for the room unit's boot message)
//   y 118  3.05 V
void drawAlert() {
  char buf[16], name[12], dir[12];
  upper(name, nameFor(lastAlert.class_id), sizeof(name));
  printCentered(name, strlen(name) * 18 <= OLED_SIZE ? 3 : 2, 8, GRAY_FULL);

  if (dirKnown(lastAlert.dir)) {
    upper(dir, dirName(lastAlert.dir), sizeof(dir));
    printCentered(dir, 2, 44, GRAY_FULL);
  } else {
    printCentered("direction unknown", 1, 48, GRAY_DIM);
  }

  const int16_t w = (int16_t)((uint16_t)lastAlert.conf * (OLED_SIZE - 2) / 255u);
  display.drawRect(0, 72, OLED_SIZE, 10, GRAY_FULL);
  if (w > 0) display.fillRect(1, 73, w, 8, GRAY_FULL);

  snprintf(buf, sizeof(buf), "%u%%%s", (unsigned)confPct(lastAlert.conf),
           lastAlert.type == MSG_TEST ? " TEST" : "");
  printCentered(buf, 2, 90, GRAY_FULL);

  printBattery(118);
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

  Serial.printf("[wristband] %s class %u (%s) conf %u%% dir %s seq %u\n",
                m.type == MSG_ALERT ? "alert" : m.type == MSG_TEST ? "test" : "heartbeat",
                (unsigned)m.class_id, nameFor(m.class_id), (unsigned)confPct(m.conf),
                dirName(m.dir), (unsigned)m.seq);

  if (m.type == MSG_ALERT || m.type == MSG_TEST) {
    lastAlert = m;
    alertCount++;
    lastAlertMs = millis();
    haveAlert = true;
    digitalWrite(PIN_MOTOR, HIGH);
    motorUntil = millis() + MOTOR_MS;
    dirty = true;
    sendAck(mac, &m);
  } else if (m.type == MSG_HEARTBEAT) {
    // No buzz: a heartbeat only updates the "hearing" block (and the room unit sees the ACK).
    live = m;
    liveMs = lastRxMs;
    haveLive = true;
    if (!alertActive()) dirty = true;   // the alert screen doesn't show it, so no redraw
    sendAck(mac, &m);
  }
}

// Looks for the screen before starting it: the library allocates memory on every begin(), so
// calling begin() every 5 s with no screen attached would slowly use up the heap.
uint8_t findOled() {
  for (uint8_t addr : OLED_ADDRS) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) return addr;
  }
  return 0;
}

// A missing or failed screen is retried from loop(), since no button can restart the band.
bool displayUp() {
  const uint8_t addr = findOled();
  displayOk = addr != 0 && display.begin(addr);
  if (!displayOk) {
    Serial.println("[wristband] no OLED at 0x3D or 0x3C: check Vin, GND, SDA GPIO0, SCL GPIO2");
    return false;
  }
  Serial.printf("[wristband] OLED found at 0x%02X\n", addr);
  display.setTextWrap(false);
  display.clearDisplay();
  printCentered("Surround Sense", 1, 60, GRAY_FULL);
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

#if STATUS_MS
  // The boot line goes out before a replugged Serial Monitor reconnects, and nothing else prints
  // until a message arrives. This line makes "silent" mean "not running", not "nothing heard".
  if (now - lastStatus >= STATUS_MS) {
    lastStatus = now;
    uint8_t mac[6];
    WiFi.macAddress(mac);
    Serial.printf("[wristband] alive %lus | radio %s | oled %s | ch %d | rx %lu | %u mV | MAC %02X:%02X:%02X:%02X:%02X:%02X\n",
                  (unsigned long)(now / 1000), radioOk ? "ok" : "DOWN", displayOk ? "ok" : "MISSING",
                  ESPNOW_CHANNEL, (unsigned long)rxCount, (unsigned)ESP.getVcc(),
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  }
#endif

  // Redraw only on a change (a message, an alert expiring) or every REDRAW_MS: a full frame takes
  // about 0.2 s, and an incoming alert waits for it to finish.
  const bool showAlert = alertActive();
  if (dirty || showAlert != alertShown || now - lastRender >= REDRAW_MS) {
    dirty = false;
    alertShown = showAlert;
    lastRender = now;
    if (displayOk) {
      display.clearDisplay();
      if (showAlert) drawAlert();
      else drawIdle();
      display.display();
    }
  }
}
