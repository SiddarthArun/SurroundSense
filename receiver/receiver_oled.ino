/**
 * @file receiver_oled.ino
 * @brief ESP8266-01 Receiver Module for yamDetector
 * 
 * Formats and outputs detection messages from the yamDetector wearable onto
 * a standard 4-pin I2C display (SSD1306 OLED).
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP8266WiFi.h>
#include <espnow.h>

// ==========================================
// 1. CONFIGURABLE PINOUT & DISPLAY SETTINGS
// ==========================================
#define SCREEN_WIDTH  128     // OLED display width, in pixels
#define SCREEN_HEIGHT 64      // OLED display height (change to 32 if using 128x32)
#define OLED_RESET    -1      // Reset pin (-1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C   // I2C address for 128x64 (often 0x3D for some 128x32)

// Configurable I2C Pins for ESP8266-01
// On ESP8266-01: GPIO 0 is D3, GPIO 2 is D4
#define PIN_SDA       0       // GPIO 0 (D3 on NodeMCU mapping)
#define PIN_SCL       2       // GPIO 2 (D4 on NodeMCU mapping)

// ==========================================
// 2. DATA STRUCTURE (Must match Transmitter)
// ==========================================
typedef struct struct_message {
    char event[32];   // e.g., "SIREN", "ALARM", "KNOCK", "HORN", "BACKGROUND"
    float confidence; // Classification confidence (0.0 to 1.0)
    uint32_t count;   // Event counter
} struct_message;

struct_message incomingReadings;

// Display instance
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// State tracking for display timeout / heartbeat
unsigned long lastEventTime = 0;
const unsigned long DISPLAY_HOLD_MS = 5000; // Keep event on screen for 5s before returning to standby
bool eventActive = false;

// Function declarations
void displayStandby();
void displayEvent(const char* eventName, float confidence, uint32_t count);
void OnDataRecv(uint8_t *mac, uint8_t *incomingData, uint8_t len);

void setup() {
    // Initialize Serial for debugging
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[yamDetector] ESP8266-01 Receiver Starting...");

    // Initialize I2C with configurable pinout
    Wire.begin(PIN_SDA, PIN_SCL);

    // Initialize SSD1306 OLED display
    if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println(F("SSD1306 allocation failed. Check wiring and I2C address!"));
        for(;;); // Halt
    }

    // Initial splash screen
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 10);
    display.println(F(" yamDetector Rx "));
    display.setCursor(0, 30);
    display.println(F("Initializing..."));
    display.display();
    delay(1500);

    // Set Wi-Fi to Station mode for ESP-NOW
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();

    // Initialize ESP-NOW
    if (esp_now_init() != 0) {
        Serial.println(F("Error initializing ESP-NOW"));
        display.clearDisplay();
        display.setCursor(0, 20);
        display.println(F("ESP-NOW Init Fail!"));
        display.display();
        return;
    }

    // Register receive callback function
    esp_now_set_self_role(ESP_NOW_ROLE_SLAVE);
    esp_now_register_recv_cb(OnDataRecv);

    Serial.println(F("ESP-NOW Receiver Ready. Listening for events..."));
    displayStandby();
}

void loop() {
    // Check if event display hold time has expired
    if (eventActive && (millis() - lastEventTime > DISPLAY_HOLD_MS)) {
        eventActive = false;
        displayStandby();
    }
    yield();
}

// Callback when data is received via ESP-NOW
void OnDataRecv(uint8_t *mac, uint8_t *incomingData, uint8_t len) {
    if (len == sizeof(incomingReadings)) {
        memcpy(&incomingReadings, incomingData, sizeof(incomingReadings));
        
        Serial.printf("Received Event: %s (Conf: %.2f, #%u)\n", 
                      incomingReadings.event, 
                      incomingReadings.confidence, 
                      incomingReadings.count);

        // Ignore background events for display popups unless wanted, or show all
        if (strcasecmp(incomingReadings.event, "background") != 0) {
            eventActive = true;
            lastEventTime = millis();
            displayEvent(incomingReadings.event, incomingReadings.confidence, incomingReadings.count);
        }
    } else {
        Serial.printf("Received packet of unexpected size: %d (expected %d)\n", len, sizeof(incomingReadings));
    }
}

// Render Standby / Listening Screen
void displayStandby() {
    display.clearDisplay();
    
    // Top status bar
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.print(F("yamDetector | READY"));
    display.drawLine(0, 10, SCREEN_WIDTH, 10, SSD1306_WHITE);

    // Center status
    display.setTextSize(2);
    display.setCursor(10, 25);
    display.print(F("LISTENING"));

    // Animated dots or status footer
    display.setTextSize(1);
    display.setCursor(10, 52);
    display.print(F("Status: Connected"));
    
    display.display();
}

// Render Active Detection Alert Screen
void displayEvent(const char* eventName, float confidence, uint32_t count) {
    display.clearDisplay();

    // Top alert header
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.print(F(">>> ALERT DETECTED <<<"));
    display.drawLine(0, 10, SCREEN_WIDTH, 10, SSD1306_WHITE);

    // Event name in large bold text
    display.setTextSize(2);
    display.setCursor(0, 18);
    display.print(eventName);

    // Confidence and count footer
    display.setTextSize(1);
    display.setCursor(0, 42);
    display.printf("Conf: %d%%", (int)(confidence * 100));
    
    display.setCursor(70, 42);
    display.printf("ID: #%u", count);

    // Bottom progress/hold indicator line
    display.drawLine(0, 58, SCREEN_WIDTH, 58, SSD1306_WHITE);

    display.display();
}
