# ESP8266-01 Receiver Module for yamDetector

This folder contains the Arduino firmware for the **ESP8266-01 receiver module** connected to a 4-pin I2C OLED display (such as the SSD1306 128x64 or 128x32). It listens for event detections from the `yamDetector` wearable transmitter and formats them into clean, high-contrast messages on the screen.

---

## Hardware Wiring (4-Pin I2C Display)

The ESP8266-01 has limited exposed GPIO pins (mostly GPIO0 and GPIO2). Connect your 4-pin I2C display as follows:

| Display Pin | ESP8266-01 Pin | Function / Notes |
| :--- | :--- | :--- |
| **VCC** | 3.3V | **Important:** Connect to 3.3V (never 5V). |
| **GND** | GND | Common ground. |
| **SCL** | GPIO 2 (TX1 / D4) | Clock line (configurable in code). |
| **SDA** | GPIO 0 (D3) | Data line (configurable in code). |

---

## Configuration

Open `receiver_oled.ino` in the Arduino IDE. At the top of the file, you can easily configure:
1. **Display Dimensions**: 128x64 or 128x32.
2. **I2C Pinout**: Custom SDA and SCL pin definitions.
3. **Communication Mode**: ESP-NOW (direct wireless from transmitter) or Serial UART (for testing/wired bridge).

## Libraries Required (Arduino IDE Library Manager)
* **ESP8266WiFi** (built into ESP8266 Arduino core)
* **ESPNow** (built into ESP8266 Arduino core)
* **Adafruit GFX Library**
* **Adafruit SSD1306**
