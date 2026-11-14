# yamDetector Hardware Pinout & Configuration

This document defines the assumed pinout and wiring connections for both the **Listener Module (ESP32-S3)** and the **Receiver Module (ESP8266-01)**. You can update this file as your physical wiring evolves.

---

## 1. Listener Module (ESP32-S3 + 4 PDM MEMS Microphones)

The ESP32-S3 uses its hardware I2S peripheral in PDM receive mode (I2S_MODE_PDM_RX) to capture audio from 4 PDM MEMS microphones arranged in a square array.

| Component / Signal | ESP32-S3 Pin | Description / Notes |
| :--- | :--- | :--- |
| **PDM Clock (CLK)** | GPIO 4 | Shared clock line connected to all 4 PDM mics |
| **PDM Data 0 (DAT0)** | GPIO 5 | Data input line 0 (Mics 1 & 2 via left/right select) |
| **PDM Data 1 (DAT1)** | GPIO 6 | Data input line 1 (Mics 3 & 4 via left/right select) |
| **VCC (Power)** | 3.3V | Power supply for MEMS microphones |
| **GND (Ground)** | GND | Common ground |

---

## 2. Receiver Module (ESP8266-01 + 4-Pin I2C OLED Display)

The ESP8266-01 receives wireless classification packets from the ESP32-S3 via **ESP-NOW** and displays formatted alerts (e.g., CAR 6:00, ALARM 12:00) onto a standard 4-pin I2C display (such as the SSD1306 128x64 OLED).

| Display Pin | ESP8266-01 Pin | Description / Notes |
| :--- | :--- | :--- |
| **VCC** | 3.3V | **Important:** Must be 3.3V (never connect to 5V) |
| **GND** | GND | Common ground |
| **SCL** | GPIO 2 (TX1 / D4) | I2C Clock Line |
| **SDA** | GPIO 0 (D3) | I2C Data Line |

---

## 3. Wireless Communication (ESP-NOW)
* **Protocol**: ESP-NOW (connectionless peer-to-peer wireless communication).
* **Pairing**: Ensure the ESP8266-01 receiver's MAC address is updated in listener/listener_esp32s3.ino (or vice-versa).
