# yamDetector - Listener Prototype Instructions

This guide covers everything needed to set up, build, flash, and run the first **listener-only prototype** for yamDetector using the ESP32-S3 and 4 PDM MEMS microphones.

---

## 1. Hardware Connections (Listener Module)
Wire your 4 PDM MEMS microphones to the ESP32-S3 according to **hardware.md**:

| Component / Signal | ESP32-S3 Pin | Description |
| :--- | :--- | :--- |
| **PDM Clock (CLK)** | GPIO 1 | Shared clock line to all mics |
| **PDM Data 0 (DAT0)** | GPIO 2 | Data input (Mics 1 & 2) |
| **PDM Data 1 (DAT1)** | GPIO 4 | Data input (Mics 3 & 4) |
| **VCC** | 3.3V | Microphone power |
| **GND** | GND | Common ground |

---

## 2. Software Setup & Model Training (Computer Side)

1. **Install Python Dependencies**:
   `ash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   `

2. **Prepare Dataset**:
   `ash
   python scripts/prepare_esc50.py
   `

3. **Train the Model**:
   `ash
   python train.py
   `

4. **Export Model & C Header**:
   `ash
   python export.py
   `
   *This generates rtifacts/model.h, rtifacts/labels.txt, and TFLite model weights ready for embedded deployment.*

---

## 3. Flashing the ESP32-S3 Listener Firmware

1. Open **Arduino IDE** (or PlatformIO).
2. Install the **ESP32 Board Package** (Espressif Systems) via Boards Manager.
3. Select your ESP32-S3 board (e.g., *ESP32S3 Dev Module*).
4. Open listener/listener_esp32s3.ino.
5. Connect your ESP32-S3 to your PC via USB.
6. Click **Upload** to flash the firmware.

---

## 4. Viewing Real-Time Terminal Output

1. In the Arduino IDE, open the **Serial Monitor** (Ctrl + Shift + M).
2. Set the baud rate to **115200**.
3. Reset the ESP32-S3 board using the EN/RST button.
4. You will see real-time inference predictions printed directly to your terminal:
   `	ext
   PDM initialized!
   [yamDetector] Event: CAR | Direction: 6:00 | Conf: 94% | ID: #1
   [yamDetector] Event: CAR | Direction: 6:00 | Conf: 94% | ID: #2
   `
