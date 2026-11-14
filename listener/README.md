# ESP32-S3 Listener Firmware (listener/)

This folder contains the Arduino / ESP-IDF firmware for the **ESP32-S3 wearable listener module** equipped with **4 PDM MEMS microphones** placed in a square configuration.

---

## Hardware Configuration (ESP32-S3 + 4 PDM Mics)

The ESP32-S3 supports hardware PDM (Pulse Density Modulation) input through its I2S peripheral. 

### Default PDM Pinout (Configurable in Code)
* **CLK (Clock)**: Shared clock line sent to all 4 PDM mics (e.g., GPIO 4).
* **DATA 0**: GPIO 5 
* **DATA 1**: GPIO 6

---

## Pipeline Architecture
1. **PDM Audio Capture**: Samples 4-channel audio at 16 kHz, 16-bit PCM.
2. **Spatial / Direction Estimation**: Calculates direction (e.g., 6:00, 3:00) from the square mic array.
3. **Microfrontend & Inference**: Runs TFLite Micro yamDetector.
4. **Wireless Output (ESP-NOW)**: Broadcasts EVENT @ DIRECTION to the ESP8266-01 receiver.
