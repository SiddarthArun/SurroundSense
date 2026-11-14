# Sound-awareness wristband (yamDetector)

Hackathon project, two people. A room unit (XIAO ESP32S3 Sense, 4 PDM mics) classifies
urgent sounds on-device and will radio a wristband (ESP-01 / ESP8266, vibration motor, OLED)
over ESP-NOW. The laptop records training data and trains the model. Keep changes small and
demo-focused. README.md is the single user-facing doc; don't add more markdown files.

## Layout
- `config.yaml`, `scripts/`, `train.py`, `export.py`, `src/yamdetector/`: ML pipeline.
  Training reads only `data/processed/manifest.csv` and `config.yaml`.
- `data/ESC-50/`: the dataset (read-only).
- `libraries/protocol/protocol.h`: shared constants, `msg_t` (10 bytes), class list.
  Sketches use `#include <protocol.h>`; the Arduino sketchbook location is the repo root.
- `libraries/microfrontend/`: TensorFlow audio microfrontend C code (vendored; uses the core's kissfft).
- `room_capture/`: room unit firmware, the working one. `MODE` picks levels / stream / on-device
  inference (`yam_infer.h`, needs `model.h` from `export.py --firmware-dir room_capture`).
- `bringup/sense_hello/`, `bringup/esp01_hello/`: bring-up sketches (MAC, PSRAM / Vcc, msg_t size).
- `host/recorder.py`: records the room unit's `MODE_STREAM` output into `my_recordings_raw/<label>/`.
- Not built yet: wristband firmware (ESP-NOW receive, motor, OLED), direction estimation.

## Hardware facts
- Sense PDM pins: CLK D0 (GPIO1), DATA0 D1 (GPIO2, mics 1-2), DATA1 D3 (GPIO4, mics 3-4).
  Avoid D2 (strapping pin) and D8-D10 (Sense SD slot).
- PDM capture uses the ESP-IDF driver (`driver/i2s_pdm.h`). The Arduino `ESP_I2S` wrapper
  ignores the PDM slot mask, so it can only read data line 0.
- Audio: 16 kHz, int16, 4 channels interleaved. The classifier uses mic 1 (channel 0).
- Serial frame (room unit -> laptop, native USB): `A5 5A`, type, seq u16 LE, len u16 LE, payload.
- ESP-01: 3.3 V only (2xAA alkaline, no regulator). Motor on GPIO3 via NPN, OLED I2C SDA GPIO0,
  SCL GPIO2. Flashed through the ESP32 DevKit V1 (V1 EN tied to GND, TX->TX, RX->RX,
  ESP-01 GPIO0 low while flashing).
- ESP-NOW channel 1 (`ESPNOW_CHANNEL`). Planned: the wristband ACKs alerts, echoing `seq` and `t_ms`.

## Rules
- Class IDs = model output order = `config.yaml` `classes.order` = `protocol.h` `CLASS_NAMES`.
  Change all of them together; `export.py` warns on a mismatch.
- On-device features must match training: DC block, per-window peak normalize
  (0.95, max +20 dB), microfrontend with identical parameters and fresh state per 1 s
  window, divide by 255, quantize with `kYamInputScale` / `kYamInputZeroPoint` from `model.h`.
- Toolchains: Arduino-ESP32 core 3.x (board XIAO_ESP32S3, USB CDC On Boot enabled,
  PSRAM OPI); ESP8266 core for the ESP-01 (Generic ESP8266 Module, reset method "no dtr").

## Checks
- Python tests: `python -m unittest discover -s tests -t . -v`
- Firmware compile (arduino-cli ships inside the Arduino IDE at
  `C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe`):
  `arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3:CDCOnBoot=default,PSRAM=opi --libraries libraries room_capture`
