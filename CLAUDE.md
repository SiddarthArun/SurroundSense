# Sound-awareness wristband (yamDetector)

Hackathon project, two people. A wearable harness (XIAO ESP32S3 Sense, 4 PDM mics: two on the
chest, two on the back) classifies urgent sounds on-device, works out which side of the wearer
they came from, and radios a wristband (ESP-01 / ESP8266, vibration motor, OLED) over ESP-NOW. The laptop records training data and trains the model. Keep changes small and
demo-focused. README.md is the single user-facing doc; don't add more markdown files.

## Layout
- `config.yaml`, `scripts/`, `train.py`, `export.py`, `src/yamdetector/`: ML pipeline.
  Training reads only `data/processed/manifest.csv` and `config.yaml`.
- `data/ESC-50/`: the dataset (read-only).
- `libraries/protocol/protocol.h`: shared constants, `msg_t` (10 bytes), class list.
  Sketches use `#include <protocol.h>`; the Arduino sketchbook location is the repo root.
- `libraries/microfrontend/`: TensorFlow audio microfrontend C code (vendored; uses the core's kissfft).
- `room_capture/`: harness firmware, the working one. `MODE` picks levels / stream / on-device
  inference (`yam_infer.h`, needs `model.h` from `export.py --firmware-dir room_capture`) plus
  direction (`doa.h`, runs on the same 1 s window from all four mics).
- `bringup/sense_hello/`, `bringup/esp01_hello/`: bring-up sketches (MAC, PSRAM / Vcc, msg_t size).
- `host/recorder.py`: records the harness's `MODE_STREAM` output into `my_recordings_raw/<label>/`.
- `wristband/`: ESP-01 receiver (ESP-NOW in, motor on GPIO3, SSD1306 on GPIO0/GPIO2, ACK back).
  Shows class + direction.

## Hardware facts
- Sense PDM pins: CLK D0 (GPIO1), DATA0 D1 (GPIO2, mics 1-2), DATA1 D3 (GPIO4, mics 3-4).
  Avoid D2 (strapping pin) and D8-D10 (Sense SD slot).
- PDM capture uses the ESP-IDF driver (`driver/i2s_pdm.h`). The Arduino `ESP_I2S` wrapper
  ignores the PDM slot mask, so it can only read data line 0.
- Audio: 16 kHz, int16, 4 channels interleaved. The classifier uses mic 1 (channel 0).
- Harness geometry: mic 1 chest-right, 2 chest-left, 3 back-right, 4 back-left (wearer's right/
  left), a 30 x 30 cm square with the torso inside. Body shadowing and diffraction break
  free-field angle maths, so `doa.h` decides front/behind and right/left from timing sign
  (GCC-PHAT) + level difference (1-6 kHz) instead of solving for an angle. Mic positions live in
  `MIC_*` at the top of `doa.h`.
- MODE_INFER needs PSRAM (OPI): Wi-Fi takes internal RAM first, the tensor arena and the 4-mic
  ring buffer go to PSRAM. The sketch `#error`s without `BOARD_HAS_PSRAM`.
- ESP-NOW on the ESP32 side needs `esp_now_add_peer()` even for broadcast, and the STA kept idle
  (`persistent(false)`, `setAutoReconnect(false)`) so the home channel stays on 1.
- Serial frame (harness -> laptop, native USB): `A5 5A`, type, seq u16 LE, len u16 LE, payload.
- ESP-01: 3.3 V only (2xAA alkaline, no regulator). Motor on GPIO3 via NPN, OLED I2C SDA GPIO0,
  SCL GPIO2 (the OLED's pull-ups are what keep both boot straps high). GPIO3 is UART RX, so the
  band prints on GPIO1 only. Flashed through the ESP32 DevKit V1 (V1 EN tied to GND, TX->TX,
  RX->RX, ESP-01 GPIO0 low while power arrives). The assembled band has no buttons or header,
  so flash mode is GPIO0 low at power-up and restarting means pulling a cell: firmware must
  never need a reset (retry radio/OLED startup in loop, no blocking waits in setup).
- ESP-NOW channel 1 (`ESPNOW_CHANNEL`). The wristband ACKs every message, echoing `seq` and
  `t_ms`. `msg_t.dir` carries `DIR_*` from `protocol.h` (0 = unknown).

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
- Wristband compile (esp8266 core 3.1.x, needs Adafruit SSD1306 + GFX installed):
  `arduino-cli compile --fqbn esp8266:esp8266:generic:eesz=1M --libraries libraries wristband`
- `doa.h` can be tested on the PC with g++ (msys64 ucrt64) by stubbing `dsps_fft2r.h` /
  `dsps_wind_hann.h` with a plain DFT and feeding synthetic 4-mic audio with known delays.
