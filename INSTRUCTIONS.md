# yamDetector - Listener Prototype Instructions

How to wire, train, flash and run the first **listener-only prototype**: the XIAO ESP32S3 Sense
with 4 PDM MEMS microphones classifies sounds on the device and prints the results over USB serial.

---

## 1. Hardware connections

Wire the 4 PDM mics to the XIAO ESP32S3 Sense (see **hardware.md**):

| Signal | XIAO pin | Mics |
| :--- | :--- | :--- |
| **PDM clock (CLK)** | D0 (GPIO1) | all four, shared |
| **PDM data 0** | D1 (GPIO2) | mics 1 and 2 |
| **PDM data 1** | D3 (GPIO4) | mics 3 and 4 |
| **SEL** | GND: mics 1 and 3, 3V3: mics 2 and 4 | |
| **VCC / GND** | 3V3 / GND | all |

The classifier listens to mic 1.

---

## 2. Train and export the model (laptop)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/prepare_esc50.py
python scripts/convert_recordings.py           # optional: adds your own recordings
python train.py
python export.py --firmware-dir room_capture   # writes artifacts/ and room_capture/model.h
```

`export.py` warns if the class list in `libraries/protocol/protocol.h` doesn't match the model.

---

## 3. Flash the room unit

1. Arduino IDE: install **esp32 by Espressif Systems** (3.x) in Boards Manager.
2. **File > Preferences > Sketchbook location** = this repo's folder, so the IDE finds
   `libraries/protocol` and `libraries/microfrontend`. TFLite Micro ships with the esp32 core.
3. Open `room_capture/room_capture.ino` and set `#define MODE MODE_INFER`.
4. **Tools**: Board **XIAO_ESP32S3**, USB CDC On Boot **Enabled**, PSRAM **OPI PSRAM**.
5. Connect over USB and click **Upload**.

Command line equivalent:
```powershell
arduino-cli compile --upload -p COM11 --fqbn esp32:esp32:XIAO_ESP32S3:CDCOnBoot=default,PSRAM=opi --libraries libraries room_capture
```

First time with new mics? Do the bring-up steps at the top of `room_capture.ino`
(`MODE_LEVELS` tap test, then `MODE_STREAM` with `host/recorder.py`) before `MODE_INFER`.

---

## 4. Watch the results

Open the **Serial Monitor** (Ctrl + Shift + M). With native USB, any baud rate works.

```text
[yamDetector] ready: model 31234 B, arena used 52112 of 163840 B (internal RAM), threshold 0.70
siren 0.01  car_horn 0.02  alarm 0.01  knock 0.03  background 0.93  | gain 10.0 | 38 ms
siren 0.91  car_horn 0.04  alarm 0.03  knock 0.00  background 0.02  | gain  2.1 | 38 ms
[yamDetector] Event: SIREN | Conf: 91% | #1
```

- One probability line per window (1 s of audio, every 0.5 s). `gain` is the peak-normalization
  boost applied, and the last number is the processing time.
- An **Event** line prints when the top class isn't background, its probability is at least
  `DETECT_THRESHOLD`, and the same class hasn't fired in the last `COOLDOWN_MS`.
- Tune these settings at the top of `room_capture/yam_infer.h`. `PRINT_PROBS 0` leaves only
  the Event lines.
- To test, play ESC-50 fold-5 clips (siren, car_horn, clock_alarm, door_wood_knock) from the
  laptop speakers.

Not done yet: the room unit doesn't radio alerts to the wristband yet (ESP-NOW), and it doesn't
estimate direction.
