# yamDetector

Audio event classifier for an ESP32-S3 wearable, trained with TensorFlow/Keras and
exported to int8 TFLite Micro. The device listens for **siren / car_horn / alarm /
knock** and must stay quiet on everything else (**background**).

The training code never touches raw audio. It only reads a standardized dataset
folder, so any audio source (ESC-50, your own recordings, later a second
microphone array) flows through the same pipeline:

```
        ESC-50 (audio/ + meta/esc50.csv)
                   │
     scripts/prepare_esc50.py ──┐
                                ├──► data/processed/<class>/*.wav + manifest.csv
  my_recordings_raw/<class>/ ───┘            │
     scripts/convert_recordings.py           │  (manifest.csv + config.yaml only)
                                             ├──► train.py     -> artifacts/model.keras
                                             └──► export.py    -> model_float/int8.tflite,
                                                                  model.h, labels.txt
```

`manifest.csv` columns: `path,class,source,source_file_id,split`. Splits are
decided per *source recording*, never per window, so windows from one recording
can never leak across train/val/test.

## Layout

```
config.yaml                  every setting (classes, audio, features, splits, seeds)
requirements.txt
scripts/prepare_esc50.py     stage 1: ESC-50 -> data/processed
scripts/convert_recordings.py stage 2: own recordings -> data/processed
train.py                     stage 3: features -> DS-CNN -> metrics
export.py                    stage 4: float/int8 TFLite, model.h, labels.txt
src/yamdetector/             shared library (audio, manifest, features, model, metrics)
tests/                       unit tests for slicing and split logic
audio/, meta/esc50.csv       the ESC-50 dataset committed to this repo (read-only)
```

The ESC-50 files live in `audio/` and `meta/esc50.csv` at the repo root. A
`data/ESC-50/audio` + `data/ESC-50/meta/esc50.csv` layout is detected too;
otherwise set the paths in `config.yaml`. Generated output (`data/processed/`,
`data/features/`, `artifacts/`) is gitignored and reproducible from the commands
below.

## Install

CPU (works on Windows, macOS, Linux):

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# Linux/macOS:           source .venv/bin/activate
pip install -r requirements.txt
```

**GPU (RTX 3050):** TensorFlow has no GPU support on native Windows. Install WSL2
with Ubuntu, then:

```bash
sudo apt install python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install "tensorflow[and-cuda]>=2.16" -r requirements.txt
```

Run the scripts from the WSL2 side of the project (`/mnt/c/Users/.../yamDetector`).
Reading thousands of small files over `/mnt/c` is slow; copying the project into
the WSL2 filesystem is faster. `train.py` prints whether it picked GPU or CPU and
falls back to CPU automatically when no GPU is visible.

## Run order

```bash
# 1. ESC-50 -> data/processed (folds 1-3 train, 4 val, 5 test; background subsampled 4x)
python scripts/prepare_esc50.py

# 2. optional: your own recordings (default channel 0; see options below)
python scripts/convert_recordings.py
python scripts/convert_recordings.py --mix             # average all channels
python scripts/convert_recordings.py --channel 1       # use channel 1
python scripts/convert_recordings.py --raw-dir D:\takes  # other recordings root

# 3. train (features are cached, so the second run starts immediately)
python train.py
python train.py --epochs 10 --batch-size 8            # quick experiment
python train.py --no-augment --force-features         # no augmentation, recompute cache

# 4. export for the firmware
python export.py
```

`--limit N` on stage 1 processes only the first N rows of `esc50.csv` (smoke test;
splits will be incomplete). Each stage is idempotent: re-running stage 1 replaces
its own windows and manifest rows, stage 2 replaces only `source=own` rows, and
stage 3 reuses the cached features.

### Try the trained model

```bash
python demo.py                                   # built-in synthetic sounds (no dataset needed)
python demo.py siren --detail                    # one sound, every class probability
python demo.py --file audio/5-218980-A-30.wav    # a real clip, second by second
python demo.py --file audio/5-218980-A-30.wav --model both   # float vs int8
python demo.py --file take.wav --mix --threshold 0.6         # your own recording
```

`demo.py` loads `artifacts/model_int8.tflite` (the firmware model), shows the top
classes per 1 s window, matches ESC-50 clips against `meta/esc50.csv` to display the
true label and split, and counts detections above `--threshold`. Synthetic presets
are illustrative only - real recordings are the faithful check.

### Own recordings

```
my_recordings_raw/
  knock/take1.wav          any sample rate, any channel count
  doorbell/take2.flac      a new folder name is a new class, no code change
```

Channel handling: channel 0 by default, `--channel N`, or `--mix` to average all
channels. Multi-channel files log a warning. Recordings are resampled to 16 kHz,
peak-normalized, cut into 1 s windows (near-silent windows skipped for non-background
classes) and split ~70/15/15 **across files**, stable per class.

Add new class names to `classes.order` in `config.yaml` if you want to pin their
position in the model output; otherwise new labels follow alphabetically.
`labels.txt` always lists the exact output order after training.

## What training does

* **Features** – TensorFlow `audio_microfrontend` custom op: 40 channels, 30 ms
  window, 20 ms step, 16 kHz, log + PCAN (the same frontend runs on the ESP32-S3).
  One 1 s window becomes `49 x 40 x 1`, with values ~0-2.6 (mean ~1) after dividing
  by `features.value_scale`. Features are cached under `data/features/` as one small
  `.npy` per window and streamed back with `tf.data`, so RAM use does not grow with
  the dataset.
* **Augmentation (train split only)** – background-noise mixing from the
  `background` class at random 0-20 dB SNR, random gain (-6..+6 dB) and random time
  shift (±100 ms). `augment.copies_per_sample` augmented copies are generated once
  and cached; `--no-augment` disables them.
* **Model** – DS-CNN: one 3x3 conv (stride 2) + four depthwise-separable blocks with
  64 filters + global average pooling + softmax, ~21k parameters (printed at start,
  budget 80k). ReLU6 and BatchNorm for a clean int8 conversion.
* **Training** – Adam, sparse cross-entropy with square-root inverse-frequency class
  weights (capped, normalized: full inverse frequency over-suppresses `background`
  and drives the false-positive rate up), early stopping on `val_loss`, seeds fixed
  in `config.yaml`.
* **Reporting** – per-class precision/recall/F1, a confusion matrix on stdout and as
  `artifacts/confusion_matrix.png`, overall accuracy, and the **false-positive rate
  on background test windows** (how often the device fires on ambient sound).

## Outputs

| file | what |
| --- | --- |
| `data/processed/manifest.csv` | the dataset contract (path, class, source, source_file_id, split) |
| `data/features/` | cached microfrontend features (safe to delete) |
| `artifacts/model.keras` | trained Keras model |
| `artifacts/model_float.tflite` | float32 TFLite |
| `artifacts/model_int8.tflite` | fully int8-quantized TFLite |
| `artifacts/model.h` | int8 model as a C array + input scale/zero point/shape constants |
| `artifacts/labels.txt` | class names in model output order |
| `artifacts/confusion_matrix.png` | test confusion matrix |

Export prints the int8 input quantization (`scale`, `zero point`) and warns if int8
accuracy is more than 3 points below float.

## Firmware notes

1. The feature pipeline on the device must match training: 16 kHz int16 audio,
   microfrontend 30 ms window / 20 ms step / 40 channels / 125-7500 Hz, log + PCAN,
   then divide each value by `features.value_scale` (`255` in `config.yaml`, giving
   roughly 0-2.6 per frame).
2. Quantize the normalized feature `f` with the printed constants:
   `int8 = clip(round(f / scale) + zero_point, -128, 127)`. These are also in
   `model.h` as `kYamInputScale` / `kYamInputZeroPoint`.
3. Output index -> class name comes from `labels.txt` (same order as
   `classes.order`, unknown labels appended alphabetically).

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

Covers window slicing (exact windows, padded tails, drops, short files, hop),
level normalization, channel selection and the manifest logic (fold mapping,
per-file split assignment, background subsampling, manifest IO).

## Troubleshooting

* `TensorFlow's audio_microfrontend op is unavailable` – use a full `tensorflow`
  wheel (the op lives in `tensorflow.lite.experimental.microfrontend`), not a
  stripped build.
* `could not find the ESC-50 dataset` – the message lists every location tried;
  set `paths.esc50_audio_dir` / `paths.esc50_meta_csv` or pass `--esc50-root`.
* Stale features after editing `config.yaml` – not an issue: the cache folder is
  keyed by the feature settings. Use `--force-features` to rebuild anyway.
* Out of memory – lower `training.batch_size` and
  `augment.noise_pool_size`; features are already streamed from disk.
