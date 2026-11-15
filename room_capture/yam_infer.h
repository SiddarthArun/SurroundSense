// yam_infer.h: on-device classifier for room_capture (MODE_INFER only).
// Included once, from room_capture.ino. MODEL_CLASSES below picks the model header;
// `python export.py --firmware-dir room_capture` writes a new model.h.
//
// Every 0.5 s it takes the last 1 s of mic 1 and runs the same steps as training:
//   peak normalize (0.95, max +20 dB)    -> src/yamdetector/audio.py peak_normalize
//   microfrontend, fresh state per window -> src/yamdetector/features.py (op defaults, PCAN on)
//   divide by 255, quantize round(f / scale + zp) -> export.py evaluate_tflite
// then prints a probability line per window and an Event line on a detection. The same window,
// from all four mics, goes to doa.h for the direction.

#pragma once
#include <math.h>
#include <esp_heap_caps.h>
#include <microfrontend.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------- settings ----------
// Which model runs. Both take the same 49 x 40 features and both report classes by their
// protocol.h id, so the wristband needs no reflash when this changes.
//    5: model_5class.h, the original ESC-50 model: siren, car_horn, alarm, knock (+ background)
//   13: model.h, the FSD50K + RAVDESS retrain (the file export.py writes)
#ifndef MODEL_CLASSES
#define MODEL_CLASSES    5
#endif
#define COOLDOWN_MS      1500    // per class: no repeat Event lines within this time
#define PRINT_PROBS      1       // 1: one line per window with every class the model has

// The only classes that make an Event, and so the only ones that buzz and show on the band.
// Every other class still prints its probability but never fires. Applies to either model; to
// bring one back, add its CLS_* name (protocol.h) here.
const uint8_t ALERT_ON[] = {CLS_SIREN, CLS_KNOCK, CLS_ALARM};

#if MODEL_CLASSES == 5
#include "model_5class.h"
#else
#include "model.h"
#endif

// Min probability for an Event, one per class in CLS_* order (protocol.h). Tune on the harness.
#if MODEL_CLASSES == 5
// The original model is confident (clear sounds score 0.9+), so one bar fits its four classes.
const float CLASS_THRESHOLD[] = {
    0.70f, 0.70f, 0.70f, 0.70f,                 // siren, car_horn, alarm, knock
    1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f,   // glass .. thunder: not in this model
    1.00f, 1.00f, 1.00f,                        // speech, music, background: never fire
};
#else
// This model is trained with label smoothing, so even clear alarms and sirens peak around 0.7
// and need a low bar, while thunder and dog_bark also fire on rumble and noise and need a high
// one. Picked from ESC-50 fold 5 (8 clips per class, 336 background clips).
const float CLASS_THRESHOLD[] = {
    0.40f,  // siren
    0.40f,  // car_horn
    0.40f,  // alarm
    0.70f,  // knock
    0.60f,  // glass
    0.60f,  // scream     (not in ESC-50: set from its false alerts only)
    0.50f,  // baby_cry
    0.80f,  // dog_bark
    0.60f,  // gunshot    (not in ESC-50: set from its false alerts only)
    0.80f,  // thunder
    1.00f,  // speech      silent tier (CLASS_ALERT false): these three never fire,
    1.00f,  // music       so their values are unused
    1.00f,  // background
};
#endif
static_assert(sizeof(CLASS_THRESHOLD) / sizeof(CLASS_THRESHOLD[0]) == CLS_COUNT,
              "CLASS_THRESHOLD needs one entry per class in protocol.h");

constexpr int    WIN_SAMPLES = AUDIO_SAMPLE_RATE;   // 1 s window (audio.window_seconds)
constexpr int    HOP_BLOCKS  = 31;                  // 31 x 16 ms = ~0.5 s between windows
constexpr size_t ARENA_BYTES = 512 * 1024;          // trim using the "arena used" startup line

// Training contract, from config.yaml. Change them together with the model.
constexpr float kPeakTarget = 0.95f;    // audio.peak_target
constexpr float kMaxGain    = 10.0f;    // audio.max_gain_db = 20 dB
constexpr float kValueScale = 255.0f;   // features.value_scale

static_assert(kYamOutputClasses <= CLS_COUNT,
              "the model has more classes than protocol.h: add them to CLASS_NAMES, CLS_* and CLASS_ALERT");

#include "doa.h"   // direction; uses WIN_SAMPLES

// Model output index -> protocol.h class id (CLS_*), filled by name in inferSetup. The radio and
// the wristband only ever see protocol ids, so a model may have fewer classes or another order.
uint8_t modelClass[kYamOutputClasses];

// An alert-tier class (CLASS_ALERT) that is also listed in ALERT_ON.
bool alertOn(int c) {
  if (!CLASS_ALERT[c]) return false;
  for (uint8_t on : ALERT_ON) {
    if (on == c) return true;
  }
  return false;
}

int16_t *ring = nullptr;      // last 1 s of every mic, CH channels interleaved (PSRAM)
int16_t  win[WIN_SAMPLES];    // window being classified: mic 1, oldest sample first
int      ringPos = 0;
bool     ringFull = false;
int      blocksSinceRun = 0;
uint32_t lastEventMs[CLS_COUNT];
uint32_t eventCount = 0;
uint8_t  lastEventClass = CLS_BACKGROUND;   // latest Event, read by room_capture's radio code
uint8_t  lastEventConf  = 0;                // 0-255, the confidence the band displays
uint8_t  lastEventDir   = DIR_UNKNOWN;      // where it came from, from doa.h

FrontendState             frontend;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor             *inputT = nullptr;
TfLiteTensor             *outputT = nullptr;

// Setup error: repeat the message (the Serial Monitor may open late) and blink.
void inferFail(const char *msg) {
  pinMode(LED_BUILTIN, OUTPUT);
  while (true) {
    Serial.printf("[yamDetector] ERROR: %s\n", msg);
    for (int i = 0; i < 10; i++) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(200);
    }
  }
}

void inferSetup() {
  // Same settings TensorFlow's audio_microfrontend uses in features.py: the arguments it
  // passes (30 ms / 20 ms / 40 channels / 125-7500 Hz / scale_shift 6), the rest are the
  // Python wrapper's defaults.
  FrontendConfig cfg;
  FrontendFillConfigWithDefaults(&cfg);
  cfg.window.size_ms                         = 30;
  cfg.window.step_size_ms                    = 20;
  cfg.filterbank.num_channels                = kYamInputChannels;
  cfg.filterbank.lower_band_limit            = 125.0f;
  cfg.filterbank.upper_band_limit            = 7500.0f;
  cfg.noise_reduction.smoothing_bits         = 10;
  cfg.noise_reduction.even_smoothing         = 0.025f;
  cfg.noise_reduction.odd_smoothing          = 0.06f;
  cfg.noise_reduction.min_signal_remaining   = 0.05f;
  cfg.pcan_gain_control.enable_pcan          = 1;
  cfg.pcan_gain_control.strength             = 0.95f;
  cfg.pcan_gain_control.offset               = 80.0f;
  cfg.pcan_gain_control.gain_bits            = 21;
  cfg.log_scale.enable_log                   = 1;
  cfg.log_scale.scale_shift                  = 6;
  if (!FrontendPopulateState(&cfg, &frontend, AUDIO_SAMPLE_RATE)) inferFail("microfrontend setup failed");

  const tflite::Model *model = tflite::GetModel(g_yam_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) inferFail("model.h schema version does not match TFLite Micro");

  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMean();             // GlobalAveragePooling2D
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  const char *where = "internal RAM";
  uint8_t *arena = (uint8_t *)heap_caps_malloc(ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!arena) {
    arena = (uint8_t *)heap_caps_malloc(ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    where = "PSRAM";
  }
  if (!arena) inferFail("no memory for the tensor arena: is Tools > PSRAM set to OPI PSRAM?");

  static tflite::MicroInterpreter interp(model, resolver, arena, ARENA_BYTES);
  interpreter = &interp;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    inferFail("AllocateTensors failed: a missing op (see the line above) or ARENA_BYTES too small");
  }
  inputT = interpreter->input(0);
  outputT = interpreter->output(0);
  if (inputT->type != kTfLiteInt8 || inputT->bytes != (size_t)(kYamInputFrames * kYamInputChannels)) {
    inferFail("model input is not int8 49x40: re-export with export.py");
  }
  if (outputT->type != kTfLiteInt8 || outputT->dims->data[outputT->dims->size - 1] != kYamOutputClasses) {
    inferFail("model output is not int8 with kYamOutputClasses classes");
  }
  for (int i = 0; i < kYamOutputClasses; i++) {
    int id = -1;
    for (int c = 0; c < CLS_COUNT; c++) {
      if (strcmp(kYamLabels[i], CLASS_NAMES[c]) == 0) id = c;
    }
    if (id < 0) inferFail("the model header names a class that CLASS_NAMES in protocol.h doesn't list");
    modelClass[i] = (uint8_t)id;
  }

  ring = (int16_t *)heap_caps_malloc(WIN_SAMPLES * CH * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!ring) inferFail("no memory for the audio ring: is Tools > PSRAM set to OPI PSRAM?");

  Serial.printf("[yamDetector] ready: model %u B, %d classes, arena used %u of %u B (%s)\n",
                g_yam_model_tflite_len, kYamOutputClasses, (unsigned)interpreter->arena_used_bytes(),
                (unsigned)ARENA_BYTES, where);
  Serial.print("[yamDetector] alerts on:");
  int alerts = 0;
  for (int i = 0; i < kYamOutputClasses; i++) {
    const int c = modelClass[i];
    if (alertOn(c)) {
      Serial.printf(" %s %.2f", CLASS_NAMES[c], CLASS_THRESHOLD[c]);
      alerts++;
    }
  }
  Serial.println(alerts ? "" : " none (ALERT_ON lists no class this model has)");
  if (doaSetup()) Serial.printf("[yamDetector] direction: on, %d directions\n", DOA_SECTORS);
  else            Serial.println("[yamDetector] direction: off (needs all 4 mics, PDM_LINES 2)");
}

// "front-R" -> "FRONT-R", for the Event line.
void upperCopy(char *dst, const char *src, size_t n) {
  size_t i = 0;
  for (; src[i] && i < n - 1; i++) dst[i] = (char)toupper(src[i]);
  dst[i] = '\0';
}

// Classify win[]. Returns false on a silent window (nothing to classify).
bool inferRun() {
  const uint32_t t0 = millis();

  int32_t peak = 0;
  for (int i = 0; i < WIN_SAMPLES; i++) {
    const int32_t a = abs((int32_t)win[i]);
    if (a > peak) peak = a;
  }
  if (peak == 0) return false;

  // audio.peak_normalize on x / 32768, then features.py's round(w * 32767).
  const float gain = fminf(kPeakTarget * 32768.0f / (float)peak, kMaxGain);
  const float k = gain * 32767.0f / 32768.0f;
  for (int i = 0; i < WIN_SAMPLES; i++) {
    long v = lrintf((float)win[i] * k);
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    win[i] = (int16_t)v;
  }

  // Microfrontend with fresh state, 16000 samples -> 49 frames x 40 channels, quantized
  // straight into the input tensor.
  FrontendReset(&frontend);
  const float   inScale = inputT->params.scale;
  const int32_t inZero = inputT->params.zero_point;
  int8_t       *in = inputT->data.int8;
  const int16_t *p = win;
  size_t left = WIN_SAMPLES;
  int frames = 0;
  while (left > 0) {
    size_t used = 0;
    const FrontendOutput out = FrontendProcessSamples(&frontend, p, left, &used);
    p += used;
    left -= used;
    if (out.values != nullptr) {
      if (frames < kYamInputFrames) {
        for (size_t c = 0; c < out.size; c++) {
          long q = lrintf((float)out.values[c] / kValueScale / inScale + (float)inZero);
          if (q > 127) q = 127;
          if (q < -128) q = -128;
          in[frames * kYamInputChannels + c] = (int8_t)q;
        }
      }
      frames++;
    }
    if (used == 0 && out.values == nullptr) break;   // safety: no progress
  }
  if (frames != kYamInputFrames) {
    Serial.printf("[yamDetector] ERROR: microfrontend gave %d frames, model wants %d\n", frames, kYamInputFrames);
    return false;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[yamDetector] ERROR: Invoke failed");
    return false;
  }

  // Probabilities by protocol.h class id; classes this model doesn't have stay at 0.
  float prob[CLS_COUNT] = {};
  int best = modelClass[0];
  for (int i = 0; i < kYamOutputClasses; i++) {
    const int c = modelClass[i];
    prob[c] = (outputT->data.int8[i] - outputT->params.zero_point) * outputT->params.scale;
    if (prob[c] > prob[best]) best = c;
  }
  const uint32_t now = millis();

#if PRINT_PROBS
  for (int i = 0; i < kYamOutputClasses; i++) Serial.printf("%s %.2f  ", kYamLabels[i], prob[modelClass[i]]);
  Serial.printf("| gain %4.1f | %lu ms\n", gain, (unsigned long)(now - t0));
#endif

  // Only ALERT_ON classes fire (and never the silent tier: speech, music, background). When the
  // top class is one that's off, nothing fires, even if siren or knock is second.
  const bool isEvent = alertOn(best) && prob[best] >= CLASS_THRESHOLD[best] &&
                       (lastEventMs[best] == 0 || now - lastEventMs[best] >= COOLDOWN_MS);

  // Direction from all four mics over this same window (ring still holds exactly it).
  DoaResult where{};   // dir 0 = DIR_UNKNOWN until estimated
  if (isEvent || DOA_PRINT) {
    doaEstimate(ring, ringPos, &where);
#if DOA_PRINT
    doaPrint(where);
#endif
  }

  if (isEvent) {
    lastEventMs[best] = now;
    eventCount++;
    lastEventClass = (uint8_t)best;
    lastEventConf  = (uint8_t)lroundf(prob[best] * 255.0f);
    lastEventDir   = where.dir;
    char name[24], dir[12];
    upperCopy(name, CLASS_NAMES[best], sizeof(name));
    upperCopy(dir, DIR_NAMES[where.dir], sizeof(dir));
    Serial.printf("[yamDetector] Event: %s | Conf: %d%% | Dir: %s | #%lu\n", name,
                  (int)lroundf(prob[best] * 100.0f), dir, (unsigned long)eventCount);
  }
  return true;
}

// Feed one processed block (mic order, CH channels interleaved). Mic 1 = channel 0.
// The ring keeps every mic: the classifier reads mic 1, doa.h reads all four.
void inferPushBlock(const int16_t *blk, int frames, int channels) {
  for (int f = 0; f < frames; f++) {
    memcpy(&ring[ringPos * CH], &blk[f * channels], CH * sizeof(int16_t));
    if (++ringPos == WIN_SAMPLES) {
      ringPos = 0;
      ringFull = true;
    }
  }
  if (!ringFull || ++blocksSinceRun < HOP_BLOCKS) return;
  blocksSinceRun = 0;
  for (int i = 0; i < WIN_SAMPLES; i++) win[i] = ring[((ringPos + i) % WIN_SAMPLES) * CH];
  inferRun();
}
