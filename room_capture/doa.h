// doa.h: which side of the wearer a sound came from. Included from classifier.h (MODE_INFER),
// after WIN_SAMPLES. Needs all four mics (PDM_LINES 2).
//
// The mics are on a harness: mics 1-2 on the chest, mics 3-4 on the back, a 30 x 30 cm square
// with the torso in the middle. Sound bends around the body, so the free-field "solve for an
// exact angle" maths doesn't hold. Instead this answers two questions, each from two cues:
//
//   front or behind?  chest mics vs back mics      right or left?  right mics vs left mics
//
//   timing: which side heard it first, from GCC-PHAT on two mic pairs per question. The nearer
//           mic always hears it first, body or not. Sharp on knocks and horns. On a pure tone the
//           30 cm spacing makes it ambiguous, and the timing score then drops to about 0 on its own.
//   level:  which side is louder, 1-6 kHz. The torso shadows the far side by several dB, and
//           this keeps working on pure tones such as alarm beeps.
//
// Each question gets a score from -1 to +1 (positive = front / right). The stronger question
// decides, and a weak answer is reported as unknown rather than guessed. Only the loudest 32 ms
// frames of the window are used, so the quiet part of the window doesn't dilute the result.

#pragma once
#include <math.h>
#include <string.h>
#include "dsps_fft2r.h"
#include "dsps_wind_hann.h"

// ---------- settings ----------
#define DOA_SECTORS    4        // 4: front / right / behind / left. 8 adds the diagonals.
#define DOA_PRINT      1        // 1: a "dir" line with every cue for each loud window (tuning)
#define DOA_MIN_SCORE  0.20f    // weaker answers are reported as unknown
#define DOA_MIN_DBFS   -50.0f   // quieter windows aren't localized
#define DOA_W_TIMING   0.5f     // weight of the timing cue; the level cue gets the rest
#define DOA_TIME_FULL  0.30f    // a timing lead this clear counts as a full-strength vote
#define DOA_LEVEL_FULL 6.0f     // a level difference this big (dB) counts as a full-strength vote

// Channel = mic number - 1 (room_capture's SLOT_FOR_MIC order). Where each mic is worn:
constexpr int MIC_FR = 0;   // mic 1: chest, right
constexpr int MIC_FL = 1;   // mic 2: chest, left
constexpr int MIC_BR = 2;   // mic 3: back, right
constexpr int MIC_BL = 3;   // mic 4: back, left
// Level trim per mic in dB, if one mic reads louder or quieter than the others in MODE_LEVELS.
const float MIC_TRIM_DB[4] = {0.0f, 0.0f, 0.0f, 0.0f};

constexpr int DOA_N          = 512;   // 32 ms FFT frames
constexpr int DOA_HOP        = 256;
constexpr int DOA_FRAMES     = 1 + (WIN_SAMPLES - DOA_N) / DOA_HOP;   // frames in the 1 s window
constexpr int DOA_MAX_FRAMES = 16;    // loudest frames used, about 0.25 s of sound
constexpr float DOA_FRAME_DB = 15.0f; // frames this far below the loudest are skipped
constexpr int DOA_MAX_LAG    = 24;    // samples: 1.5 ms = 51 cm of path, enough for 30 cm plus
                                      // the detour around the body
constexpr int K_TIME_LO  = 300  * DOA_N / AUDIO_SAMPLE_RATE;   // timing band 300-5000 Hz
constexpr int K_TIME_HI  = 5000 * DOA_N / AUDIO_SAMPLE_RATE;
constexpr int K_TIME_N   = K_TIME_HI - K_TIME_LO + 1;
constexpr int K_LEVEL_LO = 1000 * DOA_N / AUDIO_SAMPLE_RATE;   // level band 1-6 kHz (body shadow)
constexpr int K_LEVEL_HI = 6000 * DOA_N / AUDIO_SAMPLE_RATE;

// Two pairs per question, as {a, b}: a positive lag means mic a heard the sound first.
const uint8_t DOA_PAIRS[2][2][2] = {
  {{MIC_FR, MIC_BR}, {MIC_FL, MIC_BL}},   // [0] front (+) vs behind (-)
  {{MIC_FR, MIC_FL}, {MIC_BR, MIC_BL}},   // [1] right (+) vs left (-)
};

struct DoaResult {
  bool     loud;         // false: window too quiet, nothing localized
  uint8_t  dir;          // DIR_* from protocol.h
  float    azimuth;      // degrees, 0 = front, 90 = right. Rough: the body bends the geometry
  float    score[2];     // per question, -1..+1: [0] front/behind, [1] right/left
  float    timing[2];    // timing cue per question, -1..+1
  float    levelDb[2];   // level cue per question, dB
  uint32_t ms;           // time the estimate took
};

bool  doaReady = false;
float doaFftSign = 1.0f;              // +1 if the FFT uses e^(-j...), measured in doaSetup()
float doaWin[DOA_N];                  // Hann window
float doaCos[DOA_N];                  // cos(2 pi m / N); sin is read a quarter turn later
float doaX[4][2 * DOA_N];             // per-mic spectrum of the current frame (re, im)
float doaCross[2][2 * K_TIME_N];      // PHAT cross-spectrum per question, summed over pairs and frames

inline float doaClamp(float v) { return fmaxf(-1.0f, fminf(1.0f, v)); }

bool doaSetup() {
  if (CH != 4) return false;   // needs all four mics
  if (dsps_fft2r_init_fc32(nullptr, DOA_N) != ESP_OK) return false;
  dsps_wind_hann_f32(doaWin, DOA_N);
  for (int m = 0; m < DOA_N; m++) doaCos[m] = cosf(2.0f * (float)M_PI * m / DOA_N);

  // Which way the FFT turns decides which mic "leads", so measure it rather than assume it:
  // an impulse at n = 1 gives X[1] = e^(-j 2 pi / N) under the usual convention (imaginary < 0).
  memset(doaX[0], 0, sizeof(doaX[0]));
  doaX[0][2] = 1.0f;
  dsps_fft2r_fc32(doaX[0], DOA_N);
  dsps_bit_rev_fc32(doaX[0], DOA_N);
  doaFftSign = doaX[0][3] < 0.0f ? 1.0f : -1.0f;

  doaReady = true;
  return true;
}

// Sum of squares over all four mics for one frame. ring is WIN_SAMPLES frames x CH channels,
// oldest frame at head.
float doaFrameEnergy(const int16_t *ring, int head, int start) {
  float e = 0.0f;
  for (int i = 0; i < DOA_N; i++) {
    const int16_t *s = &ring[((head + start + i) % WIN_SAMPLES) * CH];
    for (int m = 0; m < 4; m++) e += (float)s[m] * (float)s[m];
  }
  return e;
}

// Where did the sound in this window come from? Fills r; r->dir stays DIR_UNKNOWN when the
// window is too quiet or the cues are too weak or disagree.
void doaEstimate(const int16_t *ring, int head, DoaResult *r) {
  const uint32_t t0 = millis();
  *r = DoaResult{};
  r->dir = DIR_UNKNOWN;
  if (!doaReady) return;

  // 1. Pick the loudest frames.
  float energy[DOA_FRAMES];
  float maxE = 0.0f;
  for (int f = 0; f < DOA_FRAMES; f++) {
    energy[f] = doaFrameEnergy(ring, head, f * DOA_HOP);
    maxE = fmaxf(maxE, energy[f]);
  }
  const float peakRms = sqrtf(maxE / (DOA_N * 4.0f)) / 32768.0f;
  if (peakRms <= 0.0f || 20.0f * log10f(peakRms) < DOA_MIN_DBFS) {
    r->ms = millis() - t0;
    return;
  }
  r->loud = true;
  const float minE = maxE * powf(10.0f, -DOA_FRAME_DB / 10.0f);
  int picked[DOA_MAX_FRAMES];
  int nPicked = 0;
  while (nPicked < DOA_MAX_FRAMES) {
    int best = -1;
    for (int f = 0; f < DOA_FRAMES; f++) {
      if (energy[f] >= minE && (best < 0 || energy[f] > energy[best])) best = f;
    }
    if (best < 0) break;
    picked[nPicked++] = best;
    energy[best] = -1.0f;   // taken
  }

  // 2. Spectrum of each mic per frame: band energy for the level cue, and the phase-only
  //    (PHAT) cross-spectrum of each pair for the timing cue, so loud bins don't dominate.
  memset(doaCross, 0, sizeof(doaCross));
  double bandE[4] = {0.0, 0.0, 0.0, 0.0};
  for (int p = 0; p < nPicked; p++) {
    const int start = picked[p] * DOA_HOP;
    for (int i = 0; i < DOA_N; i++) {
      const int16_t *s = &ring[((head + start + i) % WIN_SAMPLES) * CH];
      const float w = doaWin[i] / 32768.0f;
      for (int m = 0; m < 4; m++) {
        doaX[m][2 * i]     = (float)s[m] * w;
        doaX[m][2 * i + 1] = 0.0f;
      }
    }
    for (int m = 0; m < 4; m++) {
      dsps_fft2r_fc32(doaX[m], DOA_N);
      dsps_bit_rev_fc32(doaX[m], DOA_N);
      for (int k = K_LEVEL_LO; k <= K_LEVEL_HI; k++) {
        const float re = doaX[m][2 * k], im = doaX[m][2 * k + 1];
        bandE[m] += (double)re * re + (double)im * im;
      }
    }
    for (int q = 0; q < 2; q++) {
      for (int pr = 0; pr < 2; pr++) {
        const float *a = doaX[DOA_PAIRS[q][pr][0]];
        const float *b = doaX[DOA_PAIRS[q][pr][1]];
        for (int k = K_TIME_LO; k <= K_TIME_HI; k++) {
          const float re = a[2 * k] * b[2 * k] + a[2 * k + 1] * b[2 * k + 1];       // a * conj(b)
          const float im = a[2 * k + 1] * b[2 * k] - a[2 * k] * b[2 * k + 1];
          const float mag = sqrtf(re * re + im * im) + 1e-20f;
          const int j = 2 * (k - K_TIME_LO);
          doaCross[q][j]     += re / mag;
          doaCross[q][j + 1] += im / mag;
        }
      }
    }
  }

  // 3. Timing cue: correlation at every lag up to DOA_MAX_LAG either way. Compare the best
  //    "a first" lag with the best "b first" lag. A clean onset makes one side win clearly; a
  //    pure tone repeats every period, both sides score alike, and the cue comes out near 0.
  const float terms = (float)nPicked * 2.0f * K_TIME_N;   // a perfectly coherent source peaks here
  for (int q = 0; q < 2; q++) {
    float bestPos = -1e30f, bestNeg = -1e30f;
    for (int lag = -DOA_MAX_LAG; lag <= DOA_MAX_LAG; lag++) {
      if (lag == 0) continue;   // dead centre favours neither side
      float v = 0.0f;
      for (int k = K_TIME_LO; k <= K_TIME_HI; k++) {
        const int m = ((k * lag) % DOA_N + DOA_N) % DOA_N;
        const float c = doaCos[m];
        const float s = doaCos[(m + 3 * DOA_N / 4) % DOA_N];   // sin(2 pi m / N)
        const int j = 2 * (k - K_TIME_LO);
        v += doaCross[q][j] * c + doaFftSign * doaCross[q][j + 1] * s;
      }
      if (lag > 0) bestPos = fmaxf(bestPos, v);
      else         bestNeg = fmaxf(bestNeg, v);
    }
    r->timing[q] = doaClamp((bestPos - bestNeg) / terms / DOA_TIME_FULL);
  }

  // 4. Level cue: the torso shadows the far side.
  for (int m = 0; m < 4; m++) bandE[m] *= pow(10.0, MIC_TRIM_DB[m] / 10.0);
  const double tiny = 1e-30;
  r->levelDb[0] = (float)(10.0 * log10((bandE[MIC_FR] + bandE[MIC_FL] + tiny) / (bandE[MIC_BR] + bandE[MIC_BL] + tiny)));
  r->levelDb[1] = (float)(10.0 * log10((bandE[MIC_FR] + bandE[MIC_BR] + tiny) / (bandE[MIC_FL] + bandE[MIC_BL] + tiny)));

  // 5. Combine, and name the direction only when one question has a clear answer.
  for (int q = 0; q < 2; q++) {
    r->score[q] = DOA_W_TIMING * r->timing[q] + (1.0f - DOA_W_TIMING) * doaClamp(r->levelDb[q] / DOA_LEVEL_FULL);
  }
  float az = atan2f(r->score[1], r->score[0]) * 180.0f / (float)M_PI;
  if (az < 0.0f) az += 360.0f;
  r->azimuth = az;
  if (fmaxf(fabsf(r->score[0]), fabsf(r->score[1])) >= DOA_MIN_SCORE) {
    const float step = 360.0f / DOA_SECTORS;
    const int sector = (int)lroundf(az / step) % DOA_SECTORS;
    r->dir = (uint8_t)(DIR_FRONT + sector * (8 / DOA_SECTORS));
  }
  r->ms = millis() - t0;
}

void doaPrint(const DoaResult &r) {
  if (!r.loud) return;
  Serial.printf("[SurroundSense] dir %-8s az %3.0f | front/behind %+.2f (time %+.2f, level %+5.1f dB)"
                " | right/left %+.2f (time %+.2f, level %+5.1f dB) | %lu ms\n",
                DIR_NAMES[r.dir], r.azimuth, r.score[0], r.timing[0], r.levelDb[0],
                r.score[1], r.timing[1], r.levelDb[1], (unsigned long)r.ms);
}
