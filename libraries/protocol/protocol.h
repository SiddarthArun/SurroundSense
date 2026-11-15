// protocol.h: shared constants and the radio message, used by every board.
// Lives in <repo>/libraries/protocol/ so sketches can #include <protocol.h>.
// Change it here only, then re-upload to every board.

#pragma once
#include <stdint.h>

// Audio stream, room unit -> laptop
#define AUDIO_SAMPLE_RATE      16000
#define AUDIO_CHANNELS         4
#define AUDIO_FRAMES_PER_BLOCK 256

// ESP-NOW radio channel (both boards must match)
#define ESPNOW_CHANNEL         1

// Serial framing, room unit -> laptop
#define FRAME_MAGIC_0    0xA5
#define FRAME_MAGIC_1    0x5A
#define FRAME_TYPE_AUDIO 0x01
#define FRAME_TYPE_EVENT 0x02

// Radio message types
enum : uint8_t { MSG_ALERT = 1, MSG_HEARTBEAT = 2, MSG_ACK = 3, MSG_TEST = 4 };

// Sound classes. MUST match classes.order in config.yaml (= artifacts/labels.txt):
// the room unit sends the model's output index as class_id, and the wristband looks
// the name up here. When a class is added to config.yaml, add it here in the same
// position too; export.py warns if CLASS_NAMES and labels.txt differ.
enum : uint8_t {
  CLS_SIREN        = 0,
  CLS_CAR_HORN     = 1,
  CLS_ALARM        = 2,
  CLS_KNOCK        = 3,
  CLS_BACKGROUND   = 4,
  CLS_COUNT        = 5,
  CLS_CUSTOM_FIRST = 8,   // 8-15 = teach-a-sound custom classes (laptop)
};
static const char *const CLASS_NAMES[CLS_COUNT] = {"siren", "car_horn", "alarm", "knock", "background"};

// Where a sound came from, relative to the wearer (chest = front). The room unit's doa.h sends
// FRONT / RIGHT / BEHIND / LEFT, or all eight with DOA_SECTORS 8.
enum : uint8_t {
  DIR_UNKNOWN      = 0,
  DIR_FRONT        = 1,
  DIR_FRONT_RIGHT  = 2,
  DIR_RIGHT        = 3,
  DIR_BEHIND_RIGHT = 4,
  DIR_BEHIND       = 5,
  DIR_BEHIND_LEFT  = 6,
  DIR_LEFT         = 7,
  DIR_FRONT_LEFT   = 8,
  DIR_COUNT        = 9,
};
// At most 10 characters each, so the wristband can show them in large text.
static const char *const DIR_NAMES[DIR_COUNT] = {"unknown", "front",  "front-R",  "right", "behind-R",
                                                 "behind",  "behind-L", "left",   "front-L"};

// Radio message: 10 bytes, identical layout on ESP32 and ESP8266
typedef struct __attribute__((packed)) {
  uint8_t  type;      // MSG_*
  uint8_t  class_id;  // CLS_* or custom 8-15
  uint8_t  conf;      // confidence 0-255
  uint8_t  dir;       // DIR_*: where the sound came from
  uint16_t seq;       // spots drops and duplicates
  uint32_t t_ms;      // sender's clock; the wrist echoes it in its ACK
} msg_t;

static_assert(sizeof(msg_t) == 10, "msg_t must be 10 bytes on every board");
