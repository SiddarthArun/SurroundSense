#pragma once
#include <stdint.h>

#define AUDIO_SAMPLE_RATE      16000
#define AUDIO_CHANNELS         4
#define AUDIO_FRAMES_PER_BLOCK 256
#define ESPNOW_CHANNEL         1

#define FRAME_MAGIC_0    0xA5
#define FRAME_MAGIC_1    0x5A
#define FRAME_TYPE_AUDIO 0x01
#define FRAME_TYPE_EVENT 0x02

enum : uint8_t { MSG_ALERT = 1, MSG_HEARTBEAT = 2, MSG_ACK = 3, MSG_TEST = 4 };
enum : uint8_t { CLS_BACKGROUND = 0, CLS_ALARM = 1, CLS_KNOCK = 2, CLS_DOORBELL = 3 };  // 8-15 = custom

typedef struct __attribute__((packed)) {
  uint8_t  type;
  uint8_t  class_id;
  uint8_t  conf;
  uint8_t  zone;
  uint16_t seq;
  uint32_t t_ms;
} msg_t;

static_assert(sizeof(msg_t) == 10, "msg_t must be 10 bytes on every board");