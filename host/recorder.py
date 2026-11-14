#!/usr/bin/env python3
"""Record the room unit's mic stream to a WAV file (build plan Step 1).

Setup:   pip install -r requirements.txt     (needs pyserial and numpy)
Usage:   python host/recorder.py --port COM11 --label knock --seconds 10
         python host/recorder.py --port COM11 --seconds 0        # until Ctrl+C
         python host/recorder.py --port COM11 --channels 2       # while only mics 1-2 are wired

The Sense must run room_capture with TEXT_MODE 0, and the Arduino Serial Monitor must be closed.
Saves my_recordings_raw/<label>/<label>_<timestamp>.wav at the repo root (16 kHz, 16-bit,
one channel per mic, mic 1 = channel 0), ready for scripts/convert_recordings.py.
"""
import argparse
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "my_recordings_raw"   # where scripts/convert_recordings.py looks

MAGIC = b"\xA5\x5A"
HEADER_LEN = 7
FRAME_TYPE_AUDIO = 0x01
SAMPLE_RATE = 16000
FRAMES_PER_BLOCK = 256


class FrameParser:
    """Turns the raw byte stream into (seq, samples[frames, channels]) blocks."""

    def __init__(self, channels):
        self.channels = channels
        self.expected_len = FRAMES_PER_BLOCK * channels * 2
        self.buf = bytearray()
        self.resyncs = 0

    def feed(self, data):
        self.buf += data
        blocks = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                del self.buf[:-1]            # keep a possible first magic byte
                break
            if i > 0:
                del self.buf[:i]
                self.resyncs += 1
            if len(self.buf) < HEADER_LEN:
                break
            ftype = self.buf[2]
            seq = self.buf[3] | (self.buf[4] << 8)
            length = self.buf[5] | (self.buf[6] << 8)
            if ftype != FRAME_TYPE_AUDIO or length == 0 or length > 16384:
                del self.buf[:2]             # magic bytes inside audio; skip them
                self.resyncs += 1
                continue
            if length != self.expected_len:
                if length % (FRAMES_PER_BLOCK * 2) == 0:
                    sent = length // (FRAMES_PER_BLOCK * 2)
                    sys.exit(f"\nThe firmware is sending {sent} channels. "
                             f"Run again with --channels {sent}.")
                del self.buf[:2]
                self.resyncs += 1
                continue
            if len(self.buf) < HEADER_LEN + length:
                break
            payload = bytes(self.buf[HEADER_LEN:HEADER_LEN + length])
            del self.buf[:HEADER_LEN + length]
            samples = np.frombuffer(payload, dtype="<i2").reshape(-1, self.channels)
            blocks.append((seq, samples))
        return blocks


def open_port(port):
    ser = serial.serial_for_url(port, do_not_open=True)
    ser.baudrate = 115200          # ignored by native USB, needed by pyserial
    ser.timeout = 0.5
    ser.rts = False                # RTS can reset the ESP32-S3 over USB
    ser.dtr = True
    ser.open()
    ser.reset_input_buffer()
    return ser


def strongest_frequency(x):
    x = x.astype(np.float64) * np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x))
    spectrum[:5] = 0                # ignore what is left of DC
    return np.argmax(spectrum) * SAMPLE_RATE / len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="e.g. COM11")
    ap.add_argument("--seconds", type=float, default=10.0, help="length to record; 0 = until Ctrl+C")
    ap.add_argument("--label", default="test", help="class name, used for the folder and file name")
    ap.add_argument("--channels", type=int, default=4, help="4 with all mics, 2 with mics 1-2 only")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output folder (default: my_recordings_raw/ at the repo root)")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds of stream to discard after connecting (stale buffered frames)")
    args = ap.parse_args()

    parser = FrameParser(args.channels)
    target = int(args.seconds * SAMPLE_RATE) if args.seconds > 0 else None
    chunks, meter = [], []
    total = drops = 0
    drop_times = []          # (seconds into the recording, blocks missing)
    last_seq = None
    start = last_print = time.time()

    ser = open_port(args.port)
    opened = time.time()
    print(f"Recording '{args.label}' from {args.port}, {args.channels} ch. Ctrl+C stops early.")
    try:
        while target is None or total < target:
            for seq, samples in parser.feed(ser.read(4096)):
                if time.time() - opened < args.warmup:
                    last_seq = seq   # discard stale frames buffered before we connected
                    continue
                if last_seq is not None:
                    gap = (seq - last_seq - 1) & 0xFFFF
                    if gap:
                        drops += gap
                        drop_times.append((total / SAMPLE_RATE, gap))
                last_seq = seq
                chunks.append(samples)
                meter.append(samples)
                total += samples.shape[0]

            now = time.time()
            if total == 0 and now - start > 3:
                print("No audio frames yet: flash room_capture with TEXT_MODE 0 and close the Serial Monitor.")
                start = now
            if meter and now - last_print >= 0.5:
                x = np.concatenate(meter).astype(np.float64)
                meter.clear()
                dbfs = 20 * np.log10(np.sqrt(np.mean(x ** 2, axis=0)) / 32768 + 1e-9)
                levels = "  ".join(f"m{c + 1} {d:6.1f} dB" for c, d in enumerate(dbfs))
                print(f"\r{total / SAMPLE_RATE:6.1f} s | {levels} | dropped blocks {drops}  ", end="", flush=True)
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    print()

    if not chunks:
        print("Nothing recorded.")
        return 1
    audio = np.concatenate(chunks)
    if target is not None:
        audio = audio[:target]

    folder = os.path.join(args.out, args.label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{args.label}_{datetime.now():%Y%m%d_%H%M%S}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(args.channels)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype("<i2").tobytes())

    print(f"Saved {path}: {len(audio) / SAMPLE_RATE:.1f} s, {args.channels} ch, "
          f"{drops} dropped blocks, {parser.resyncs} resyncs")
    if drop_times:
        where = ", ".join(f"{t:.1f} s (x{n})" for t, n in drop_times[:12])
        more = f" and {len(drop_times) - 12} more" if len(drop_times) > 12 else ""
        print(f"Drops at: {where}{more}")
    print(f"Strongest frequency on mic 1: {strongest_frequency(audio[:, 0]):.0f} Hz "
          f"(play a 1 kHz tone: this should read about 1000)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
