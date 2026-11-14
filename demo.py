#!/usr/bin/env python
"""yamDetector demo: hear what the trained model says about a sound.

Three modes, all using the same microfrontend + normalization as training:

    python demo.py                          # every built-in synthetic sound
    python demo.py siren                    # one built-in sound, full bars
    python demo.py --file audio/x.wav       # a real recording, second by second
    python demo.py --file audio/x.wav --model both   # float vs int8 side by side

The built-in sounds are synthesized in numpy, so this runs with no dataset and no
preprocessing step. ESC-50 clips passed with --file are matched against
meta/esc50.csv so the true label (and whether the file was held out) is shown.
"""
from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

# --- keep the console readable: TF/absl/oneDNN chatter off --------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # C++ INFO/WARNING (oneDNN, cpu_feature_guard)
warnings.filterwarnings("ignore")                    # python UserWarnings (TFLite deprecations)

import argparse  # noqa: E402
import csv  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402

logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from yamdetector import audio, features, manifest  # noqa: E402
from yamdetector.config import ConfigError, load_config  # noqa: E402

MODEL_FILES = {
    "int8": ROOT / "artifacts" / "model_int8.tflite",
    "float": ROOT / "artifacts" / "model_float.tflite",
}
LABELS_FILE = ROOT / "artifacts" / "labels.txt"

# Built-in sounds: name -> (generator, intended class or None when the sound is
# not one of the trained classes).
SR = 16000
_t = np.arange(SR) / SR


def _siren():
    freq = 700 + 350 * np.sin(2 * np.pi * 0.7 * _t)          # sweeping tone
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)


def _knock():
    out = np.zeros_like(_t)
    rng = np.random.default_rng(0)
    for start in (0.15, 0.5, 0.8):
        length = int(0.15 * SR)
        tt = np.arange(length) / SR
        hit = np.exp(-tt * 40) * (np.sin(2 * np.pi * 180 * tt) + 0.5 * rng.standard_normal(length))
        offset = int(start * SR)
        out[offset : offset + length] += hit
    return out


def _car_horn():
    return (
        np.sin(2 * np.pi * 400 * _t)
        + 0.8 * np.sin(2 * np.pi * 505 * _t)
        + 0.3 * np.sin(2 * np.pi * 800 * _t)
    )


def _alarm():
    """Clock-alarm style beeps: 2 kHz fundamentals, 2 beeps per second, harmonics."""
    gate = (np.sin(2 * np.pi * 2 * _t) > 0).astype(float)     # 2 beeps per second
    tone = sum(
        amp * np.sin(2 * np.pi * freq * _t) for freq, amp in ((2000, 1.0), (4000, 0.5), (6000, 0.25))
    )
    return tone * gate


def _background():
    rng = np.random.default_rng(1)
    hiss = np.convolve(rng.standard_normal(SR), np.ones(20) / 20, mode="same")
    return 0.3 * hiss + 0.2 * np.sin(2 * np.pi * 50 * _t)     # room hum + hiss


def _doorbell():
    """Two-tone ding-dong: not a trained class, shows the fallback behaviour."""
    out = np.zeros_like(_t)
    for start, freq in ((0.1, 660.0), (0.55, 520.0)):
        length = int(0.45 * SR)
        tt = np.arange(length) / SR
        tone = np.sin(2 * np.pi * freq * tt) + 0.35 * np.sin(4 * np.pi * freq * tt)
        offset = int(start * SR)
        out[offset : offset + length] += tone * np.exp(-tt * 4)
    return out


SOUNDS = {
    "siren": (_siren, "siren"),
    "car_horn": (_car_horn, "car_horn"),
    "alarm": (_alarm, "alarm"),
    "knock": (_knock, "knock"),
    "background": (_background, "background"),
    "doorbell": (_doorbell, None),
}


# ------------------------------------------------------------------------ helpers
@contextmanager
def quiet_stderr(enabled: bool = True):
    """Silence C-level chatter (TFLite prints its XNNPACK delegate to stderr)."""
    if not enabled:
        yield
        return
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def bar(value: float, width: int = 28) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def features_for(windows: np.ndarray, cfg: dict) -> np.ndarray:
    """(n, 16000) waveforms -> (n, 49, 40, 1) microfrontend features."""
    return features.microfrontend(windows, cfg)[..., None].astype(np.float32)


def esc50_lookup(meta_csv: Path) -> dict[str, dict]:
    if not meta_csv.exists():
        return {}
    with meta_csv.open("r", newline="", encoding="utf-8") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle) if row.get("filename")}


def runners_up(labels: list[str], probs: np.ndarray, count: int) -> str:
    order = np.argsort(probs)[::-1][1 : 1 + max(0, count)]
    return ", ".join(f"{labels[i]}={probs[i]:.2f}" for i in order) or "-"


_TF = None


def tensorflow(silent: bool = True):
    """Import TensorFlow once, with its C++ startup chatter suppressed.

    The absl/oneDNN banner is printed by C++ before the Python logger exists, so
    TF_CPP_MIN_LOG_LEVEL alone does not silence it - the import itself is wrapped.
    """
    global _TF
    if _TF is None:
        with quiet_stderr(silent):
            import tensorflow as tensorflow_module

        _TF = tensorflow_module
    return _TF


class Runner:
    """One exported TFLite model; predicts on (n, frames, channels, 1) features."""

    def __init__(self, kind: str, silent: bool = True):
        tf = tensorflow(silent)

        path = MODEL_FILES[kind]
        if not path.exists():
            raise SystemExit(f"error: {path} not found. Run:  python export.py")
        with quiet_stderr(silent):
            self.interpreter = tf.lite.Interpreter(model_path=str(path))
            self.interpreter.allocate_tensors()
        self.kind = kind
        self.path = path
        self.silent = silent
        details = self.interpreter.get_input_details()[0]
        self.input_scale, self.input_zero = details["quantization"]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Features (n, frames, channels, 1) -> probabilities (n, num_classes)."""
        outputs = []
        for start in range(0, len(x), 32):
            chunk = x[start : start + 32]
            details = self.interpreter.get_input_details()[0]
            if details["dtype"] == np.int8:
                if self.input_scale == 0:
                    raise SystemExit("error: int8 model has an uncalibrated input")
                batch = np.clip(
                    np.round(chunk / self.input_scale) + self.input_zero, -128, 127
                ).astype(np.int8)
            else:
                batch = chunk.astype(np.float32)

            if tuple(batch.shape) != tuple(details["shape"]):
                self.interpreter.resize_tensor_input(details["index"], batch.shape, strict=False)
                self.interpreter.allocate_tensors()
                details = self.interpreter.get_input_details()[0]

            with quiet_stderr(self.silent):
                self.interpreter.set_tensor(details["index"], batch)
                self.interpreter.invoke()
            output = self.interpreter.get_output_details()[0]
            scores = self.interpreter.get_tensor(output["index"]).copy()
            if scores.dtype == np.int8:
                scale, zero = output["quantization"]
                scores = (scores.astype(np.float32) - zero) * scale
            outputs.append(scores.astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def describe(self) -> str:
        text = f"{self.kind} TFLite ({self.path.name}, {features.human_bytes(self.path.stat().st_size)})"
        if self.kind == "int8":
            text += f"  input scale={self.input_scale:.6f} zero_point={self.input_zero}"
        return text


# -------------------------------------------------------------------------- modes
def run_presets(args, cfg: dict, runner: Runner, labels: list[str]) -> int:
    names = [args.target] if args.target else list(SOUNDS)
    detail = args.detail or len(names) == 1
    print(f"\npresets ({len(names)}):")
    header = (
        f"  {'sound':<11}{'verdict':<12}{'conf':>6}   "
        f"{'intended':<10}{'match':<7}runners-up"
    )
    print(header + "\n  " + "-" * (len(header) - 2))

    matches = judged = 0
    for name in names:
        generator, intended = SOUNDS[name]
        signal = generator()
        signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.9
        probs = runner.predict(features_for(signal[None, :], cfg))[0]

        best = int(np.argmax(probs))
        if intended is None:
            judged_text, match = "-", "  -  "
        else:
            judged += 1
            ok = labels[best] == intended
            matches += ok
            judged_text = intended
            match = "  ok " if ok else " MISS"
        print(
            f"  {name:<11}{labels[best]:<12}{probs[best]:>6.3f}   "
            f"{judged_text:<10}{match:<7}{runners_up(labels, probs, args.top - 1)}"
        )
        if detail:
            print(f"\n  {name} - all {len(labels)} classes:")
            for index in np.argsort(probs)[::-1]:
                print(f"    {labels[index]:<12}{bar(probs[index])}  {probs[index]:.3f}")
            print()

    if judged:
        print(f"\n  presets matching their intended class: {matches}/{judged}")
        if matches < judged:
            print(
                "  (synthetic tones are not real recordings, so misses here are expected "
                "for some sounds; --file is the faithful check)"
            )
    return 0


def run_file(args, cfg: dict, runner: Runner, labels: list[str], extra: Runner | None) -> int:
    path = Path(args.target).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    sample_rate = cfg["audio"]["sample_rate"]
    background = cfg["classes"]["background"]

    channel_mode = "mix" if args.mix else "index"
    channel = 0 if args.channel is None else args.channel
    try:
        mono, channel_warning = audio.load_clip(path, sample_rate, channel_mode, channel)
        info = audio.read_audio(path)
    except audio.AudioError as exc:
        raise SystemExit(f"error: {exc}")

    windows = audio.slice_windows(
        mono, cfg["audio"]["window_samples"], cfg["audio"]["hop_samples"]
    )
    if windows.shape[0] == 0:
        raise SystemExit(f"error: {path.name} produced no windows")

    print(f"\nfile: {path.name}")
    print(
        f"  {info[1]} Hz, {info[0].shape[1]} channel(s), {info[0].shape[0] / info[1]:.2f} s"
        f" -> {sample_rate} Hz mono, {windows.shape[0]} x 1 s window(s)"
    )
    if channel_warning:
        print(f"  note: {channel_warning}")

    truth_label = None
    meta = esc50_lookup(cfg["paths"]["esc50_meta_csv"])
    if path.name in meta:
        row = meta[path.name]
        truth_label = cfg["classes"]["esc50_map"].get(row["category"], background)
        try:
            split = manifest.split_for_esc50_fold(
                int(str(row["fold"]).strip()), cfg["splits"]["esc50_folds"]
            )
        except (ValueError, manifest.ManifestError):
            split = "unknown"
        print(
            f"  ESC-50: category '{row['category']}' -> label '{truth_label}' "
            f"(fold {row['fold']}, {split} split)"
        )

    x = features_for(windows, cfg)
    probs = runner.predict(x)
    truth_index = labels.index(truth_label) if truth_label in labels else None
    levels = [audio.rms(window) for window in windows]

    print(f"\n  {'sec':>4}  {'verdict':<12}{'conf':>6}   {'truth':<12}runners-up")
    print("  " + "-" * 66)
    correct = 0
    target_events: dict[str, int] = {}
    for index, row in enumerate(probs):
        best = int(np.argmax(row))
        hit = truth_index is not None and best == truth_index
        correct += hit
        if labels[best] != background and row[best] >= args.threshold:
            target_events[labels[best]] = target_events.get(labels[best], 0) + 1
        truth_text = (" ok " if hit else " -- ") if truth_index is not None else "  ? "
        silent = "  [digital silence]" if levels[index] < 1e-6 else ""
        print(
            f"  {index:>4}  {labels[best]:<12}{row[best]:>6.3f}  {truth_text}"
            f"{truth_label or 'unknown':<9} {runners_up(labels, row, args.top - 1)}{silent}"
        )

    total = len(probs)
    print()
    print(f"  mean top-1 confidence: {float(np.max(probs, axis=1).mean()):.3f}")
    if truth_label:
        print(f"  windows matching '{truth_label}': {correct}/{total} ({correct / total:.0%})")
        if truth_label == background:
            print(f"  false-positive windows: {total - correct}/{total} ({(total - correct) / total:.0%})")
    else:
        print("  truth: unknown (not an ESC-50 clip) - judge from the probabilities above")
    events = ", ".join(f"{name} x{count}" for name, count in sorted(target_events.items()))
    print(f"  target detections at conf >= {args.threshold:.2f}: {events or 'none'}")

    if extra is not None:
        other = extra.predict(x)
        agreement = int(np.sum(np.argmax(other, axis=1) == np.argmax(probs, axis=1)))
        print(
            f"\n  {extra.kind} model vs {runner.kind}: agrees on {agreement}/{total} windows, "
            f"max probability difference {float(np.max(np.abs(other - probs))):.3f}"
        )
        if truth_index is not None:
            other_hits = int(np.sum(np.argmax(other, axis=1) == truth_index))
            print(f"  {extra.kind} model matches '{truth_label}' on {other_hits}/{total} windows")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try the yamDetector model on synthetic sounds or real recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets: " + ", ".join(SOUNDS),
    )
    parser.add_argument("target", nargs="?", help="preset name (siren, knock, ...) or audio file path")
    parser.add_argument("--file", dest="file_path", help="audio file to classify (same as a path argument)")
    parser.add_argument(
        "--model", choices=("int8", "float", "both"), default="int8",
        help="which exported model to run (default: int8, the firmware model)",
    )
    parser.add_argument("--top", type=int, default=3, help="classes listed per row (default: 3)")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="confidence needed to count a target detection (default: 0.5)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--channel", type=int, default=None, help="channel to use for multi-channel files")
    group.add_argument("--mix", action="store_true", help="average all channels of the input file")
    parser.add_argument("--detail", action="store_true", help="show every class probability per window")
    parser.add_argument("--verbose", action="store_true", help="do not silence TensorFlow/TFLite logging")
    args = parser.parse_args(argv)

    if args.file_path:
        args.target = args.file_path
    if args.top < 1:
        parser.error("--top must be >= 1")
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    silent = not args.verbose

    tf = tensorflow(silent)
    if silent:
        tf.get_logger().setLevel("ERROR")

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not LABELS_FILE.exists():
        print(f"error: {LABELS_FILE} not found. Run:  python export.py", file=sys.stderr)
        return 2
    labels = LABELS_FILE.read_text(encoding="utf-8").split()

    target = args.target
    is_file = bool(target) and Path(target).is_file()
    if target and not is_file and target not in SOUNDS:
        print(
            f"error: '{target}' is neither a preset nor an existing audio file.\n"
            f"presets: {', '.join(SOUNDS)}",
            file=sys.stderr,
        )
        return 2

    main_kind = "float" if args.model == "float" else "int8"
    runner = Runner(main_kind, silent=silent)
    extra = None
    if args.model == "both":
        extra = Runner("float" if main_kind == "int8" else "int8", silent=silent)

    print("yamDetector demo")
    print(f"  model: {runner.describe()}")
    if extra is not None:
        print(f"  also: {extra.describe()}")
    print("  labels: " + ", ".join(f"{index}={label}" for index, label in enumerate(labels)))
    if not is_file:
        print("  (presets are synthesized in numpy - no dataset needed)")

    if is_file:
        return run_file(args, cfg, runner, labels, extra)
    return run_presets(args, cfg, runner, labels)


if __name__ == "__main__":
    raise SystemExit(main())
