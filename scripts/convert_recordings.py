#!/usr/bin/env python
"""Stage 2: convert my own recordings into the standardized dataset folder.

Reads ``my_recordings_raw/<class>/<recording>`` with any sample rate and any
channel count, then:

* picks channel 0 by default, one channel with ``--channel N``, or the average
  of all channels with ``--mix`` (a warning is logged for multi-channel files)
* resamples to 16 kHz, peak-normalizes safely, cuts 1 s windows
* writes ``data/processed/<class>/own_*.wav`` and appends rows to manifest.csv

Splits are assigned per *source file* (~70/15/15 across files), so all windows
of one recording land in the same split and nothing leaks. Any folder name
becomes a class, so adding ``doorbell/`` needs no code change.

Own rows are regenerated on each run (idempotent): existing manifest rows with
``source: own`` are replaced, ESC-50 rows are left untouched.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from surroundsense import audio, manifest  # noqa: E402
from surroundsense.config import ConfigError, load_config, set_seed  # noqa: E402

AUDIO_SUFFIXES = {
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".aif",
    ".aiff",
    ".mp3",
    ".m4a",
    ".opus",
    ".w64",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert own recordings to 16 kHz mono 1 s windows (stage 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="recordings root (default: paths.raw_recordings_dir from config.yaml)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--channel",
        type=int,
        default=None,
        help="use this channel index (default: channel 0)",
    )
    group.add_argument(
        "--mix",
        action="store_true",
        help="average all channels instead of using a single one",
    )
    return parser.parse_args(argv)


def find_recordings(raw_dir: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    """Return ``{class: [files]}`` and files found directly in the root (no class)."""
    classes: dict[str, list[Path]] = {}
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in AUDIO_SUFFIXES
        )
        if files:
            classes[class_dir.name.strip()] = files
    loose = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in AUDIO_SUFFIXES
    )
    return classes, loose


def safe_file_id(path: Path, raw_dir: Path) -> str:
    """Unique, filesystem-safe id from the recording's path inside the raw folder."""
    relative = path.relative_to(raw_dir).with_suffix("")
    identifier = "__".join(relative.parts)
    cleaned = "".join(
        character if (character.isalnum() or character in "-_.") else "_" for character in identifier
    )
    return cleaned or "recording"


def clear_previous_own_windows(processed_dir: Path, classes: set[str], source: str) -> int:
    removed = 0
    for class_name in classes:
        class_dir = processed_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in class_dir.glob(f"{source}_*.wav"):
            path.unlink()
            removed += 1
    return removed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.channel is not None and args.channel < 0:
        print("error: --channel must be >= 0", file=sys.stderr)
        return 2

    channel_mode = "mix" if args.mix else "index"
    channel_index = 0 if args.channel is None else args.channel
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else cfg["paths"]["raw_recordings_dir"]
    if not raw_dir.is_dir():
        print(
            f"error: recordings folder not found: {raw_dir}\n"
            "Create it and drop recordings in per-class subfolders, e.g.\n"
            f"  {raw_dir / 'doorbell' / 'take1.wav'}\n"
            "Then re-run: python scripts/convert_recordings.py",
            file=sys.stderr,
        )
        return 2

    classes, loose = find_recordings(raw_dir)
    if loose:
        print(
            f"warning: ignoring {len(loose)} file(s) directly in {raw_dir} "
            f"(put them in a class subfolder, e.g. {raw_dir / 'knock'})",
            file=sys.stderr,
        )
    if not classes:
        print(
            f"error: no class subfolders with audio found in {raw_dir}\n"
            "Expected: my_recordings_raw/<class>/<recording>.wav",
            file=sys.stderr,
        )
        return 2

    sample_rate = cfg["audio"]["sample_rate"]
    window_samples = cfg["audio"]["window_samples"]
    hop_samples = cfg["audio"]["hop_samples"]
    silence_rms = cfg["audio"]["silence_rms"]
    peak_target = cfg["audio"]["peak_target"]
    max_gain_db = cfg["audio"]["max_gain_db"]
    background = cfg["classes"]["background"]
    source = cfg["prepare"]["own_source"]
    processed_dir = cfg["paths"]["processed_dir"]
    manifest_path = cfg["paths"]["manifest_csv"]
    ratios = cfg["splits"]["own_ratios"]
    base_seed = cfg["splits"]["seed"]

    set_seed(base_seed)
    kept_rows = [row for row in manifest.read_manifest(manifest_path) if row["source"] != source]
    removed = clear_previous_own_windows(processed_dir, set(classes), source)
    if removed:
        print(f"removed {removed} window(s) from a previous run")

    print(
        f"{sum(len(files) for files in classes.values())} recording(s) in "
        f"{len(classes)} class folder(s) -> {sample_rate} Hz mono, "
        f"channel mode: {'mix (average all)' if channel_mode == 'mix' else f'channel {channel_index}'}"
    )

    rows: list[dict] = []
    warnings: list[str] = []
    failed: list[tuple[Path, str]] = []
    window_counts: dict[str, int] = {}
    split_files: dict[str, Counter] = defaultdict(Counter)

    for class_name, files in classes.items():
        # Per-class seed keeps a class's split stable when other classes change.
        class_seed = base_seed + (zlib.crc32(class_name.encode("utf-8")) % 10_000)
        file_splits = manifest.assign_splits_by_file([safe_file_id(path, raw_dir) for path in files], ratios, class_seed)
        is_background = class_name == background
        produced = 0

        for path in files:
            file_id = safe_file_id(path, raw_dir)
            split = file_splits.get(file_id, "train")
            try:
                mono, warning = audio.load_clip(path, sample_rate, channel_mode, channel_index)
            except audio.AudioError as exc:
                failed.append((path, str(exc)))
                continue
            if warning and len(warnings) < 10:
                warnings.append(f"{path.relative_to(raw_dir)}: {warning}")

            windows = audio.slice_windows(mono, window_samples, hop_samples)
            if windows.shape[0] == 0:
                failed.append((path, "recording is shorter than one window"))
                continue

            kept = 0
            for window_index, window in enumerate(windows):
                level = audio.rms(window)
                if level <= 0.0:
                    continue
                if not is_background and level < silence_rms:
                    continue
                normalized = audio.peak_normalize(window, peak_target, max_gain_db)
                out_path = processed_dir / class_name / f"{source}_{file_id}_w{window_index:03d}.wav"
                try:
                    audio.write_wav(out_path, normalized, sample_rate)
                except audio.AudioError as exc:
                    failed.append((path, str(exc)))
                    break
                rows.append(
                    {
                        "path": manifest.to_manifest_path(out_path, PROJECT_ROOT),
                        "class": class_name,
                        "source": source,
                        "source_file_id": file_id,
                        "split": split,
                    }
                )
                kept += 1

            produced += kept
            split_files[class_name][split] += 1
            if kept == 0:
                warnings.append(f"{path.relative_to(raw_dir)}: only silent windows, nothing written")

        window_counts[class_name] = produced
        print(f"  {class_name:<18}{len(files):>4} file(s){produced:>6} window(s)")

    if warnings:
        print("\nwarnings:")
        for message in warnings:
            print(f"  {message}")
        multi_channel = sum("channels" in message for message in warnings)
        if multi_channel:
            print("  (use --channel N or --mix to control which channels are used)")

    if failed:
        print(f"\nwarning: {len(failed)} recording(s) failed:")
        for path, reason in failed[:5]:
            print(f"  {path.name}: {reason}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    if not rows:
        print("error: no windows were written - check the recordings and config", file=sys.stderr)
        return 2

    written = manifest.write_manifest(manifest_path, kept_rows + rows)

    print("\nrecordings per split (split is assigned per recording, never per window):")
    print(f"  {'class':<18}{'train':>7}{'val':>7}{'test':>7}")
    for class_name in classes:
        counts = split_files[class_name]
        print(
            f"  {class_name:<18}{counts.get('train', 0):>7}"
            f"{counts.get('val', 0):>7}{counts.get('test', 0):>7}"
        )
    print(f"\nmanifest: {manifest_path} ({written} row(s), {len(rows)} from own recordings)")
    print("next: python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
