#!/usr/bin/env python
"""Stage 1b: FSD50K and RAVDESS -> the standardized dataset folder.

Both datasets are "extra sources" next to ESC-50 (stage 1) and our own
recordings (stage 2). They write the same 16 kHz mono 1 s windows and the same
manifest rows, so training never learns where a window came from.

* FSD50K is multi-label: a clip is mapped to one class through
  ``classes.fsd50k_map``, and ``classes.order`` breaks the tie (Screaming beats
  Speech, Music beats nothing). Clips with none of the mapped labels become the
  background class, which is where most of the room-noise diversity comes from.
* RAVDESS is acted speech; every clip becomes ``classes.ravdess_label``.
* Splits follow the source recording: FSD50K by clip id, RAVDESS by actor, so
  no clip or speaker ever straddles train and test.
* Long clips contribute at most ``prepare.max_windows_per_clip`` windows, spread
  evenly over the clip.

Only rows of the sources being rebuilt are replaced; ESC-50 and own-recording
rows in the manifest are left untouched. Re-run it any time.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yamdetector import audio, manifest  # noqa: E402
from yamdetector.config import ConfigError, load_config, set_seed  # noqa: E402

SOURCES = ("fsd50k", "ravdess")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FSD50K / RAVDESS windows for yamDetector (stage 1b)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument(
        "--source",
        choices=(*SOURCES, "all"),
        default="all",
        help="which dataset to rebuild (default: all)",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="only use the first N clips per dataset (smoke test)",
    )
    return parser.parse_args(argv)


def clear_previous_windows(processed_dir: Path, source: str) -> int:
    """Delete wavs from an earlier run so stale windows cannot survive."""
    removed = 0
    for path in processed_dir.glob(f"*/{source}_*.wav"):
        path.unlink()
        removed += 1
    return removed


def read_ground_truth(gt_dir: Path) -> list[dict]:
    """Read dev.csv and eval.csv (whichever exist); both have fname + labels."""
    rows: list[dict] = []
    for name in ("dev.csv", "eval.csv"):
        path = gt_dir / name
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in ("fname", "labels") if column not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(
                    f"error: {path} is missing column(s): {', '.join(missing)}. "
                    "Expected the official FSD50K ground-truth CSV."
                )
            for row in reader:
                if (row.get("fname") or "").strip():
                    rows.append(row)
    return rows


def emit_windows(
    cfg: dict,
    windows: np.ndarray,
    label: str,
    source: str,
    file_id: str,
    file_stem: str,
    split: str,
) -> tuple[list[dict], int]:
    """Write the usable windows of one recording; returns (rows, skipped)."""
    processed_dir = cfg["paths"]["processed_dir"]
    background = cfg["classes"]["background"]
    peak_target = cfg["audio"]["peak_target"]
    max_gain_db = cfg["audio"]["max_gain_db"]
    silence_rms = cfg["audio"]["silence_rms"]
    sample_rate = cfg["audio"]["sample_rate"]

    rows: list[dict] = []
    skipped = 0
    for window_index, window in enumerate(windows):
        level = audio.rms(window)
        if level <= 0.0:
            skipped += 1
            continue
        if label != background and level < silence_rms:
            skipped += 1
            continue
        normalized = audio.peak_normalize(window, peak_target, max_gain_db)
        out_path = processed_dir / label / f"{source}_{file_stem}_w{window_index:03d}.wav"
        audio.write_wav(out_path, normalized, sample_rate)
        rows.append(
            {
                "path": manifest.to_manifest_path(out_path, PROJECT_ROOT),
                "class": label,
                "source": source,
                "source_file_id": file_id,
                "split": split,
            }
        )
    return rows, skipped


def summarize(rows: list[dict], source: str) -> None:
    per_class: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per_class[row["class"]][row["split"]] += 1
    print(f"\n{source} windows per class (before the training cap):")
    print(f"  {'class':<18}{'train':>7}{'val':>7}{'test':>7}{'total':>8}")
    for label in sorted(per_class):
        counts = per_class[label]
        print(
            f"  {label:<18}{counts.get('train', 0):>7}{counts.get('val', 0):>7}"
            f"{counts.get('test', 0):>7}{sum(counts.values()):>8}"
        )


def prepare_fsd50k(cfg: dict, limit: int | None) -> list[dict]:
    audio_dir = cfg["paths"]["fsd50k_audio_dir"]
    gt_dir = cfg["paths"]["fsd50k_ground_truth_dir"]
    if not audio_dir.is_dir():
        raise SystemExit(
            f"error: FSD50K audio folder not found: {audio_dir}\n"
            "Download FSD50K (https://zenodo.org/record/4060432) and set "
            "'paths.fsd50k_audio_dir' / 'paths.fsd50k_ground_truth_dir' in config.yaml."
        )
    if not gt_dir.is_dir():
        raise SystemExit(f"error: FSD50K ground truth folder not found: {gt_dir}")

    source = cfg["prepare"]["fsd50k_source"]
    label_map = cfg["classes"]["fsd50k_map"]
    order = cfg["classes"]["order"]
    background = cfg["classes"]["background"]
    sample_rate = cfg["audio"]["sample_rate"]
    max_windows = cfg["prepare"]["max_windows_per_clip"]

    gt_rows = read_ground_truth(gt_dir)
    if not gt_rows:
        raise SystemExit(f"error: no dev.csv / eval.csv found in {gt_dir}")

    # Splits are assigned over every id in the ground truth, not just the clips
    # that happen to be on disk, so re-downloading missing audio never reshuffles
    # the split. Windows inherit their clip's split.
    assignments = manifest.assign_splits_by_file(
        [row["fname"].strip() for row in gt_rows],
        cfg["splits"]["fsd50k_ratios"],
        int(cfg["splits"]["seed"]),
    )

    if limit:
        gt_rows = gt_rows[:limit]
    print(f"FSD50K: {audio_dir} ({len(gt_rows)} clip(s), at most {max_windows} window(s) each)")

    removed = clear_previous_windows(cfg["paths"]["processed_dir"], source)
    if removed:
        print(f"removed {removed} window(s) from a previous run")

    rows: list[dict] = []
    skipped_empty = 0
    skipped_unreadable = 0
    skipped_unknown = 0
    skipped_absent = 0
    failed: list[tuple[str, str]] = []

    for index, gt in enumerate(gt_rows, start=1):
        fname = gt["fname"].strip()
        clip_path = audio_dir / f"{fname}.wav"
        if not clip_path.is_file():
            # The ground truth ships dev.csv and eval.csv; only the audio that was
            # actually downloaded is used (the split is still assigned over all ids).
            skipped_absent += 1
            continue
        if clip_path.stat().st_size == 0:
            skipped_empty += 1
            continue

        labels = [item.strip() for item in (gt.get("labels") or "").split(",") if item.strip()]
        label = manifest.class_for_labels(labels, label_map, order, background)
        if labels and label == background:
            skipped_unknown += 1  # none of this clip's labels are mapped: pure background
        try:
            mono, _ = audio.load_clip(clip_path, sample_rate, channel_mode="index")
        except audio.AudioError as exc:
            skipped_unreadable += 1
            failed.append((fname, str(exc)))
            continue

        windows = audio.limit_windows(
            audio.slice_windows(mono, cfg["audio"]["window_samples"], cfg["audio"]["hop_samples"]),
            max_windows,
        )
        clip_rows, _ = emit_windows(
            cfg, windows, label, source, fname, fname, assignments[fname]
        )
        rows.extend(clip_rows)

        if index % 250 == 0 or index == len(gt_rows):
            print(f"  {index}/{len(gt_rows)} clips, {len(rows)} window(s) so far")

    if skipped_absent:
        print(f"\n{skipped_absent} clip(s) in the ground truth have no audio on disk, skipped")
    if skipped_empty:
        print(
            f"warning: {skipped_empty} FSD50K clip(s) are 0 bytes and were skipped. "
            "Your FSD50K download is incomplete: re-extract the archive into "
            f"{audio_dir} to use them."
        )
    if failed:
        print(f"warning: {len(failed)} clip(s) could not be converted:")
        for name, reason in failed[:5]:
            print(f"  {name}.wav: {reason}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    if skipped_unknown:
        print(f"{skipped_unknown} clip(s) matched no mapped label and became '{background}'")
    if not rows:
        raise SystemExit("error: no FSD50K windows were produced")
    return rows


def ravdess_group(path: Path) -> str:
    """RAVDESS filenames end in the actor number: 03-01-06-01-02-01-12.wav."""
    parts = path.stem.split("-")
    actor = parts[-1] if parts and parts[-1].isdigit() else ""
    return f"actor{int(actor):02d}" if actor else path.stem


def prepare_ravdess(cfg: dict, limit: int | None) -> list[dict]:
    audio_dir = cfg["paths"]["ravdess_audio_dir"]
    if not audio_dir.is_dir():
        raise SystemExit(
            f"error: RAVDESS folder not found: {audio_dir}\n"
            "Download RAVDESS (https://zenodo.org/record/1188976, Audio_Speech_Actors_01-24.zip), "
            "extract it so the actors sit directly under 'paths.ravdess_audio_dir'."
        )

    source = cfg["prepare"]["ravdess_source"]
    label = cfg["classes"]["ravdess_label"]
    sample_rate = cfg["audio"]["sample_rate"]
    max_windows = cfg["prepare"]["max_windows_per_clip"]

    clips = sorted(path for path in audio_dir.rglob("*.wav") if path.is_file())
    if limit:
        clips = clips[:limit]
    if not clips:
        raise SystemExit(f"error: no .wav files under {audio_dir}")
    print(f"RAVDESS: {audio_dir} ({len(clips)} clip(s) -> '{label}')")

    # Split by actor: the same speaker must not be in train and test.
    assignments = manifest.assign_splits_by_file(
        [ravdess_group(path) for path in clips],
        cfg["splits"]["ravdess_ratios"],
        int(cfg["splits"]["seed"]),
    )

    removed = clear_previous_windows(cfg["paths"]["processed_dir"], source)
    if removed:
        print(f"removed {removed} window(s) from a previous run")

    rows: list[dict] = []
    skipped_empty = 0
    failed: list[tuple[str, str]] = []
    for index, clip_path in enumerate(clips, start=1):
        if clip_path.stat().st_size == 0:
            skipped_empty += 1
            continue
        try:
            mono, _ = audio.load_clip(clip_path, sample_rate, channel_mode="index")
        except audio.AudioError as exc:
            failed.append((clip_path.name, str(exc)))
            continue
        windows = audio.limit_windows(
            audio.slice_windows(mono, cfg["audio"]["window_samples"], cfg["audio"]["hop_samples"]),
            max_windows,
        )
        group = ravdess_group(clip_path)
        clip_rows, _ = emit_windows(
            cfg, windows, label, source, group, clip_path.stem, assignments[group]
        )
        rows.extend(clip_rows)
        if index % 250 == 0 or index == len(clips):
            print(f"  {index}/{len(clips)} clips, {len(rows)} window(s) so far")

    if skipped_empty:
        print(f"\nwarning: {skipped_empty} RAVDESS clip(s) are 0 bytes and were skipped")
    if failed:
        print(f"warning: {len(failed)} clip(s) could not be converted:")
        for name, reason in failed[:5]:
            print(f"  {name}: {reason}")
    if not rows:
        raise SystemExit("error: no RAVDESS windows were produced")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    set_seed(int(cfg["splits"]["seed"]))
    sources = SOURCES if args.source == "all" else (args.source,)

    try:
        new_rows: list[dict] = []
        if "fsd50k" in sources:
            fsd50k_rows = prepare_fsd50k(cfg, args.limit_clips)
            summarize(fsd50k_rows, cfg["prepare"]["fsd50k_source"])
            new_rows.extend(fsd50k_rows)
        if "ravdess" in sources:
            ravdess_rows = prepare_ravdess(cfg, args.limit_clips)
            summarize(ravdess_rows, cfg["prepare"]["ravdess_source"])
            new_rows.extend(ravdess_rows)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    rebuilt = {cfg["prepare"][f"{source}_source"] for source in sources}
    manifest_path = cfg["paths"]["manifest_csv"]
    kept_rows = [row for row in manifest.read_manifest(manifest_path) if row["source"] not in rebuilt]
    written = manifest.write_manifest(manifest_path, kept_rows + new_rows)
    print(f"\nmanifest: {manifest_path} ({written} row(s), {len(kept_rows)} kept from other sources)")
    print("next: python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
