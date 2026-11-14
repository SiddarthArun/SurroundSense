#!/usr/bin/env python
"""Stage 1: turn ESC-50 into the standardized dataset folder.

* maps ESC-50 categories to classes through ``classes.esc50_map`` in config.yaml
  (anything unmapped becomes the background class)
* resamples to 16 kHz mono, cuts 1 s windows, skips near-silent windows for
  target classes only, peak-normalizes and writes ``data/processed/<class>/*.wav``
* subsamples background to ``prepare.background_ratio`` x the average target
  class size
* writes ``data/processed/manifest.csv`` (folds 1-3 train / 4 val / 5 test),
  preserving rows written by stage 2 (own recordings)

The dataset in ``yamDetector/audio`` + ``yamDetector/meta`` is read-only here.
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
from yamdetector.config import ConfigError, esc50_paths, load_config, set_seed  # noqa: E402

META_COLUMNS = ("filename", "fold", "category")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ESC-50 windows for yamDetector (stage 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument(
        "--esc50-root",
        default=None,
        help="folder containing audio/ and meta/esc50.csv (overrides config paths)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only use the first N rows of esc50.csv (smoke test)",
    )
    return parser.parse_args(argv)


def read_meta(meta_csv: Path) -> list[dict]:
    with meta_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in META_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"error: {meta_csv} is missing column(s): {', '.join(missing)}.\n"
                "Expected the official ESC-50 meta/esc50.csv."
            )
        rows = [row for row in reader if (row.get("filename") or "").strip()]
    if not rows:
        raise SystemExit(f"error: {meta_csv} contains no rows")
    return rows


def resolve_dataset(cfg: dict, esc50_root: str | None) -> tuple[Path, Path]:
    if esc50_root:
        root = Path(esc50_root).expanduser().resolve()
        audio_dir, meta_csv = root / "audio", root / "meta" / "esc50.csv"
        if not audio_dir.is_dir() or not meta_csv.is_file():
            raise SystemExit(
                f"error: --esc50-root {root} does not look like ESC-50 "
                f"(missing {audio_dir.name}/ or meta/esc50.csv)"
            )
        return audio_dir, meta_csv
    return esc50_paths(cfg)


def clear_previous_esc50_windows(processed_dir: Path, source: str) -> int:
    """Delete wavs from an earlier run so stale windows cannot survive."""
    removed = 0
    for path in processed_dir.glob(f"*/{source}_*.wav"):
        path.unlink()
        removed += 1
    return removed


def summarize(rows: list[dict], labels: list[str]) -> None:
    per_class: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per_class[row["class"]][row["split"]] += 1
    print("\nwindows per class:")
    print(f"  {'class':<18}{'train':>7}{'val':>7}{'test':>7}{'total':>8}")
    for label in labels:
        counts = per_class.get(label, Counter())
        total = sum(counts.values())
        print(
            f"  {label:<18}{counts.get('train', 0):>7}{counts.get('val', 0):>7}"
            f"{counts.get('test', 0):>7}{total:>8}"
        )
    print(f"  {'TOTAL':<18}{sum(r['split'] == 'train' for r in rows):>7}"
          f"{sum(r['split'] == 'val' for r in rows):>7}"
          f"{sum(r['split'] == 'test' for r in rows):>7}{len(rows):>8}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    set_seed(cfg["splits"]["seed"])
    sample_rate = cfg["audio"]["sample_rate"]
    window_samples = cfg["audio"]["window_samples"]
    hop_samples = cfg["audio"]["hop_samples"]
    peak_target = cfg["audio"]["peak_target"]
    max_gain_db = cfg["audio"]["max_gain_db"]
    silence_rms = cfg["audio"]["silence_rms"]
    background = cfg["classes"]["background"]
    category_map = cfg["classes"]["esc50_map"]
    fold_map = cfg["splits"]["esc50_folds"]
    source = cfg["prepare"]["esc50_source"]
    processed_dir = cfg["paths"]["processed_dir"]
    manifest_path = cfg["paths"]["manifest_csv"]

    try:
        audio_dir, meta_csv = resolve_dataset(cfg, args.esc50_root)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    all_meta_rows = read_meta(meta_csv)
    known_categories = {str(row["category"]).strip() for row in all_meta_rows}
    for mapped_category in category_map:
        if mapped_category not in known_categories:
            print(
                f"warning: classes.esc50_map maps '{mapped_category}' but no ESC-50 file has "
                "that category - check the spelling",
                file=sys.stderr,
            )
    meta_rows = all_meta_rows[: args.limit] if args.limit else all_meta_rows

    own_rows = [row for row in manifest.read_manifest(manifest_path) if row["source"] != source]
    removed = clear_previous_esc50_windows(processed_dir, source)
    if removed:
        print(f"removed {removed} window(s) from a previous run")

    print(f"ESC-50: {audio_dir.parent}")
    print(
        f"converting {len(meta_rows)} file(s) to {sample_rate} Hz mono, "
        f"{window_samples}-sample windows ({hop_samples}-sample hop)"
    )
    if args.limit:
        print(f"note: --limit {args.limit} processes only the first rows (splits may be incomplete)")

    rows: list[dict] = []
    skipped_silent = 0
    skipped_silence_background = 0
    failed: list[tuple[Path, str]] = []

    for index, meta in enumerate(meta_rows, start=1):
        filename = str(meta["filename"]).strip()
        category = str(meta["category"]).strip()
        label = category_map.get(category, background)
        try:
            fold = int(str(meta["fold"]).strip())
        except ValueError:
            failed.append((Path(filename), f"invalid fold {meta['fold']!r}"))
            continue
        try:
            split = manifest.split_for_esc50_fold(fold, fold_map)
        except manifest.ManifestError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        clip_path = audio_dir / filename
        try:
            mono, _ = audio.load_clip(clip_path, sample_rate, channel_mode="mix")
        except audio.AudioError as exc:
            failed.append((clip_path, str(exc)))
            continue

        file_id = str(meta.get("src_file") or Path(filename).stem).strip()
        take = str(meta.get("take") or "").strip()
        if take:
            file_id = f"{file_id}-{take}"
        file_id = file_id.replace(" ", "_")

        windows = audio.slice_windows(mono, window_samples, hop_samples)
        for window_index, window in enumerate(windows):
            level = audio.rms(window)
            if level <= 0.0:
                skipped_silent += 1
                continue
            if label != background and level < silence_rms:
                skipped_silence_background += 1
                continue
            normalized = audio.peak_normalize(window, peak_target, max_gain_db)
            out_path = processed_dir / label / f"{source}_{file_id}_w{window_index:03d}.wav"
            try:
                audio.write_wav(out_path, normalized, sample_rate)
            except audio.AudioError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            rows.append(
                {
                    "path": manifest.to_manifest_path(out_path, PROJECT_ROOT),
                    "class": label,
                    "source": source,
                    "source_file_id": file_id,
                    "split": split,
                }
            )

        if index % 250 == 0 or index == len(meta_rows):
            print(f"  {index}/{len(meta_rows)} files, {len(rows)} window(s) so far")

    if failed:
        print(f"\nwarning: {len(failed)} file(s) could not be converted:")
        for path, reason in failed[:5]:
            print(f"  {path.name}: {reason}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    if not rows:
        print("error: no windows were produced - check the ESC-50 path and config", file=sys.stderr)
        return 2

    target_labels = sorted({row["class"] for row in rows if row["class"] != background})
    kept_rows, stats = manifest.subsample_background(
        rows,
        ratio=cfg["prepare"]["background_ratio"],
        seed=cfg["splits"]["seed"],
        background_label=background,
        target_labels=target_labels,
    )
    if stats["budget"] < stats["background_available"]:
        dropped_paths = {row["path"] for row in rows} - {row["path"] for row in kept_rows}
        for rel_path in dropped_paths:
            (PROJECT_ROOT / rel_path).unlink(missing_ok=True)
        print(
            f"\nbackground: kept {stats['background_kept']} of "
            f"{stats['background_available']} windows "
            f"({cfg['prepare']['background_ratio']:g}x average target class = {stats['budget']})"
        )

    all_rows = own_rows + kept_rows
    written = manifest.write_manifest(manifest_path, all_rows)
    summarize(kept_rows, sorted({row["class"] for row in kept_rows}))

    print(f"\nskipped {skipped_silent} silent background window(s) [digital silence]")
    print(f"skipped {skipped_silence_background} near-silent target window(s) (rms < {silence_rms})")
    print(f"manifest: {manifest_path} ({written} row(s), {len(own_rows)} kept from own recordings)")
    print("next: python scripts/convert_recordings.py   (optional) then python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
