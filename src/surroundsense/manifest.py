"""manifest.csv handling: the single contract between data prep and training.

Columns: path, class, source, source_file_id, split.

* ``path``           - processed wav, relative to the project root
* ``class``          - label the model learns
* ``source``         - "esc50" or "own" (anything the preprocessing stages write)
* ``source_file_id`` - the recording the window came from; splits never split it
* ``split``          - train / val / test
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MANIFEST_COLUMNS = ("path", "class", "source", "source_file_id", "split")
SPLIT_NAMES = ("train", "val", "test")


class ManifestError(RuntimeError):
    """Raised for malformed manifests or impossible split requests."""


def read_manifest(path: str | Path) -> list[dict]:
    """Read manifest.csv into a list of row dicts (empty list if it does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in MANIFEST_COLUMNS if column not in fieldnames]
        if missing:
            raise ManifestError(
                f"{path} is missing column(s) {', '.join(missing)}; "
                f"expected {', '.join(MANIFEST_COLUMNS)}"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            cleaned = {column: (row.get(column) or "").strip() for column in MANIFEST_COLUMNS}
            if not cleaned["path"]:
                continue
            if cleaned["split"] not in SPLIT_NAMES:
                raise ManifestError(
                    f"{path}:{line_number} has split {cleaned['split']!r}; "
                    f"expected one of {SPLIT_NAMES}"
                )
            if not cleaned["class"]:
                raise ManifestError(f"{path}:{line_number} has an empty class column")
            rows.append(cleaned)
    return rows


def to_manifest_path(path: str | Path, project_root: str | Path) -> str:
    """Path as stored in manifest.csv: relative to the project root, posix slashes."""
    path = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(os.path.relpath(path, root))
    return relative.as_posix()


def write_manifest(path: str | Path, rows: Iterable[dict]) -> int:
    """Write rows sorted deterministically; returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("source", ""),
            row.get("class", ""),
            row.get("source_file_id", ""),
            row.get("path", ""),
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in ordered:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})
    return len(ordered)


def split_for_esc50_fold(fold: int, fold_map: dict[str, Sequence[int]]) -> str:
    """Map an ESC-50 fold (1-5) to its split using ``splits.esc50_folds``."""
    for split in SPLIT_NAMES:
        if int(fold) in {int(item) for item in fold_map.get(split, ())}:
            return split
    raise ManifestError(
        f"ESC-50 fold {fold} is not assigned to any split in 'splits.esc50_folds'"
    )


def _split_counts(count: int, ratios: dict[str, float]) -> dict[str, int]:
    """Turn ratios into integer counts, giving val/test at least one file when possible."""
    if count <= 0:
        return {split: 0 for split in SPLIT_NAMES}
    if count < 3:
        # Not enough recordings to hold out; all of them must stay in train.
        return {"train": count, "val": 0, "test": 0}

    raw = {split: count * ratios[split] for split in SPLIT_NAMES}
    counts = {split: int(np.floor(raw[split])) for split in SPLIT_NAMES}
    remainder = count - sum(counts.values())
    for split in sorted(SPLIT_NAMES, key=lambda name: raw[name] - counts[name], reverse=True):
        if remainder <= 0:
            break
        counts[split] += 1
        remainder -= 1
    for split in ("val", "test"):
        if counts[split] == 0 and ratios[split] > 0:
            donor = max(("train", "val", "test"), key=lambda name: counts[name])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[split] += 1
    return counts


def assign_splits_by_file(
    file_ids: Sequence[str], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign whole source recordings to train/val/test (~70/15/15 by default).

    Deterministic for a given ``(file_ids, ratios, seed)`` set: the ids are
    sorted, shuffled with the seed, then cut into the three splits. Every window
    of a recording therefore lands in the same split, so no recording leaks
    across splits through different windows.
    """
    unique_ids = sorted({str(file_id) for file_id in file_ids})
    if not unique_ids:
        return {}
    counts = _split_counts(len(unique_ids), ratios)
    rng = np.random.default_rng(seed)
    shuffled = [unique_ids[index] for index in rng.permutation(len(unique_ids))]

    assignment: dict[str, str] = {}
    cursor = 0
    for split in SPLIT_NAMES:
        for file_id in shuffled[cursor : cursor + counts[split]]:
            assignment[file_id] = split
        cursor += counts[split]
    return assignment


def class_for_labels(
    labels: Sequence[str],
    label_map: dict[str, str],
    order: Sequence[str],
    background_label: str,
) -> str:
    """Pick one class for a multi-label clip.

    FSD50K clips carry several labels (a scream is also ``Human_voice`` and
    often ``Speech``). ``order`` decides the winner, so the alert classes listed
    first in ``classes.order`` beat the broad ``speech``/``music`` labels.
    """
    hits = {label_map[label] for label in labels if label in label_map}
    if not hits:
        return background_label
    for label in order:
        if label in hits:
            return label
    return sorted(hits)[0]


def assert_unique_paths(rows: Sequence[dict]) -> None:
    """Every window file is one row in one split, and no recording leaks.

    A window appearing twice (possibly in two splits) silently inflates the
    results: the same audio is trained on and tested on. Splits are assigned per
    source recording, so every window of one recording must share a split too.
    """
    seen: dict[str, str] = {}
    for row in rows:
        path, split = row["path"], row["split"]
        if path in seen:
            detail = (
                f"in both '{seen[path]}' and '{split}' (train/test leakage)"
                if seen[path] != split
                else "more than once"
            )
            raise ManifestError(f"{path} appears {detail}")
        seen[path] = split

    file_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        file_splits[(row.get("source", ""), row.get("source_file_id", ""))].add(row["split"])
    leaked = sorted(key for key, splits in file_splits.items() if len(splits) > 1)
    if leaked:
        source, file_id = leaked[0]
        raise ManifestError(
            f"recording '{file_id}' (source '{source}') has windows in more than one split "
            f"({', '.join(sorted(file_splits[leaked[0]]))}); splits must follow the "
            "source recording, never the window"
        )


def class_budgets(
    rows: Sequence[dict],
    background_label: str,
    background_ratio: float,
    max_per_class: int,
    max_background: int | None = None,
) -> dict[str, int]:
    """Windows to keep per class: a flat cap, with background scaled by ratio.

    Background gets its own, larger cap (``max_background``): it is the class the
    wearable cares about most, because every background window the model calls a
    target is a false buzz. Capping it with the same flat number as the targets
    is what made the first 13-class run fire on 61% of ordinary room sound.
    """
    counts = Counter(row["class"] for row in rows)
    budgets = {label: int(max_per_class) for label in counts}
    targets = [count for label, count in counts.items() if label != background_label]
    if targets:
        average = sum(targets) / len(targets)
        ceiling = int(max_background) if max_background else int(max_per_class)
        budgets[background_label] = max(
            1, min(ceiling, int(round(background_ratio * average)))
        )
    return budgets


def cap_class_windows(
    rows: Sequence[dict], budgets: dict[str, int], seed: int
) -> tuple[list[dict], dict]:
    """Keep at most ``budgets[class]`` windows per class, spread over the splits.

    Deterministic for a given ``(rows, budgets, seed)``: the same windows come
    back on every run, so train.py and export.py see the same test set. Each
    split keeps its share, so the fold-based split ratio survives the cap.
    Returns ``(kept_rows, stats)``.
    """
    stats: dict[str, dict] = {}
    kept: list[dict] = []

    for label in sorted({row["class"] for row in rows}):
        label_rows = sorted(
            (row for row in rows if row["class"] == label), key=lambda row: row["path"]
        )
        budget = int(budgets.get(label, len(label_rows)))
        if budget >= len(label_rows) or budget <= 0:
            kept.extend(label_rows if budget > 0 else [])
            stats[label] = {
                "available": len(label_rows),
                "kept": len(label_rows) if budget > 0 else 0,
                "budget": max(budget, 0),
            }
            continue

        by_split: dict[str, list[dict]] = defaultdict(list)
        for row in label_rows:
            by_split[row["split"]].append(row)
        total_available = len(label_rows)
        # A stable per-class stream: adding a class never reshuffles another one.
        rng = np.random.default_rng([int(seed), len(label), sum(map(ord, label))])

        chosen: dict[str, list[dict]] = {}
        for split in SPLIT_NAMES:
            available = by_split.get(split, [])
            if not available:
                continue
            share = int(round(budget * len(available) / total_available))
            share = min(max(share, 1), len(available))
            indices = np.sort(rng.permutation(len(available))[:share])
            chosen[split] = [available[int(index)] for index in indices]

        kept_count = sum(len(entries) for entries in chosen.values())
        while kept_count > budget:  # rounding overshoot: trim the biggest split
            split = max(chosen, key=lambda name: len(chosen[name]))
            if len(chosen[split]) <= 1:
                break
            chosen[split].pop()
            kept_count -= 1

        for split in SPLIT_NAMES:
            kept.extend(chosen.get(split, []))
        stats[label] = {
            "available": total_available,
            "kept": sum(len(entries) for entries in chosen.values()),
            "budget": budget,
        }

    kept.sort(key=lambda row: row["path"])
    return kept, stats
