"""Unit tests for manifest IO and the split/subsampling logic."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yamdetector import manifest  # noqa: E402

FOLD_MAP = {"train": [1, 2, 3], "val": [4], "test": [5]}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_row(path: str, label: str, split: str, file_id: str = "f") -> dict:
    return {
        "path": path,
        "class": label,
        "source": "esc50",
        "source_file_id": file_id,
        "split": split,
    }


class ManifestIOTests(unittest.TestCase):
    def test_roundtrip(self):
        rows = [
            make_row("data/processed/siren/a.wav", "siren", "train", "a"),
            make_row("data/processed/background/b.wav", "background", "test", "b"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            written = manifest.write_manifest(path, rows)
            self.assertEqual(written, 2)
            loaded = manifest.read_manifest(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(set(loaded[0]), set(manifest.MANIFEST_COLUMNS))
            self.assertEqual([row["split"] for row in loaded], ["test", "train"])  # sorted

    def test_missing_manifest_returns_empty(self):
        self.assertEqual(manifest.read_manifest("does/not/exist.csv"), [])

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text("path,class,split\nx.wav,siren,train\n", encoding="utf-8")
            with self.assertRaises(manifest.ManifestError):
                manifest.read_manifest(path)

    def test_invalid_split_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text(
                "path,class,source,source_file_id,split\nx.wav,siren,esc50,a,holdout\n",
                encoding="utf-8",
            )
            with self.assertRaises(manifest.ManifestError):
                manifest.read_manifest(path)

    def test_manifest_paths_are_relative_posix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "data" / "processed" / "siren" / "a.wav"
            self.assertEqual(
                manifest.to_manifest_path(nested, root), "data/processed/siren/a.wav"
            )


class Esc50FoldTests(unittest.TestCase):
    def test_folds_map_to_splits(self):
        self.assertEqual(manifest.split_for_esc50_fold(1, FOLD_MAP), "train")
        self.assertEqual(manifest.split_for_esc50_fold(3, FOLD_MAP), "train")
        self.assertEqual(manifest.split_for_esc50_fold(4, FOLD_MAP), "val")
        self.assertEqual(manifest.split_for_esc50_fold(5, FOLD_MAP), "test")

    def test_unmapped_fold_raises(self):
        with self.assertRaises(manifest.ManifestError):
            manifest.split_for_esc50_fold(9, FOLD_MAP)


class AssignSplitsTests(unittest.TestCase):
    def test_every_file_gets_exactly_one_split(self):
        files = [f"rec{i}" for i in range(20)]
        assignment = manifest.assign_splits_by_file(files, RATIOS, seed=42)
        self.assertEqual(set(assignment), set(files))
        counts = Counter(assignment.values())
        self.assertEqual(counts["train"], 14)
        self.assertEqual(counts["val"], 3)
        self.assertEqual(counts["test"], 3)

    def test_deterministic_for_same_seed(self):
        files = [f"rec{i}" for i in range(30)]
        first = manifest.assign_splits_by_file(files, RATIOS, seed=7)
        second = manifest.assign_splits_by_file(files, RATIOS, seed=7)
        self.assertEqual(first, second)

    def test_different_seed_changes_assignment(self):
        files = [f"rec{i}" for i in range(30)]
        first = manifest.assign_splits_by_file(files, RATIOS, seed=1)
        second = manifest.assign_splits_by_file(files, RATIOS, seed=2)
        self.assertNotEqual(first, second)

    def test_all_windows_of_a_file_share_a_split(self):
        # the function is per file id, so asserting the id map is enough
        files = ["takeA", "takeA", "takeB", "takeC", "takeD"]
        assignment = manifest.assign_splits_by_file(files, RATIOS, seed=3)
        self.assertEqual(len(assignment), 4)
        self.assertIn(assignment["takeA"], manifest.SPLIT_NAMES)

    def test_tiny_class_keeps_everything_in_train(self):
        assignment = manifest.assign_splits_by_file(["only_one"], RATIOS, seed=0)
        self.assertEqual(assignment, {"only_one": "train"})

    def test_three_files_get_one_per_split(self):
        assignment = manifest.assign_splits_by_file(["a", "b", "c"], RATIOS, seed=0)
        self.assertEqual(sorted(assignment.values()), ["test", "train", "val"])

    def test_empty_input(self):
        self.assertEqual(manifest.assign_splits_by_file([], RATIOS, seed=0), {})


class BackgroundSubsampleTests(unittest.TestCase):
    def build_rows(self, target_per_class: int, background_per_split: dict[str, int]):
        rows = []
        for label in ("siren", "knock"):
            for split in manifest.SPLIT_NAMES:
                for index in range(target_per_class):
                    rows.append(make_row(f"data/processed/{label}/{split}{index}.wav", label, split, f"{label}{split}{index}"))
        for split, count in background_per_split.items():
            for index in range(count):
                rows.append(
                    make_row(f"data/processed/background/{split}{index}.wav", "background", split, f"bg{split}{index}")
                )
        return rows

    def test_subsamples_to_ratio_of_average_target_class(self):
        rows = self.build_rows(target_per_class=10, background_per_split={"train": 500, "val": 100, "test": 100})
        kept, stats = manifest.subsample_background(rows, 4.0, 42, "background", ["siren", "knock"])
        # two target classes with 30 windows each -> average 30 -> budget 120
        self.assertEqual(stats["budget"], 120)
        self.assertEqual(stats["background_kept"], 120)
        background_kept = [row for row in kept if row["class"] == "background"]
        self.assertEqual(len(background_kept), 120)
        self.assertEqual(len({row["path"] for row in background_kept}), 120)
        # target classes untouched
        self.assertEqual(len([row for row in kept if row["class"] != "background"]), 60)

    def test_keeps_split_proportions(self):
        rows = self.build_rows(target_per_class=10, background_per_split={"train": 500, "val": 100, "test": 100})
        kept, _ = manifest.subsample_background(rows, 4.0, 42, "background", ["siren", "knock"])
        splits = Counter(row["split"] for row in kept if row["class"] == "background")
        self.assertGreater(splits["train"], splits["val"])
        self.assertGreater(splits["val"], 0)
        self.assertGreater(splits["test"], 0)

    def test_deterministic(self):
        rows = self.build_rows(target_per_class=5, background_per_split={"train": 300, "val": 60, "test": 60})
        first, _ = manifest.subsample_background(rows, 4.0, 7, "background", ["siren", "knock"])
        second, _ = manifest.subsample_background(rows, 4.0, 7, "background", ["siren", "knock"])
        self.assertEqual([row["path"] for row in first], [row["path"] for row in second])

    def test_no_subsample_when_background_is_small(self):
        rows = self.build_rows(target_per_class=10, background_per_split={"train": 20, "val": 5, "test": 5})
        kept, stats = manifest.subsample_background(rows, 4.0, 42, "background", ["siren", "knock"])
        self.assertEqual(stats["background_kept"], 30)
        self.assertEqual(len(kept), len(rows))

    def test_without_targets_nothing_is_dropped(self):
        rows = [make_row("data/processed/background/a.wav", "background", "train", "a")]
        kept, _ = manifest.subsample_background(rows, 4.0, 42, "background", [])
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
