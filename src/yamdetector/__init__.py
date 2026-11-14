"""Shared helpers for the yamDetector pipeline.

Stage 1 (scripts/prepare_esc50.py) and stage 2 (scripts/convert_recordings.py)
both write the standardized dataset folder (data/processed + manifest.csv);
stage 3 (train.py) and stage 4 (export.py) only ever read that folder plus
config.yaml.
"""

__all__ = ["audio", "config", "features", "manifest", "model"]
