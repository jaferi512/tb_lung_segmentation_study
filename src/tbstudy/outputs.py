"""Readers for the saved stage outputs (no PyTorch needed)."""
from __future__ import annotations

import csv

import numpy as np

from .config import Config


def load_roi_masks(cfg: Config, k: int, shape) -> np.ndarray:
    """Dilated lung masks produced by the fold-k U-Net for all images (input to the lung-ROI classifier)."""
    packed = np.load(cfg.seg_dir / f"roi_masks_fold{k + 1}.npz")["roi_masks"]
    return np.unpackbits(packed, axis=-1)[..., : shape[-1]].astype(bool)


def load_oof_masks(cfg: Config):
    """Out-of-fold U-Net masks and the Otsu baseline masks."""
    d = np.load(cfg.seg_dir / "oof_masks.npz")
    w = int(d["shape"][-1])
    unpack = lambda a: np.unpackbits(a, axis=-1)[..., :w].astype(bool)
    return unpack(d["unet"]), unpack(d["otsu"])


def load_oof(cfg: Config) -> dict:
    with open(cfg.cls_dir / "oof_predictions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "image_id": [r["image_id"] for r in rows],
        "fold": np.array([int(r["fold"]) for r in rows]),
        "label": np.array([int(r["label"]) for r in rows]),
        "p_whole": np.array([float(r["p_whole"]) for r in rows]),
        "p_roi": np.array([float(r["p_roi"]) for r in rows]),
    }
