"""A tiny synthetic dataset in the Montgomery folder layout, used only for smoke tests.

The images are cartoons (two dark ellipses as lungs; bright blobs for "TB"). Results on them mean
nothing; they only show that every stage of the pipeline runs end to end.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_synthetic_montgomery(root: Path, n_normal: int = 15, n_tb: int = 15, size: int = 320, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    for sub in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask", "ClinicalReadings"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    idx = 1
    for label, n in ((0, n_normal), (1, n_tb)):
        for _ in range(n):
            name = f"MCUCXR_{idx:04d}_{label}"
            idx += 1
            yy, xx = np.mgrid[:size, :size]
            img = np.full((size, size), 0.75) + rng.normal(0, 0.03, (size, size))
            masks = []
            for cx in (0.32, 0.68):
                cxp, cyp = cx * size + rng.normal(0, 4), 0.5 * size + rng.normal(0, 4)
                ax, ay = 0.13 * size * rng.uniform(0.9, 1.1), 0.30 * size * rng.uniform(0.9, 1.1)
                m = ((xx - cxp) / ax) ** 2 + ((yy - cyp) / ay) ** 2 <= 1
                img[m] -= 0.45
                masks.append(m)
            if label == 1:
                for _ in range(rng.integers(2, 5)):
                    side = masks[rng.integers(0, 2)]
                    ys, xs = np.where(side)
                    j = rng.integers(0, len(ys))
                    blob = (xx - xs[j]) ** 2 + (yy - ys[j]) ** 2 <= (size * rng.uniform(0.03, 0.06)) ** 2
                    img[blob] += 0.25
            img = np.clip(img, 0, 1)
            cv2.imwrite(str(root / "CXR_png" / f"{name}.png"), (img * 4095).astype(np.uint16))
            cv2.imwrite(str(root / "ManualMask/leftMask" / f"{name}.png"), masks[0].astype(np.uint8) * 255)
            cv2.imwrite(str(root / "ManualMask/rightMask" / f"{name}.png"), masks[1].astype(np.uint8) * 255)
            sex = "M" if rng.random() < 0.5 else "F"
            age = int(rng.integers(15, 80))
            finding = "normal" if label == 0 else rng.choice(["right pleural effusion", "inactive scars RUL", "cavitary infiltrate LUL"])
            (root / "ClinicalReadings" / f"{name}.txt").write_text(
                f"Patient's Sex: {sex}\nPatient's Age: {age:03d}Y\n{finding}\n", encoding="utf-8")
    return root
