"""On-the-fly augmentation (numpy/OpenCV, framework-independent).

Horizontal flips are deliberately NOT used [paper]: they would put the heart on the wrong side.
"""
from __future__ import annotations

import cv2
import numpy as np


def random_affine_matrix(size: int, rng: np.random.Generator, rotation_deg: float,
                         scale: float, shift: float) -> np.ndarray:
    angle = rng.uniform(-rotation_deg, rotation_deg)
    s = rng.uniform(1.0 - scale, 1.0 + scale)            # isotropic scaling
    tx, ty = rng.uniform(-shift, shift, size=2) * size   # translation as fraction of size
    c = (size - 1) / 2.0
    M = cv2.getRotationMatrix2D((c, c), angle, s)
    M[:, 2] += (tx, ty)
    return M


def apply_affine(img: np.ndarray, M: np.ndarray, is_mask: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    out = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def jitter_intensity(img: np.ndarray, rng: np.random.Generator, brightness: float,
                     contrast: float) -> np.ndarray:
    """img in [0, 1]. Contrast about the image mean, then additive brightness."""
    c = rng.uniform(1.0 - contrast, 1.0 + contrast)
    b = rng.uniform(-brightness, brightness)
    m = img.mean()
    return np.clip((img - m) * c + m + b, 0.0, 1.0)


def augment_pair(img: np.ndarray, mask: np.ndarray | None, rng: np.random.Generator, *,
                 rotation_deg, scale, shift, brightness, contrast):
    """Same geometric transform for image and mask; intensity jitter on the image only."""
    M = random_affine_matrix(img.shape[0], rng, rotation_deg, scale, shift)
    img_t = jitter_intensity(apply_affine(img, M), rng, brightness, contrast)
    mask_t = None if mask is None else (apply_affine(mask.astype(np.float32), M, is_mask=True) > 0.5)
    return img_t.astype(np.float32), mask_t
