"""Mask post-processing and construction of the lung-ROI classifier input."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk


def keep_largest_components(mask: np.ndarray, n: int = 2) -> np.ndarray:
    lab, num = ndi.label(mask)
    if num <= n:
        return mask.astype(bool)
    sizes = ndi.sum(mask, lab, index=np.arange(1, num + 1))
    keep = np.argsort(sizes)[::-1][:n] + 1
    return np.isin(lab, keep)


def clean_mask(prob: np.ndarray, threshold: float = 0.5, n_components: int = 2,
               fill_holes: bool = True) -> np.ndarray:
    """[paper] threshold 0.5 -> two largest connected components -> (optionally) fill holes."""
    m = keep_largest_components(prob >= threshold, n_components)
    if fill_holes:
        m = ndi.binary_fill_holes(m)
    return m.astype(bool)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    return ndi.binary_dilation(mask, structure=disk(radius)).astype(bool)


def lung_roi(image: np.ndarray, mask: np.ndarray, out_size: int = 224) -> np.ndarray:
    """[paper] multiply image by (dilated) mask, crop to the mask bounding box, resize to out_size.

    image: float array in [0, 1] (256 x 256); mask: bool array of the same size.
    Falls back to the whole image if the mask is empty.
    """
    if not mask.any():
        return cv2.resize(image.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    masked = image.astype(np.float32) * mask
    rows, cols = np.where(mask)
    crop = masked[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def whole_image(image: np.ndarray, out_size: int = 224) -> np.ndarray:
    return cv2.resize(image.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_LINEAR)
