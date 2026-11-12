"""Segmentation metrics: DSC, IoU, pixel sensitivity/precision, HD95 and lung area [paper, Eqs. 3-4]."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def hd95(pred: np.ndarray, ref: np.ndarray) -> float:
    """95th percentile of the pooled symmetric boundary distances, in pixels."""
    pred, ref = pred.astype(bool), ref.astype(bool)
    if not pred.any() and not ref.any():
        return 0.0
    if not pred.any() or not ref.any():
        return float(np.hypot(*pred.shape))  # worst case: image diagonal
    bp, br = _boundary(pred), _boundary(ref)
    d_to_r = ndi.distance_transform_edt(~br)
    d_to_p = ndi.distance_transform_edt(~bp)
    dists = np.concatenate([d_to_r[bp], d_to_p[br]])
    return float(np.percentile(dists, 95))


def seg_scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred, ref = pred.astype(bool), ref.astype(bool)
    inter = np.logical_and(pred, ref).sum()
    p, g = pred.sum(), ref.sum()
    union = np.logical_or(pred, ref).sum()
    return {
        "dsc": 2 * inter / (p + g) if (p + g) else 1.0,
        "iou": inter / union if union else 1.0,
        "sensitivity": inter / g if g else 1.0,
        "precision": inter / p if p else 1.0,
        "hd95": hd95(pred, ref),
        "area_pred_pct": 100.0 * p / pred.size,
        "area_ref_pct": 100.0 * g / ref.size,
    }


def dice_score(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = pred.astype(bool), ref.astype(bool)
    s = pred.sum() + ref.sum()
    return float(2 * np.logical_and(pred, ref).sum() / s) if s else 1.0
