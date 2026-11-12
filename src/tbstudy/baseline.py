"""Training-free lung segmentation baseline: CLAHE + two-stage Otsu thresholding [paper, Methods]."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.segmentation import clear_border

from .postprocess import keep_largest_components


def otsu_lung_mask(image_u8: np.ndarray, border: int = 10, open_radius: int = 3) -> np.ndarray:
    """image_u8: 8-bit greyscale radiograph (256 x 256). Returns a boolean lung mask."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # [assumption] CLAHE parameters
    eq = clahe.apply(image_u8)

    # 1) global Otsu: body (bright) vs background (dark)
    body = eq > threshold_otsu(eq)
    body = ndi.binary_fill_holes(keep_largest_components(body, 1))

    # 2) Otsu inside the body: dark, air-filled regions
    vals = eq[body]
    if vals.size < 10:
        return np.zeros_like(body)
    air = (eq <= threshold_otsu(vals)) & body

    # 3) exclude a border band, open, drop edge-touching components, keep two largest, fill holes
    air[:border, :] = air[-border:, :] = False
    air[:, :border] = air[:, -border:] = False
    air = ndi.binary_opening(air, structure=disk(open_radius))
    air = clear_border(air)
    air = keep_largest_components(air, 2)
    return ndi.binary_fill_holes(air).astype(bool)
