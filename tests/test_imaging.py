import numpy as np
import pytest

from tbstudy.baseline import otsu_lung_mask
from tbstudy.postprocess import clean_mask, dilate, keep_largest_components, lung_roi, whole_image
from tbstudy.seg_metrics import hd95, seg_scores
from tbstudy.transforms import augment_pair


def square(size=64, y0=10, x0=10, s=20):
    m = np.zeros((size, size), bool)
    m[y0:y0 + s, x0:x0 + s] = True
    return m


def test_perfect_segmentation():
    m = square()
    r = seg_scores(m, m)
    assert r["dsc"] == 1 and r["iou"] == 1 and r["hd95"] == 0


def test_shifted_square_metrics():
    a, b = square(), square(x0=12)
    r = seg_scores(a, b)
    assert r["dsc"] == pytest.approx(2 * 18 * 20 / 800)
    assert hd95(a, b) == pytest.approx(2.0)


def test_empty_masks():
    z = np.zeros((32, 32), bool)
    assert hd95(z, z) == 0
    assert hd95(z, square(32, 5, 5, 5)) > 40


def test_keep_two_largest_and_fill():
    m = square(64, 5, 5, 20) | square(64, 5, 35, 20) | square(64, 50, 50, 3)
    m[10, 10] = False  # a hole
    out = clean_mask(m.astype(float), 0.5, 2, fill_holes=True)
    assert not out[51, 51] and out[10, 10]
    assert keep_largest_components(m, 2).sum() == m.sum() - 9


def test_dilate_grows_mask():
    m = square()
    assert dilate(m, 6).sum() > m.sum() and dilate(m, 0).sum() == m.sum()


def test_lung_roi_crops_and_masks():
    img = np.random.default_rng(0).random((256, 256)).astype(np.float32)
    m = np.zeros((256, 256), bool); m[60:200, 40:120] = True; m[60:200, 140:220] = True
    roi = lung_roi(img, m, 224)
    assert roi.shape == (224, 224)
    assert roi[:, 100:120].max() < 1e-6          # the masked gap between lungs is black
    assert whole_image(img).shape == (224, 224)
    assert lung_roi(img, np.zeros_like(m)).shape == (224, 224)  # empty mask falls back


def test_augment_keeps_geometry_consistent():
    rng = np.random.default_rng(0)
    img = np.zeros((128, 128), np.float32); m = square(128, 40, 40, 40)
    img[m] = 1.0
    out, mo = augment_pair(img, m, rng, rotation_deg=10, scale=0.1, shift=0.05, brightness=0.0, contrast=0.0)
    assert mo.dtype == bool and 0.7 < mo.sum() / m.sum() < 1.3
    assert out[mo].mean() > 0.9   # mask still sits on the bright square


def test_otsu_baseline_finds_two_dark_regions():
    yy, xx = np.mgrid[:256, :256]
    img = np.full((256, 256), 20, np.uint8)
    body = ((xx - 128) / 110) ** 2 + ((yy - 128) / 120) ** 2 <= 1
    img[body] = 180
    for cx in (85, 171):
        img[((xx - cx) / 30) ** 2 + ((yy - 128) / 70) ** 2 <= 1] = 60
    m = otsu_lung_mask(img)
    from scipy import ndimage as ndi
    assert ndi.label(m)[1] == 2 and m[128, 85] and m[128, 171] and not m[128, 128]
