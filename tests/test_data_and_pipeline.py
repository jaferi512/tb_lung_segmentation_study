import numpy as np
import pytest

from tbstudy.config import smoke_config
from tbstudy.data import label_from_name, load_cache, make_folds, parse_reading, preprocess, save_cache
from tbstudy.synthetic import make_synthetic_montgomery


def test_parse_reading():
    sex, age, finding = parse_reading("Patient's Sex: F\nPatient's Age: 031Y\ncavitary nodular infiltrate in RUL; active TB\n")
    assert sex == "F" and age == 31 and "cavitary" in finding
    assert parse_reading("Patient's Sex: M\nPatient's Age: 018M\nnormal")[1] == pytest.approx(1.5)


def test_label_from_name():
    assert label_from_name("MCUCXR_0001_0.png") == 0 and label_from_name("MCUCXR_0104_1.png") == 1


def test_folds_match_paper_table9():
    """80 normal + 58 TB, seed 42 -> 28/28/28/27/27 images with 12/12/12/11/11 TB (paper Table 9)."""
    labels = np.r_[np.zeros(80), np.ones(58)].astype(int)
    folds = make_folds(labels, 5, 42)
    sizes = [int((folds == k).sum()) for k in range(5)]
    tb = [int(labels[folds == k].sum()) for k in range(5)]
    assert sorted(sizes, reverse=True) == [28, 28, 28, 27, 27]
    assert sorted(tb, reverse=True) == [12, 12, 12, 11, 11]
    assert (folds >= 0).all()


def test_preprocess_and_cache_roundtrip(tmp_path):
    root = make_synthetic_montgomery(tmp_path / "MontgomerySet", n_normal=4, n_tb=4, size=200)
    images, masks, records = preprocess(root, 256)
    assert images.shape == (8, 256, 256) and images.dtype == np.uint8 and images.max() > 200
    assert masks.dtype == bool and 0.1 < masks.mean() < 0.5
    assert [r.label for r in records] == [0] * 4 + [1] * 4
    save_cache(tmp_path / "c.npz", tmp_path / "m.csv", images, masks, records)
    i2, m2, r2 = load_cache(tmp_path / "c.npz", tmp_path / "m.csv")
    assert (i2 == images).all() and (m2 == masks).all() and r2[0].sex == records[0].sex


def test_analysis_stage_end_to_end(tmp_path):
    """Run prepare + analyze on synthetic data with simulated network outputs (no PyTorch needed)."""
    from tbstudy import analyze
    from tbstudy.__main__ import cmd_prepare
    from tbstudy.baseline import otsu_lung_mask
    from tbstudy.postprocess import dilate
    from tbstudy.utils import save_json

    data = make_synthetic_montgomery(tmp_path / "data", n_normal=20, n_tb=15, size=220)
    cfg = smoke_config(data, tmp_path / "out")
    cmd_prepare(cfg)
    images, masks, records = load_cache(cfg.cache_file, cfg.meta_file)
    labels = np.array([r.label for r in records])
    folds = make_folds(labels, cfg.n_folds, cfg.seed)
    rng = np.random.default_rng(0)

    # simulated U-Net output: reference mask slightly eroded/dilated
    from scipy import ndimage as ndi
    unet = np.stack([ndi.binary_erosion(m, iterations=int(rng.integers(1, 3))) if rng.random() < .5 else m for m in masks])
    otsu = np.stack([otsu_lung_mask(im) for im in images])
    cfg.seg_dir.mkdir(parents=True)
    np.savez_compressed(cfg.seg_dir / "oof_masks.npz", unet=np.packbits(unet, axis=-1),
                        otsu=np.packbits(otsu, axis=-1), shape=np.array(masks.shape))
    for k in range(5):
        np.savez_compressed(cfg.seg_dir / f"roi_masks_fold{k + 1}.npz", roi_masks=np.packbits(np.stack([dilate(m, 6) for m in unet]), axis=-1))
        save_json({"epoch": [1, 2], "train_loss": [0.9, 0.5], "heldout_dsc": [0.8, 0.95]}, cfg.seg_dir / f"curve_fold{k + 1}.json")
    cfg.cls_dir.mkdir(parents=True)
    pw = np.clip(labels * 0.6 + rng.normal(0.2, 0.2, len(labels)), 0, 1)
    pr = np.clip(labels * 0.4 + rng.normal(0.25, 0.25, len(labels)), 0, 1)
    with open(cfg.cls_dir / "oof_predictions.csv", "w") as f:
        f.write("image_id,fold,label,p_whole,p_roi\n")
        for r, k, a, b in zip(records, folds, pw, pr):
            f.write(f"{r.image_id},{k + 1},{r.label},{a:.6f},{b:.6f}\n")
    save_json({"train_loss": {"whole": [[0.7, 0.3]] * 5, "roi": [[0.7, 0.4]] * 5}}, cfg.cls_dir / "training_curves.json")

    report = analyze.run(cfg, images, masks, records, folds, with_gradcam=False)
    assert "Headline comparison" in report and "DeLong" in report
    figs = sorted(p.name for p in (cfg.results_dir / "figures").glob("*.png"))
    assert len(figs) == 7, figs
    assert (cfg.results_dir / "results.json").exists()


def test_unet_parameter_count_matches_paper():
    torch = pytest.importorskip("torch")
    from tbstudy.models import UNet, count_parameters, resnet18_binary
    m = UNet()
    assert count_parameters(m) == 1_942_289
    assert m(torch.zeros(2, 1, 128, 128)).shape == (2, 1, 128, 128)
    assert abs(count_parameters(resnet18_binary(False)) - 11.18e6) < 0.01e6
