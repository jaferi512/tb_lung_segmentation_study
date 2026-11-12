"""Stage 1: fold-specific U-Net lung segmentation and the Otsu baseline.

For each fold k, a U-Net is trained on the other four folds only. It is then applied to all 138 images:
  * its masks on fold k's (held-out) images give the out-of-fold segmentation metrics;
  * its masks on *all* images feed the lung-ROI classifier of fold k, so neither the segmentation model
    nor the classifier ever sees the classifier's test images [paper, 'Evaluation protocol'].
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .baseline import otsu_lung_mask
from .config import Config
from .models import UNet, bce_dice_loss, count_parameters
from .postprocess import clean_mask, dilate
from .seg_metrics import dice_score
from .transforms import augment_pair
from .utils import Timer, get_device, log, save_json, seed_everything


def to_seg_input(images_u8: np.ndarray, size: int) -> np.ndarray:
    """uint8 256x256 -> float [0,1] at `size` with area averaging [paper]."""
    return np.stack([cv2.resize(im.astype(np.float32) / 255.0, (size, size), interpolation=cv2.INTER_AREA)
                     for im in images_u8])


def to_seg_target(masks: np.ndarray, size: int) -> np.ndarray:
    return np.stack([cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.5
                     for m in masks])


def standardise(x: np.ndarray) -> np.ndarray:
    """[assumption] per-image z-score for the U-Net input."""
    return (x - x.mean()) / (x.std() + 1e-6)


class SegDataset(Dataset):
    def __init__(self, images, masks, cfg, augment: bool, seed: int):
        self.images, self.masks, self.cfg, self.augment = images, masks, cfg, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img, msk = self.images[i], self.masks[i]
        if self.augment:
            c = self.cfg
            img, msk = augment_pair(img, msk, self.rng, rotation_deg=c.rotation_deg, scale=c.scale,
                                    shift=c.shift, brightness=c.brightness, contrast=c.contrast)
        x = torch.from_numpy(standardise(img)[None].astype(np.float32))
        y = torch.from_numpy(msk[None].astype(np.float32))
        return x, y


@torch.no_grad()
def predict_probs(model, images_small: np.ndarray, out_size: int, device, batch_size: int = 16) -> np.ndarray:
    """Probability maps at `out_size` (bilinear upsampling of the 128x128 output) [paper]."""
    model.eval()
    out = []
    for s in range(0, len(images_small), batch_size):
        x = torch.from_numpy(np.stack([standardise(im) for im in images_small[s:s + batch_size]])[:, None])
        logits = model(x.float().to(device))
        prob = torch.sigmoid(F.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False))
        out.append(prob[:, 0].cpu().numpy())
    return np.concatenate(out)


def train_fold(cfg: Config, k: int, images_u8, masks, folds, device):
    sc = cfg.seg
    seed_everything(cfg.seed + k)
    x_small = to_seg_input(images_u8, sc.input_size)
    y_small = to_seg_target(masks, sc.input_size)
    tr, te = np.where(folds != k)[0], np.where(folds == k)[0]

    g = torch.Generator().manual_seed(cfg.seed + k)
    loader = DataLoader(SegDataset(x_small[tr], y_small[tr], sc, True, cfg.seed + k),
                        batch_size=sc.batch_size, shuffle=True, generator=g, num_workers=cfg.num_workers)
    model = UNet(base=sc.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=sc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sc.epochs)

    curve = {"epoch": [], "train_loss": [], "heldout_dsc": []}
    for epoch in range(1, sc.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = bce_dice_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        # Monitoring only (not used for model selection; the epoch count is fixed in advance) [paper, Fig. 7a]
        probs = predict_probs(model, x_small[te], sc.eval_size, device)
        dsc = np.mean([dice_score(clean_mask(p, sc.threshold, sc.keep_components), masks[i])
                       for p, i in zip(probs, te)])
        curve["epoch"].append(epoch)
        curve["train_loss"].append(float(np.mean(losses)))
        curve["heldout_dsc"].append(float(dsc))
        log(f"U-Net fold {k + 1} epoch {epoch:2d}/{sc.epochs}  loss {np.mean(losses):.4f}  held-out DSC {dsc:.4f}")

    probs_all = predict_probs(model, x_small, sc.eval_size, device)
    return model, probs_all, curve


def run(cfg: Config, images_u8, masks, folds):
    device = get_device(cfg.device)
    cfg.seg_dir.mkdir(parents=True, exist_ok=True)
    n = len(images_u8)
    oof_masks = np.zeros_like(masks, dtype=bool)
    timings = []
    log(f"U-Net trainable parameters: {count_parameters(UNet(base=cfg.seg.base_channels)):,} (paper: 1,942,289)")
    for k in range(cfg.n_folds):
        with Timer() as t:
            model, probs, curve = train_fold(cfg, k, images_u8, masks, folds, device)
        timings.append(t.seconds)
        seg_masks = np.stack([clean_mask(p, cfg.seg.threshold, cfg.seg.keep_components, fill_holes=True) for p in probs])
        # Lung-ROI masks for the classifier: threshold, two largest components, dilate 6 px [paper]
        roi_masks = np.stack([dilate(clean_mask(p, cfg.seg.threshold, cfg.seg.keep_components, fill_holes=False),
                                     cfg.cls.roi_dilation_px) for p in probs])
        te = folds == k
        oof_masks[te] = seg_masks[te]
        torch.save(model.state_dict(), cfg.seg_dir / f"unet_fold{k + 1}.pt")
        np.savez_compressed(cfg.seg_dir / f"roi_masks_fold{k + 1}.npz", roi_masks=np.packbits(roi_masks, axis=-1))
        save_json(curve, cfg.seg_dir / f"curve_fold{k + 1}.json")
        log(f"U-Net fold {k + 1} done in {t.seconds / 60:.1f} min")

    baseline = np.stack([otsu_lung_mask(im) for im in images_u8])
    np.savez_compressed(cfg.seg_dir / "oof_masks.npz", unet=np.packbits(oof_masks, axis=-1),
                        otsu=np.packbits(baseline, axis=-1), shape=np.array(masks.shape))
    save_json({"minutes_per_fold": [s / 60 for s in timings], "n_images": n}, cfg.seg_dir / "timing.json")

