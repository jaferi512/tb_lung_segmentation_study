"""Stage 2: ResNet-18 TB classifier under two input conditions, with identical folds, seed and schedule.

  * whole : the 256x256 radiograph resized to 224x224              (Path A)
  * roi   : radiograph x dilated fold-specific U-Net mask, cropped  (Path B)
            to the mask bounding box and resized to 224x224

For a given fold both conditions start from the same seed, so the new output layer is initialised
identically and mini-batches are drawn in the same order: the input is the only difference [paper].
"""
from __future__ import annotations

import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .models import count_parameters, resnet18_binary
from .postprocess import lung_roi, whole_image
from .outputs import load_roi_masks
from .transforms import augment_pair
from .utils import IMAGENET_MEAN, IMAGENET_STD, Timer, get_device, log, save_json, seed_everything

CONDITIONS = ("whole", "roi")


def build_inputs(images_u8: np.ndarray, condition: str, roi_masks: np.ndarray | None, size: int) -> np.ndarray:
    imgs = images_u8.astype(np.float32) / 255.0
    if condition == "whole":
        return np.stack([whole_image(im, size) for im in imgs])
    if condition == "roi":
        return np.stack([lung_roi(im, m, size) for im, m in zip(imgs, roi_masks)])
    raise ValueError(condition)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """Grey [0,1] -> 3-channel ImageNet-normalised tensor [paper]."""
    x = np.repeat(img[None], 3, axis=0)
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(x.astype(np.float32))


class ClsDataset(Dataset):
    def __init__(self, inputs, labels, cfg, augment: bool, seed: int):
        self.inputs, self.labels, self.cfg, self.augment = inputs, labels, cfg, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        img = self.inputs[i]
        if self.augment:
            c = self.cfg
            img, _ = augment_pair(img, None, self.rng, rotation_deg=c.rotation_deg, scale=c.scale,
                                  shift=c.shift, brightness=c.brightness, contrast=c.contrast)
        return to_tensor(img), torch.tensor([float(self.labels[i])])


@torch.no_grad()
def predict(model, inputs: np.ndarray, device, batch_size: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(inputs), batch_size):
        x = torch.stack([to_tensor(im) for im in inputs[s:s + batch_size]]).to(device)
        out.append(torch.sigmoid(model(x))[:, 0].cpu().numpy())
    return np.concatenate(out)


def train_one(cfg: Config, k: int, condition: str, inputs, labels, folds, device):
    cc = cfg.cls
    seed_everything(cfg.seed + k)  # same seed for both conditions of a fold
    tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
    g = torch.Generator().manual_seed(cfg.seed + k)
    loader = DataLoader(ClsDataset(inputs[tr], labels[tr], cc, True, cfg.seed + k),
                        batch_size=cc.batch_size, shuffle=True, generator=g, num_workers=cfg.num_workers)
    model = resnet18_binary(cc.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cc.lr, weight_decay=cc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cc.epochs)
    losses_per_epoch = []
    for epoch in range(1, cc.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        losses_per_epoch.append(float(np.mean(losses)))
        log(f"ResNet-18 [{condition:5s}] fold {k + 1} epoch {epoch:2d}/{cc.epochs}  loss {losses_per_epoch[-1]:.4f}")
    return model, predict(model, inputs[te], device), te, losses_per_epoch


def run(cfg: Config, images_u8, labels, folds, image_ids):
    device = get_device(cfg.device)
    cfg.cls_dir.mkdir(parents=True, exist_ok=True)
    log(f"ResNet-18 trainable parameters: {count_parameters(resnet18_binary(False)):,} (paper: 11.18 M)")
    oof = {c: np.full(len(labels), np.nan) for c in CONDITIONS}
    curves = {c: [] for c in CONDITIONS}
    timings = {c: [] for c in CONDITIONS}
    for k in range(cfg.n_folds):
        roi_masks = load_roi_masks(cfg, k, images_u8.shape)
        for condition in CONDITIONS:
            inputs = build_inputs(images_u8, condition, roi_masks, cfg.cls.input_size)
            with Timer() as t:
                model, p, te, losses = train_one(cfg, k, condition, inputs, labels, folds, device)
            oof[condition][te] = p
            curves[condition].append(losses)
            timings[condition].append(t.seconds / 60)
            torch.save(model.state_dict(), cfg.cls_dir / f"resnet18_{condition}_fold{k + 1}.pt")
            log(f"ResNet-18 [{condition}] fold {k + 1} done in {t.seconds / 60:.1f} min")

    with open(cfg.cls_dir / "oof_predictions.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "fold", "label", "p_whole", "p_roi"])
        for i in range(len(labels)):
            wr.writerow([image_ids[i], int(folds[i]) + 1, int(labels[i]),
                         f"{oof['whole'][i]:.6f}", f"{oof['roi'][i]:.6f}"])
    save_json({"train_loss": curves, "minutes_per_fold": timings}, cfg.cls_dir / "training_curves.json")

