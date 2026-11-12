"""Command-line entry point.

    python -m tbstudy prepare   --data-dir data/MontgomerySet     # cache 256x256 images, masks, readings, folds
    python -m tbstudy segment                                     # stage 1: fold-specific U-Nets + Otsu baseline
    python -m tbstudy classify                                    # stage 2: ResNet-18, whole image vs lung ROI
    python -m tbstudy analyze                                     # stage 3: tables, statistics, figures
    python -m tbstudy all       --data-dir data/MontgomerySet     # everything above
    python -m tbstudy smoke                                       # 2-minute end-to-end check on synthetic data
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from .config import Config, smoke_config
from .data import load_cache, make_folds, preprocess, save_cache
from .utils import log, save_json


def cmd_prepare(cfg: Config):
    log(f"Reading Montgomery set from {cfg.data_dir}")
    images, masks, records = preprocess(cfg.data_dir, cfg.preprocess_size)
    save_cache(cfg.cache_file, cfg.meta_file, images, masks, records)
    labels = np.array([r.label for r in records])
    folds = make_folds(labels, cfg.n_folds, cfg.seed)
    with open(cfg.out_dir / "folds.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "label", "fold"])
        for r, k in zip(records, folds):
            wr.writerow([r.image_id, r.label, int(k) + 1])
    save_json(cfg.to_dict(), cfg.out_dir / "config.json")
    log(f"{len(records)} images ({labels.sum()} TB, {(labels == 0).sum()} normal) cached to {cfg.cache_file}")


def _load(cfg: Config):
    if not cfg.cache_file.exists():
        sys.exit(f"{cfg.cache_file} not found - run `python -m tbstudy prepare` first.")
    images, masks, records = load_cache(cfg.cache_file, cfg.meta_file)
    labels = np.array([r.label for r in records])
    folds = make_folds(labels, cfg.n_folds, cfg.seed)
    return images, masks, records, labels, folds


def cmd_segment(cfg: Config):
    from . import train_seg
    images, masks, _, _, folds = _load(cfg)
    train_seg.run(cfg, images, masks, folds)


def cmd_classify(cfg: Config):
    from . import train_cls
    images, _, records, labels, folds = _load(cfg)
    train_cls.run(cfg, images, labels, folds, [r.image_id for r in records])


def cmd_analyze(cfg: Config, gradcam: bool = True):
    from . import analyze
    images, masks, records, _, folds = _load(cfg)
    analyze.run(cfg, images, masks, records, folds, with_gradcam=gradcam)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tbstudy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "segment", "classify", "analyze", "all", "smoke"])
    ap.add_argument("--data-dir", type=Path, default=Path("data/MontgomerySet"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = library default)")
    ap.add_argument("--no-gradcam", action="store_true")
    args = ap.parse_args(argv)

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    if args.command == "smoke":
        from .synthetic import make_synthetic_montgomery
        root = args.out_dir / "smoke"
        data_dir = make_synthetic_montgomery(root / "data")
        cfg = smoke_config(data_dir, root / "outputs")
        cfg.device = args.device
        for step in (cmd_prepare, cmd_segment, cmd_classify):
            step(cfg)
        cmd_analyze(cfg, gradcam=not args.no_gradcam)
        log("Smoke test finished - the pipeline runs end to end (numbers on synthetic data are meaningless).")
        return

    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir, device=args.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if args.command in ("prepare", "all"):
        cmd_prepare(cfg)
    if args.command in ("segment", "all"):
        cmd_segment(cfg)
    if args.command in ("classify", "all"):
        cmd_classify(cfg)
    if args.command in ("analyze", "all"):
        cmd_analyze(cfg, gradcam=not args.no_gradcam)


if __name__ == "__main__":
    main()
