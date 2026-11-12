"""Montgomery County CXR set: discovery, pre-processing, clinical readings and folds.

Expected layout (as distributed by the U.S. National Library of Medicine)::

    MontgomerySet/
        CXR_png/MCUCXR_0001_0.png            # suffix _0 = normal, _1 = TB
        ManualMask/leftMask/MCUCXR_0001_0.png
        ManualMask/rightMask/MCUCXR_0001_0.png
        ClinicalReadings/MCUCXR_0001_0.txt
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class Record:
    image_id: str
    label: int                 # 1 = TB, 0 = normal
    sex: str                   # "M", "F" or ""
    age: float                 # years, NaN if missing
    landscape: bool            # original acquisition wider than tall
    reading: str               # free-text finding


# --------------------------------------------------------------------------- discovery
def find_images(data_dir: Path) -> list[Path]:
    img_dir = Path(data_dir) / "CXR_png"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"{img_dir} not found. Download the Montgomery set first (see README, 'Data')."
        )
    paths = sorted(p for p in img_dir.glob("*.png") if re.match(r".+_[01]\.png$", p.name))
    if not paths:
        raise FileNotFoundError(f"No *_0.png / *_1.png images in {img_dir}")
    return paths


def label_from_name(name: str) -> int:
    m = re.search(r"_([01])\.(png|txt)$", name)
    if not m:
        raise ValueError(f"Cannot read label from file name {name!r}")
    return int(m.group(1))


def parse_reading(text: str) -> tuple[str, float, str]:
    """Parse a Montgomery clinical reading -> (sex, age_years, finding text)."""
    sex, age = "", float("nan")
    finding_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m_sex = re.search(r"sex\s*:\s*([MF])", s, re.I)
        m_age = re.search(r"age\s*:\s*0*(\d+)\s*([YM]?)", s, re.I)
        if m_sex:
            sex = m_sex.group(1).upper()
        elif m_age:
            value = float(m_age.group(1))
            age = value / 12.0 if m_age.group(2).upper() == "M" else value
        else:
            finding_lines.append(s)
    return sex, age, " ".join(finding_lines)


# --------------------------------------------------------------------------- image IO
def read_cxr(path: Path) -> np.ndarray:
    """Read a (possibly 12/16-bit) PNG and return an 8-bit greyscale array at full resolution."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 255:  # 12-/16-bit data: rescale the used range to 8 bits
        lo, hi = float(img.min()), float(img.max())
        img = (img - lo) / max(hi - lo, 1.0) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def read_mask(data_dir: Path, name: str) -> np.ndarray:
    mask = None
    for side in ("leftMask", "rightMask"):
        p = Path(data_dir) / "ManualMask" / side / name
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise IOError(f"Missing lung mask {p}")
        mask = (m > 0) if mask is None else (mask | (m > 0))
    return mask


def preprocess(data_dir: Path, size: int = 256) -> tuple[np.ndarray, np.ndarray, list[Record]]:
    """[paper] 8-bit greyscale, bilinear resize to size x size; L+R masks merged, nearest-neighbour resize."""
    images, masks, records = [], [], []
    for p in find_images(data_dir):
        img = read_cxr(p)
        h, w = img.shape
        msk = read_mask(data_dir, p.name)
        images.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR))
        masks.append(cv2.resize(msk.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0)
        txt_path = Path(data_dir) / "ClinicalReadings" / p.with_suffix(".txt").name
        text = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""
        sex, age, finding = parse_reading(text)
        records.append(Record(p.stem, label_from_name(p.name), sex, age, w > h, finding))
    return np.stack(images), np.stack(masks), records


def save_cache(cache_file: Path, meta_file: Path, images, masks, records: list[Record]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, images=images, masks=masks,
                        labels=np.array([r.label for r in records]))
    with open(meta_file, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "label", "sex", "age", "landscape", "reading"])
        for r in records:
            wr.writerow([r.image_id, r.label, r.sex, "" if np.isnan(r.age) else r.age, int(r.landscape), r.reading])


def load_cache(cache_file: Path, meta_file: Path):
    d = np.load(cache_file)
    records = []
    with open(meta_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(Record(row["image_id"], int(row["label"]), row["sex"],
                                  float(row["age"]) if row["age"] else float("nan"),
                                  bool(int(row["landscape"])), row["reading"]))
    return d["images"], d["masks"].astype(bool), records


# --------------------------------------------------------------------------- folds
def make_folds(labels: np.ndarray, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """[paper] One stratified split into five folds with seed 42. Returns fold index (0..n-1) per image."""
    folds = np.full(len(labels), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        folds[test_idx] = k
    return folds


def keyword_flags(readings: list[str], keywords: tuple[str, ...]) -> np.ndarray:
    pat = re.compile("|".join(re.escape(k) for k in keywords), re.I)
    return np.array([bool(pat.search(r or "")) for r in readings])
