"""Download the Montgomery County CXR set from the U.S. National Library of Medicine.

    python scripts/download_montgomery.py --out data/MontgomerySet

Reads the public NLM listing pages and downloads CXR_png/, ManualMask/{left,right}Mask/ and
ClinicalReadings/ (about 600 MB). Already-downloaded files are skipped, so you can re-run it after
an interruption. Use --verbose to see the exact error if a listing page cannot be read.

Manual alternative: open
https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html
and save the files so that data/MontgomerySet/CXR_png/, ManualMask/leftMask/, ManualMask/rightMask/
and ClinicalReadings/ exist.

Please cite the dataset (Jaeger et al., Quant Imaging Med Surg 2014; Candemir et al., IEEE TMI 2014).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/"
FOLDERS = ["CXR_png/", "ClinicalReadings/", "ManualMask/leftMask/", "ManualMask/rightMask/"]
EXPECTED = {"CXR_png/": 138, "ClinicalReadings/": 138, "ManualMask/leftMask/": 138, "ManualMask/rightMask/": 138}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/130.0 Safari/537.36 tbstudy-downloader"}
LINK = re.compile(r"""href\s*=\s*["']?([^"'\s>]+?\.(?:png|txt))["'\s>]""", re.I)
NAME = re.compile(r"MCUCXR_\d{4}_[01]\.(?:png|txt)", re.I)


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_files(url: str, verbose: bool) -> list[str]:
    errors = []
    for candidate in (url + "index.html", url):
        try:
            html = fetch(candidate).decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{candidate}: {type(e).__name__}: {e}")
            continue
        names = {m.split("/")[-1] for m in LINK.findall(html)}
        if not names:  # fall back to any Montgomery file name in the page text
            names = set(NAME.findall(html))
        names = {n for n in names if NAME.fullmatch(n)}
        if names:
            return sorted(names)
        errors.append(f"{candidate}: page read ({len(html)} chars) but no file links found")
    for e in errors:
        print("   ", e)
    return []


def download(url: str, target: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            data = fetch(url)
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {e}") from e
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/MontgomerySet"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    for folder in FOLDERS:
        url = urljoin(BASE, folder)
        names = list_files(url, args.verbose)
        if not names:
            sys.exit(f"\nCould not list {url}\nSee the errors above, or download manually (see this script's docstring).")
        if len(names) != EXPECTED[folder]:
            print(f"   note: {folder} lists {len(names)} files (expected {EXPECTED[folder]})")
        dest = args.out / folder
        dest.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names, 1):
            target = dest / name
            if not (target.exists() and target.stat().st_size > 0):
                download(urljoin(url, name), target)
            print(f"\r{folder:24s} {i:3d}/{len(names)}", end="", flush=True)
        print(f"\r{folder:24s} {len(names)} files      ")
    n_img = len(list((args.out / "CXR_png").glob("*.png")))
    print(f"Done: {n_img} radiographs in {args.out} (expected 138).")


if __name__ == "__main__":
    main()
