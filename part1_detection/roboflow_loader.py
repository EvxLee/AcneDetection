"""
Lazy Roboflow dataset loader — fetches image URLs from the API and
downloads images on demand instead of pulling all 1.42k at once.

Usage:
    python part1_detection/roboflow_loader.py            # prints dataset info
    python part1_detection/roboflow_loader.py --download # downloads a subset
"""

import os
import io
import json
import argparse
import requests
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

API_KEY     = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY")
WORKSPACE   = os.getenv("ROBOFLOW_WORKSPACE", "evan-lee-rrndd")
PROJECT     = os.getenv("ROBOFLOW_PROJECT",   "acne04-detection-p8j0d")
VERSION     = int(os.getenv("ROBOFLOW_VERSION", "1"))
FORMAT      = "coco"
SUBSET_SIZE = 100

BASE_URL = "https://api.roboflow.com"


def get_dataset_info() -> dict:
    url = f"{BASE_URL}/{WORKSPACE}/{PROJECT}/{VERSION}"
    resp = requests.get(url, params={"api_key": API_KEY})
    resp.raise_for_status()
    return resp.json()


def get_export_url(fmt: str = FORMAT) -> str:
    url = f"{BASE_URL}/{WORKSPACE}/{PROJECT}/{VERSION}/{fmt}"
    resp = requests.get(url, params={"api_key": API_KEY})
    resp.raise_for_status()
    data = resp.json()
    link = data.get("export", {}).get("link")
    if not link:
        raise RuntimeError(f"No export link returned. Response: {data}")
    return link


def list_images(split: str = "train") -> list[dict]:
    url = f"{BASE_URL}/{WORKSPACE}/{PROJECT}/{VERSION}/dataset"
    resp = requests.get(url, params={"api_key": API_KEY, "split": split})
    if resp.status_code == 404:
        print("  /dataset endpoint not available — falling back to export JSON")
        return _images_from_export(split)
    resp.raise_for_status()
    return resp.json().get("images", [])


def _images_from_export(split: str = "train") -> list[dict]:
    import zipfile
    export_link = get_export_url(fmt="coco")
    print("  Fetching annotation JSON from export link...")
    resp = requests.get(export_link)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    annotation_files = [n for n in zf.namelist() if n.endswith(".json")]
    if not annotation_files:
        raise RuntimeError("No JSON annotation file found in export zip.")
    target = next((f for f in annotation_files if split in f), annotation_files[0])
    coco = json.loads(zf.read(target))
    return list({img["id"]: img for img in coco.get("images", [])}.values())


class RoboflowDataset:
    """
    Lazy dataset — only annotation metadata is loaded on init.
    Images are fetched one at a time via __getitem__.
    """

    def __init__(self, split: str = "train"):
        print(f"Loading {split} metadata from Roboflow (no images downloaded)...")
        self.records = list_images(split)
        print(f"  Found {len(self.records)} images in '{split}' split.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict]:
        record = self.records[idx]
        url = record.get("image", record.get("coco_url", record.get("url", "")))
        if not url:
            raise ValueError(f"No image URL in record: {record}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img, record

    def __repr__(self) -> str:
        return f"RoboflowDataset(n={len(self)})"


def download_subset(split: str = "train", n: int = SUBSET_SIZE, out_dir: str = "data/acne04"):
    dataset = RoboflowDataset(split)
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    total = min(n, len(dataset))
    print(f"Downloading {total} images to {out}/...")
    for i in range(total):
        img, record = dataset[i]
        name = record.get("file_name", record.get("name", f"{i:05d}.jpg"))
        dest = out / Path(name).name
        img.save(dest)
        print(f"  [{i+1}/{total}] {dest.name}", end="\r")
    print(f"\nDone — {total} images saved to {out}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--n", type=int, default=SUBSET_SIZE)
    args = parser.parse_args()

    print("── Roboflow Dataset Info ──────────────────────────────")
    info = get_dataset_info()
    v = info.get("version", {})
    print(f"  Project : {info.get('project', {}).get('name', PROJECT)}")
    print(f"  Version : {VERSION}")
    print(f"  Images  : {v.get('images', '?')}")
    print(f"  Splits  : {v.get('splits', {})}")
    print("───────────────────────────────────────────────────────")

    if args.download:
        download_subset(split=args.split, n=args.n)
    else:
        ds = RoboflowDataset(args.split)
        print(ds)
        print("\nRun with --download --n 100 to save a subset locally.")


if __name__ == "__main__":
    main()
