"""
Downloads the ACNE04 dataset from Roboflow using the official SDK.

Usage:
    python part1_detection/roboflow_loader.py            # dry run — prints dataset info
    python part1_detection/roboflow_loader.py --download # downloads to data/acne04/

Requires:
    pip install roboflow python-dotenv
"""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()

API_KEY   = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "evan-lee-rrndd")
PROJECT   = os.getenv("ROBOFLOW_PROJECT",   "acne04-detection-p8j0d")
VERSION   = int(os.getenv("ROBOFLOW_VERSION", "1"))
FORMAT    = "coco"
OUT_DIR   = "data/acne04"


def get_project():
    from roboflow import Roboflow
    rf = Roboflow(api_key=API_KEY)
    return rf.workspace(WORKSPACE).project(PROJECT)


def print_info():
    project = get_project()
    v = project.version(VERSION)
    print("── Roboflow Dataset Info ──────────────────────────────")
    print(f"  Project : {project.name}")
    print(f"  Version : {VERSION}")
    print(f"  Classes : {list(project.classes.keys()) if hasattr(project, 'classes') else '—'}")
    print("───────────────────────────────────────────────────────")
    return v


def download():
    print(f"Downloading ACNE04 (COCO format) to {OUT_DIR}/ ...")
    v = get_project().version(VERSION)
    dataset = v.download(FORMAT, location=OUT_DIR, overwrite=True)
    print(f"\nDone. Dataset saved to: {dataset.location}")
    print("Contents:")
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset.location, split)
        if os.path.exists(split_dir):
            n = len([f for f in os.listdir(split_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
            print(f"  {split}/  — {n} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download full dataset to data/acne04/")
    args = parser.parse_args()

    if args.download:
        download()
    else:
        print_info()
        print("\nRun with --download to pull all images locally.")


if __name__ == "__main__":
    main()
