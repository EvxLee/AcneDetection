# Acne Detection & Cross-Domain Classification

Two-part computer vision project: acne lesion detection (Part 1) and cross-domain binary classification (Part 2).

---

## Datasets

### ACNE04
1,450 facial images with bounding box annotations for acne lesions.
- **Roboflow:** https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/

### DermNet (Part 2 only)
23-class skin condition dataset for cross-domain evaluation.
- **Kaggle:** https://www.kaggle.com/datasets/shubhamgoel27/dermnet
- Download and place at `data/dermnet/`

---

## How the data pipeline works

Images are hosted on Roboflow's servers — nothing is stored in this repo. The `data/` folder starts empty and gets populated when you run the loader.

```
Roboflow's servers             Your machine
──────────────────             ──────────────────────────────
1,450 images + labels  →  roboflow_loader.py  (uses your API key)
                                   │
                          Step 1: fetch metadata only
                          (a lightweight JSON list of image
                          URLs + bounding box coords — fast,
                          no images downloaded yet)
                                   │
                          Step 2: download images on demand
                          OR batch-download to data/acne04/
                                   │
                          Step 3 (later): feed into a model
```

`roboflow_loader.py` has two modes:
- **Lazy mode** (default) — loads just the URL list. Call `ds[0]` to fetch one image at a time over the internet. Good for quick inspection.
- **Download mode** (`--download`) — saves image files to `data/acne04/` so you can work offline and train at full speed.

---

## Setup

```bash
git clone https://github.com/EvxLee/AcneDetection.git
cd AcneDetection
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and fill in your Roboflow API key (get it at app.roboflow.com → Account → API Keys):

```
ROBOFLOW_API_KEY=<your key>
ROBOFLOW_WORKSPACE=evan-lee-rrndd
ROBOFLOW_PROJECT=acne04-detection-p8j0d
ROBOFLOW_VERSION=1
```

---

## Step 1 — Access the data

```bash
# Check dataset info (no images downloaded)
python part1_detection/roboflow_loader.py

# Download all images locally
python part1_detection/roboflow_loader.py --download --split train --n 1450
python part1_detection/roboflow_loader.py --download --split valid --n 1450
python part1_detection/roboflow_loader.py --download --split test  --n 1450
```

After this, `data/acne04/train/`, `data/acne04/valid/`, and `data/acne04/test/` will contain the images and COCO annotation JSON files.

---

## Step 2 — Part 1: Detection

Train at least two object detection models on ACNE04 to localize acne lesions. Compare using mAP, Precision, Recall, and IoU.

*(Training scripts coming as we build them out.)*

---

## Step 3 — Part 2: Classification

Train a binary classifier (acne vs. non-acne) on patches cropped from ACNE04, then evaluate cross-domain on DermNet. Includes domain adaptation and Grad-CAM visualizations.

*(Coming after Part 1 is complete.)*

---

## Structure

```
AcneDetection/
├── part1_detection/
│   └── roboflow_loader.py    # lazy ACNE04 data pull via Roboflow API
├── part2_classification/     # coming soon
├── data/                     # gitignored — populated by roboflow_loader.py
│   ├── acne04/
│   └── dermnet/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Results

**Part 1 — Detection (mAP@50)**

| Model | mAP@50 | Precision | Recall |
|-------|--------|-----------|--------|
| YOLOv5 | — | — | — |
| Faster R-CNN | — | — | — |
| DINO-DETR | — | — | — |

**Part 2 — Classification on DermNet test set**

| Method | Accuracy | F1 | AUROC |
|--------|----------|----|-------|
| No adaptation | — | — | — |
| + Histogram match | — | — | — |
| + Reinhard norm | — | — | — |

---

## References

1. Redmon & Farhadi. *YOLOv5.* Ultralytics, 2020.
2. Ren et al. *Faster R-CNN.* NeurIPS 2015.
3. Zhang et al. *DINO: DETR with Improved DeNoising Anchor Boxes.* ICLR 2023.
