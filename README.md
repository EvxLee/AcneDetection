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

# Download all splits at once (train + valid + test)
python part1_detection/roboflow_loader.py --download
```

After this, `data/acne04/train/`, `data/acne04/valid/`, and `data/acne04/test/` will contain the images and COCO annotation JSON files.

---

## Step 2 — Part 1: Detection

Run the notebooks in order:

```
part1_detection/
├── 01_data_explore.ipynb      # verify data, class distribution, sample visualisation
├── 02_yolov8_train.ipynb      # convert COCO → YOLOv8 format, fine-tune yolov8s
├── 03_faster_rcnn_train.ipynb # fine-tune Faster R-CNN (ResNet-50 + FPN)
├── 04_evaluate.ipynb          # mAP@50, mAP@50-95, precision, recall comparison
└── 05_visualize.ipynb         # side-by-side prediction grids
```

---

## Step 3 — Part 2: Classification

Train a binary classifier (acne vs. non-acne) on patches cropped from ACNE04, then evaluate cross-domain on DermNet. Includes domain adaptation and Grad-CAM visualizations.

*(Coming after Part 1 is complete.)*

---

## Structure

```
AcneDetection/
├── part1_detection/
│   └── roboflow_loader.py    # downloads ACNE04 via Roboflow SDK
├── part2_classification/     # coming after Part 1
├── data/                     # gitignored — populated at runtime
│   ├── acne04/               # train/ valid/ test/ + COCO annotations
│   └── dermnet/              # populated manually from Kaggle (Part 2)
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
