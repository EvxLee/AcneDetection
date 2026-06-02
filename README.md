# Acne Detection & Cross-Domain Classification

Two-part computer vision project: acne lesion detection (Part 1) and cross-domain binary classification (Part 2).

**📊 Detailed Notion Report Pages On Findings:**
- **Part 1 Breakdown:** [Yang Lab Project Presentation](https://www.notion.so/Yang-Lab-Project-Presentation-35aa5f597a88808f902ff2727c5becb4?source=copy_link)
- **Part 2 Breakdown:** [Yang Lab Project Presentation 2](https://www.notion.so/Yang-Lab-Project-Presentation-2-35ca5f597a8880b7bbf7df8c6535e755?source=copy_link)

---

## Datasets

### ACNE04
1,450 facial images with bounding box annotations for acne lesions.
- **Roboflow:** https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/

### DermNet (Part 2 only)
23-class skin condition dataset used for cross-domain evaluation.
- **Kaggle:** https://www.kaggle.com/datasets/shubhamgoel27/dermnet
- Download and unzip — place the root folder at `data/dermnet/` so the structure is:
  ```
  data/dermnet/
  ├── train/   (23 subfolders)
  └── test/    (23 subfolders)
  ```

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

## Part 1 — Acne Detection

### Quick path: Google Colab (recommended)

Open `notebooks/part1_colab.ipynb` in Colab. Set runtime to **GPU**, then run all cells top-to-bottom. Handles data download, training, and evaluation end-to-end.

### Local GPU path: step-by-step notebooks

**Prerequisite:** Download ACNE04 first.

```bash
python part1_detection/roboflow_loader.py --download
```

This populates `data/acne04/train/`, `data/acne04/valid/`, and `data/acne04/test/` with images and COCO annotation JSON files.

Run notebooks in order:

| Notebook | What it does | Runtime |
|---|---|---|
| `01_data_explore.ipynb` | Verify data, class distribution, sample annotations | ~2 min CPU |
| `02_yolov8_train.ipynb` | Convert COCO → YOLOv8 format, fine-tune YOLOv8s | ~30 min GPU |
| `03_faster_rcnn_train.ipynb` | Fine-tune Faster R-CNN (ResNet-50 + FPN) | ~45 min GPU |
| `04_evaluate.ipynb` | mAP@50, mAP@50-95, precision, recall comparison | ~10 min GPU |
| `05_visualize.ipynb` | Side-by-side prediction grids | ~5 min GPU |

**Output files** (all written to `outputs/part1/`):

```
outputs/part1/
├── sample_annotations.png      # from 01
├── class_distribution.png      # from 01
├── yolov8_training_curves.png  # from 02
├── faster_rcnn_loss.png        # from 03
├── pr_curves.png               # from 04
├── iou_distribution.png        # from 04
├── evaluation_results.json     # from 04 (mAP, precision, recall)
└── detection_comparison.png    # from 05
```

---

## Part 2 — Cross-Domain Classification

### Quick path: Google Colab (recommended)

Open `notebooks/part2_colab.ipynb` in Colab. Set runtime to **A100 GPU**, then run all cells top-to-bottom. The notebook handles all setup, downloads, training, and evaluation end-to-end.

### Local GPU path: step-by-step notebooks

**Prerequisites:** ACNE04 already downloaded (`data/acne04/`) and DermNet placed at `data/dermnet/`.

Run notebooks in order:

| Notebook | What it does | Runtime |
|---|---|---|
| `06_patch_extraction.ipynb` | Crops acne patches (positive) and clear-skin patches (negative) from ACNE04 bounding boxes | ~5 min CPU |
| `07_train_classifier.ipynb` | Fine-tunes EfficientNet-B0 on ~22k patches for 20 epochs | ~20 min GPU |
| `08_domain_adaptation.ipynb` | Full adaptation pipeline: color norm, 2-stage few-shot fine-tuning, TTA, FaceNet; saves results | ~30 min GPU |
| `09_evaluate_dermnet.ipynb` | Lightweight standalone: baseline EfficientNet eval on DermNet test set only | ~5 min GPU |
| `10_gradcam_visualize.ipynb` | Grad-CAM heatmaps on 10 DermNet test images | ~2 min GPU |

**Output files** (all written to `outputs/part2/`):

```
outputs/part2/
├── sample_patches.png          # from 06
├── classifier_training_curves.png  # from 07
├── domain_gap.png              # from 08
├── rgb_channel_stats.png       # from 08
├── augmentation_examples.png   # from 08
├── color_normalization.png     # from 08
├── final_model_results.png     # from 08 (baseline vs full pipeline)
├── dermnet_results.json        # from 08 (full ablation metrics)
└── gradcam_dermnet.png         # from 10

outputs/classifier/
├── best.pth                    # ACNE04-trained checkpoint (from 07)
└── finetuned.pth               # DermNet-adapted checkpoint (from 08)
```

---

## Repo Structure

```
AcneDetection/
├── notebooks/
│   ├── part1_colab.ipynb           # Part 1 end-to-end (Colab, recommended)
│   ├── part2_colab.ipynb           # Part 2 end-to-end (Colab, recommended)
│   ├── 01_data_explore.ipynb       # Part 1 — local GPU
│   ├── 02_yolov8_train.ipynb
│   ├── 03_faster_rcnn_train.ipynb
│   ├── 04_evaluate.ipynb
│   ├── 05_visualize.ipynb
│   ├── 06_patch_extraction.ipynb   # Part 2 — local GPU
│   ├── 07_train_classifier.ipynb
│   ├── 08_domain_adaptation.ipynb
│   ├── 09_evaluate_dermnet.ipynb
│   └── 10_gradcam_visualize.ipynb
├── part1_detection/
│   └── roboflow_loader.py       # downloads ACNE04 via Roboflow SDK
├── data/                        # gitignored — populated at runtime
│   ├── acne04/                  # train/ valid/ test/ + COCO annotations
│   ├── dermnet/                 # populated manually from Kaggle (Part 2)
│   └── patches/                 # generated by 06_patch_extraction.ipynb
├── outputs/
│   ├── part1/                   # figures + evaluation_results.json
│   ├── part2/                   # figures + dermnet_results.json
│   └── classifier/              # best.pth + finetuned.pth
├── REPORT1.md                   # Part 1 writeup
├── REPORT2.md                   # Part 2 writeup
├── .env.example
├── requirements.txt
└── README.md
```

---
