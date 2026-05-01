# CLAUDE.md

This project is an interview assignment for Yanglab. Full brief below.

---

## Assignment Brief

This assignment consists of two parts:

### **Part 1: Acne Detection on ACNE04**

#### **Dataset**

- **ACNE04** is a publicly available dataset consisting of 1450 facial images with bounding box annotations for acne lesions.
- Dataset link: https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection/

#### **Tasks**

1. Train **at least two object detection models** on ACNE04 to localize acne lesions.

2. Select models from both classic and modern architectures. Suggested options include:
    - **YOLOv5**
    - **Faster R-CNN**
    - **DINO-DETR (2023)** – recent Transformer-based detector

3. Compare model performance using:
    - Mean Average Precision (mAP)
    - Precision / Recall
    - Intersection over Union (IoU)

4. Justify your model selection with 2–3 references to relevant literature or benchmarks.

5. Visualize several detection results using bounding boxes overlaid on sample images.

---

### **Part 2: Cross-Domain Acne Classification**

#### **Datasets**

- **ACNE04**: You create classification samples by yourself. For example: crop **positive patches** (around acne) and **negative patches** (clear skin) using bounding boxes.
- **DermNet**: Contains over 20 skin conditions. For this task:
    - Download the **DermNet test set**: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
    - Use only **20 unlabeled samples from the training set** during development to assess domain gap.
    - Evaluate your model on the full test set by predicting **"acne" vs "non-acne"**.

#### **Tasks**

1. Train a classification model using only ACNE04 patches.

2. Explore **domain adaptation techniques** to improve performance on DermNet, such as:
    - Color normalization, histogram matching
    - Image augmentation (brightness, blur, crop, zoom, etc.)
    - You can also use other pretrained networks on human face as starting weights
    - Optional: use style **transfer methods** (e.g. CycleGAN, bonus with advanced models)

3. Evaluate performance on the DermNet test set:
    - Accuracy, F1-score, and AUROC
    - Use Grad-CAM or similar techniques to visualize model attention

4. Submit visualizations of at least 10 predictions on DermNet with class activation maps.

5. Write a short reflection on how well your model transferred to this new domain and how you attempted to mitigate the domain gap.

---

## Submission Requirements

- Source code (organized and runnable)
- A 1-2 page document including:
    - Model selection rationale
    - Preprocessing steps
    - Training details and evaluation
    - Summary of findings and challenges
- A README.md explaining how to run your code

---

## Repo Structure

```
AcneDetection/
├── part1_detection/         # Part 1: object detection models + metrics
│   ├── roboflow_loader.py   # lazy data pull from Roboflow API
│   ├── yolov5_train.py
│   ├── faster_rcnn_train.py
│   ├── dino_detr_train.py
│   ├── metrics.py           # mAP, IoU, precision, recall
│   └── yolov5_data.yaml
├── part2_classification/    # Part 2: patch classifier + domain adaptation
│   ├── patch_extractor.py   # crop +/- patches from ACNE04 bboxes
│   ├── dermnet_loader.py    # DermNet binary dataset loader
│   ├── train.py             # EfficientNet-B0 classifier
│   ├── domain_adapt.py      # Reinhard norm, histogram match, augmentation
│   ├── gradcam.py           # Grad-CAM + grid export
│   └── metrics.py           # accuracy, F1, AUROC
├── data/                    # gitignored — populated by roboflow_loader.py
│   ├── acne04/
│   ├── dermnet/
│   └── patches/
├── .env.example
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Environment

- Roboflow workspace: `evan-lee-rrndd`
- Roboflow project: `acne04-detection-p8j0d`
- API key stored in `.env` (see `.env.example`)
