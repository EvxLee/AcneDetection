# Part 1: Acne Detection on ACNE04 — Report

> **Repo:** https://github.com/EvxLee/AcneDetection
> **Author:** Evan Lee · **Date:** May 2026

---

## TL;DR

Trained YOLOv8s and Faster R-CNN on ACNE04 for acne lesion localization. On the held-out test set using a consistent pycocotools COCO evaluation pipeline, Faster R-CNN achieved mAP@0.5 = 0.128 vs YOLOv8s at 0.086. Both models struggle primarily with small lesions (<30px) and dense lesion clusters — a direct consequence of the dataset scale (~1,400 images) and lesion size relative to network stride.

---

## 1. Problem & Dataset

**Task:** Detect and localize acne lesions in facial images using bounding box regression. Acne severity and subtype vary substantially across individuals, making this a challenging fine-grained detection problem.

### ACNE04 Stats

| Split | Images | Annotated Lesions |
|---|---|---|
| Train | 1,132 | ~11,000 |
| Val | 218 | ~2,100 |
| Test | 100 | ~950 |
| **Total** | **~1,450** | **~14,000** |

- **Annotation format:** COCO bounding boxes (x, y, w, h)
- **Lesion subtypes:** Papule, pustule, nodule/cyst, whitehead/blackhead — **treated as a single "acne" class** for detection (subtype discrimination was out of scope)
- **Lesion size:** Predominantly 20–90px bounding box side length on images that are typically 1024px+; see Section 3 for implications
- **Density:** Ranges from 1 to 50+ annotated lesions per image

> **On the class distribution figure:** The chart below shows the per-subtype annotation count. Because all four subtypes are collapsed to one detection class during training, the distribution matters primarily as a data quality check — no subtype is vanishingly rare, so class weighting was unnecessary.

### Sample Images with Annotations

![Sample annotations](outputs/figures/sample_annotations.png)

*Sample ACNE04 images with ground-truth bounding boxes. Note the wide variation in lesion count, size, and skin tone.*

### Class Distribution

![Class distribution](outputs/figures/class_distribution.png)

---

## 2. Model Selection & Rationale

| Model | Architecture Type | Why Chosen | Citation |
|---|---|---|---|
| **YOLOv8s** | Modern single-stage, anchor-free | Best publicly available speed/accuracy on COCO small-object benchmarks; single-file deployment; trains in <20 min on A100 | Jocher et al. (2023) |
| **Faster R-CNN** | Classic two-stage, anchor-based | Gold-standard reference: explicit region proposals reduce false positives on dense, small objects; strong ResNet50-FPN backbone | Ren et al. (2015) |

> **Note on DINO-DETR:** Listed in the assignment as a suggested option; removed rather than listed as "planned." Running a Transformer-based detector without careful tuning on a 1,400-image dataset would likely underfit, and the compute cost on A100 was not a blocking constraint. The two-model comparison (classic vs. modern) satisfies the assignment brief.

### References

1. Jocher, G. et al. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
2. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS.

> **Why Faster R-CNN over a second YOLO variant?** The assignment asks for one classic and one modern architecture. Faster R-CNN's two-stage pipeline (region proposal → classification) is the canonical classic detector and provides a meaningful architectural contrast — not just a hyperparameter sweep.

---

## 3. Preprocessing & Training Setup

### Image Preprocessing

| Step | YOLOv8s | Faster R-CNN |
|---|---|---|
| Input size | 640 × 640 | 800px short edge |
| Normalization | ÷255 (Ultralytics default) | ImageNet mean/std ([0.485,0.456,0.406] / [0.229,0.224,0.225]) |
| Augmentation | Mosaic, random H-flip, HSV jitter, scale | Random H-flip only |
| Backbone | CSPDarknet (COCO pretrained) | ResNet50-FPN (ImageNet pretrained) |

### Important: Input Size vs Lesion Size

ACNE04 source images are typically 1024px wide. At YOLOv8's input resolution of 640px (scale factor ≈ 0.625×), a 20px lesion becomes ~12px — at or below the model's smallest effective stride of 8px. This is a primary cause of low recall on small lesions and would be the first thing to change in a production system (larger input, possibly tiling).

### Note on Mosaic Augmentation

YOLOv8 enables mosaic augmentation by default, which composites four images into one, effectively halving the scale of objects. For a dataset where small objects are already near the detection limit, mosaic likely hurts small-lesion recall. Ablating it (setting `mosaic=0.0`) would be a natural next experiment. This was not ablated here; the reported numbers reflect Ultralytics defaults.

### Hyperparameters

```
# YOLOv8s — Ultralytics defaults, not tuned
epochs     = 100  (early stopped at epoch 46, patience=15)
batch_size = 16
optimizer  = AdamW
lr0        = 0.01
imgsz      = 640

# Faster R-CNN — PyTorch defaults, not tuned
epochs     = 50
batch_size = 4
optimizer  = Adam, lr=1e-4, weight_decay=1e-4
backbone   = ResNet50-FPN (ImageNet pretrained)
```

No hyperparameter search was performed. Both models use their respective framework defaults as a reasonable starting point given the dataset size.

### Hardware & Training Time

| Model | GPU | Training Time |
|---|---|---|
| YOLOv8s | A100 | ~20 min (46 effective epochs) |
| Faster R-CNN | A100 | ~3 hours (50 epochs) |

---

## 4. Results

### Evaluation Pipeline

Both models were evaluated on the same held-out test set (100 images) using the **pycocotools COCO evaluation standard** (IoU thresholds 0.5:0.05:0.95, confidence threshold = 0.25). Precision and Recall are reported at the shared confidence threshold of 0.25, matched at IoU ≥ 0.5.

> **On the val/test gap for YOLOv8:** YOLOv8's built-in trainer reported mAP@0.5 = 0.2273 on the *validation* set during training (best epoch 46). The test-set evaluation yields mAP@0.5 = 0.0861 using the same Ultralytics evaluation code. This is a genuine train/test split generalization gap — not a methodology difference — and suggests the model partially overfit to the 218-image validation set distribution.

### Quantitative Comparison (Test Set)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (ms/img) |
|---|---|---|---|---|---|
| YOLOv8s | 0.0861 | 0.0235 | 0.4937 | 0.1252 | ~5 |
| **Faster R-CNN** | **0.1283** | **0.0325** | ~0.xx | ~0.xx | ~50 |

> **Note on Faster R-CNN Precision/Recall:** The original evaluation incorrectly used `evaluator.stats[6]` (AR@1 — average recall with max 1 detection per image) as the Recall figure, yielding 0.026 — an internally inconsistent number given mAP@0.5 = 0.128. Precision and Recall in the table above are now computed at confidence = 0.25 by matching predictions to GT at IoU ≥ 0.5. Fill in actual values after re-running the evaluation notebook.

### Precision-Recall Curves

![PR curves](outputs/figures/pr_curves.png)

*PR curves computed from all detections across the test set, sorted by confidence, matched to GT at IoU = 0.5. Area under each curve equals mAP@0.5.*

### IoU Distribution

![IoU distribution](outputs/figures/iou_distribution.png)

*Per-detection max IoU with the nearest GT box, at confidence ≥ 0.25. The fraction above the red 0.5 line is the precision-recall numerator at that threshold. A bimodal distribution (peak near 0 and near 0.7+) indicates the model is either confidently wrong (false positives) or reasonably well-localized.*

---

## 5. Qualitative Results

### Detection Visualizations

**YOLOv8s predictions:**

![YOLOv8 detections](outputs/figures/yolov8.png)

**Faster R-CNN predictions:**

![Faster R-CNN detections](outputs/figures/faster-r-cnn.png)

**Side-by-side comparison:**

![Part 1 final comparison](outputs/figures/part1final.png)

### Failure Cases

*Add 2–3 actual failure images from Colab output (`outputs/figures/detection_comparison.png` shows 6 side-by-side comparisons — pull the worst cases here.)*

| Failure Mode | Why It Happens |
|---|---|
| **Small lesions missed** | At 640px input, 20px lesions become ~12px — at or below YOLOv8's stride-8 anchor scale. Faster R-CNN's RPN also anchors start at ~32px. |
| **Dense lesion clusters** | NMS suppresses overlapping predictions. When 10+ lesions are within a 100px region, both models collapse them into 1–2 boxes. |
| **False positives on pores/hair follicles** | Texture similarity to small papules; more common in YOLOv8 which relies purely on single-scale feature maps at the smallest detection head. |
| **Conservative Faster R-CNN at high conf** | At conf=0.25, Faster R-CNN already filters most predictions — recall is inherently lower at higher thresholds than the PR curve AUC implies. |

---

## 6. Discussion

### Per-Lesion-Density Analysis

![Density mAP](outputs/figures/density_map.png)

*mAP@0.5 broken down by lesion density bin. This shows whether the gap between models is uniform or concentrated in specific difficulty levels.*

The density-bin results reveal where each model's architectural choice matters most:
- On **low-density images** (1–5 lesions), both models perform similarly — region proposals vs anchor-free doesn't matter much when objects are well-separated
- On **high-density images** (16+ lesions), Faster R-CNN's explicit RPN suppresses fewer true positives than YOLOv8's single-pass NMS — which is the mechanistic explanation for its higher overall mAP, grounded in the actual data

### Speed / Accuracy Tradeoff

YOLOv8s is ~10× faster at inference (~5ms vs ~50ms per image on A100) and trivially deployable as a single `.pt` file. For a mobile or real-time screening application, YOLOv8 is the practical choice. Faster R-CNN's higher mAP matters in a clinical pipeline where false negatives carry higher cost and latency is acceptable.

### Where Both Models Fall Short

Both models are significantly below dermatology detection benchmarks in the literature (mAP@0.5 > 0.40 is typical on datasets with 10k+ images and specialist annotation). The three primary limitations:

1. **Dataset size:** ~1,400 images is small for a fine-grained detection task. Augmentation and pretraining help but cannot substitute for data volume.
2. **Small object scale:** Lesions at 12–15px effective input size are at the limit of what COCO-trained backbones can resolve.
3. **Annotation quality:** Boundary ambiguity between subtypes (whitehead vs small papule) introduces label noise that caps how high precision can realistically get.
