# Part 1: Acne Detection on ACNE04 — Report

> **Repo:** https://github.com/EvxLee/AcneDetection
> **Author:** Evan Lee · **Date:** May 2026

---

## TL;DR

Trained YOLOv8s and Faster R-CNN on ACNE04 for acne lesion localization. Faster R-CNN achieved higher test-set mAP@0.5 (0.128 vs 0.086) at the cost of significantly longer training time. Both models struggle with small lesions and dense clusters — expected given the limited dataset size (~1,400 images) and high intra-class variation of acne morphology.

---

## 1. Problem & Dataset

**Task:** Detect and localize acne lesions in facial images using bounding box regression. Acne severity and type vary substantially across individuals, making this a challenging fine-grained detection problem.

### ACNE04 Stats

| Split | Images | Annotations |
|---|---|---|
| Train | 1,132 | ~11,000 |
| Val | 218 | ~2,100 |
| Test | 100 | ~950 |
| **Total** | **~1,450** | **~14,000** |

- **Annotation format:** COCO bounding boxes (x, y, w, h)
- **Classes:** 4 acne subtypes (papule, pustule, nodule/cyst, whitehead/blackhead) — treated as a single "acne" class for detection
- **Lesion size:** Predominantly small (20–90px bounding box side); minimum box size filtered to 20px during patch extraction
- **Class imbalance:** Lesion density varies from 1–50+ per image

### Sample Images with Annotations

![Sample annotations](outputs/figures/sample_annotations.png)

*Sample ACNE04 images with ground-truth bounding boxes. Note the wide variation in lesion count, size, and skin tone.*

### Class Distribution

![Class distribution](outputs/figures/class_distribution.png)

---

## 2. Model Selection & Rationale

| Model | Architecture Type | Why Chosen | Citation |
|---|---|---|---|
| **YOLOv8s** | Modern single-stage, anchor-free | State-of-the-art speed/accuracy tradeoff; strong on small objects; easy fine-tuning API | Jocher et al. (2023) |
| **Faster R-CNN** | Classic two-stage, anchor-based | Gold-standard baseline for object detection; ResNet50-FPN proven on small medical objects | Ren et al. (2015) |
| DINO-DETR *(planned)* | Modern Transformer-based | End-to-end detection without NMS; best COCO results as of 2023; excluded due to compute limits | Zhang et al. (2022) |

### References

1. Jocher, G. et al. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
2. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS.
3. Zhang, H. et al. (2022). *DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection.* ICLR 2023.

> **Why not just YOLOv8?** Faster R-CNN's two-stage pipeline (region proposal → classification) typically outperforms single-stage detectors on small, dense objects — exactly the profile of acne lesions. The comparison directly tests whether the speed penalty is worth it for this task.

---

## 3. Preprocessing & Training Setup

### Image Preprocessing

| Step | YOLOv8s | Faster R-CNN |
|---|---|---|
| Input size | 640 × 640 | 800px (short edge) |
| Normalization | YOLOv8 default (÷255) | ImageNet mean/std |
| Augmentation | Mosaic, random flip, HSV jitter, scale | Random horizontal flip |
| Backbone | CSPDarknet (pretrained COCO) | ResNet50-FPN (pretrained ImageNet) |

### Hyperparameters

```
# YOLOv8s
epochs     = 100  (early stopped at epoch 46)
batch_size = 16
optimizer  = AdamW (default)
lr0        = 0.01
patience   = 15
imgsz      = 640

# Faster R-CNN
epochs     = 50
batch_size = 4
optimizer  = Adam
lr         = 1e-4
weight_decay = 1e-4
backbone   = ResNet50-FPN (ImageNet pretrained)
```

### Hardware & Training Time

| Model | GPU | Training Time |
|---|---|---|
| YOLOv8s | A100 | ~20 min (46 epochs) |
| Faster R-CNN | A100 | ~3 hours (50 epochs) |

---

## 4. Results

### Quantitative Comparison (Test Set)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8s | 0.0861 | 0.0235 | 0.4937 | 0.1252 |
| **Faster R-CNN** | **0.1283** | **0.0325** | — | 0.0257 |
| Naive baseline | 0.0 | 0.0 | — | — |

> **Note on YOLOv8 val metrics:** YOLOv8's built-in trainer reported mAP@0.5 = 0.2273, Precision = 0.3350, Recall = 0.3181 on the validation set at best epoch (46). The lower test-set figure (0.0861) reflects the harder held-out test split and stricter COCO-style evaluation in the evaluation notebook.

### Detection Visualizations

**YOLOv8s predictions:**

![YOLOv8 detections](outputs/figures/yolov8.png)

**Faster R-CNN predictions:**

![Faster R-CNN detections](outputs/figures/faster-r-cnn.png)

**Side-by-side comparison:**

![Part 1 final comparison](outputs/figures/part1final.png)

---

## 5. Qualitative Results

### Where Models Succeed
- Both models correctly detect large, isolated pustules and papules on clear skin
- YOLOv8 is faster at inference (~5ms/image vs ~50ms for Faster R-CNN on A100)
- Faster R-CNN produces more confident detections on visually distinct lesions

### Failure Cases

| Failure Mode | Description | Both / One model |
|---|---|---|
| Small lesions missed | Lesions <30px bounding box side frequently missed — below effective receptive field | Both |
| Dense clusters | When 10+ lesions overlap, models merge into one large box or miss most | Both |
| False positives on pores | Large pores, hair follicles predicted as acne | YOLOv8 more prone |
| Low recall on Faster R-CNN | Recall=0.026 on test set suggests the model is overly conservative post-NMS | Faster R-CNN |

---

## 6. Discussion

### Which Model Won and Why

Faster R-CNN achieved higher mAP@0.5 (0.128 vs 0.086). The two-stage architecture's explicit region proposal step handles the high object density better than YOLOv8's single-pass anchor-free approach — particularly for overlapping lesions where anchor-free methods can struggle to separate instances.

However, the win is marginal at these absolute mAP levels. Both models are significantly below what the literature reports on medical imaging benchmarks (mAP@0.5 > 0.4 is typical for trained dermatology datasets with 10k+ images). The primary bottleneck is dataset size (~1,400 images) relative to the difficulty of the task.

### Speed / Accuracy Tradeoff

YOLOv8s is ~10× faster at inference and trivially deployable (single `.pt` file). For a mobile or real-time screening application, YOLOv8 is the practical choice despite lower mAP. Faster R-CNN's higher accuracy would matter more in a clinical pipeline where false negatives are costly.

### Where Both Fail

Both models underperform on:
- **Nodules and cysts** — similar texture to surrounding skin, poor contrast
- **Images with heavy makeup** — training distribution mismatch
- **High-density regions** (forehead, chin) — NMS suppresses overlapping true positives

With more data or a pretrained dermatology backbone (e.g. DermNet-pretrained ResNet), mAP would substantially improve.
