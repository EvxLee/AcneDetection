# Part 1: Acne Detection on ACNE04 — Report

> **Repo:** https://github.com/EvxLee/AcneDetection
> **Author:** Evan Lee · **Date:** May 2026

---

## TL;DR

Trained YOLOv8s and Faster R-CNN on ACNE04 for acne lesion localization. On the held-out test set (142 images) using a consistent pycocotools COCO evaluation pipeline, Faster R-CNN achieved mAP@0.5 = 0.156 vs YOLOv8s at 0.057. The models exhibit opposite failure modes: YOLOv8 is highly conservative (P=0.54, R=0.08) while Faster R-CNN finds far more lesions with lower precision (P=0.32, R=0.45). Both models struggle primarily with small lesions (<30px) and dense lesion clusters — a direct consequence of the dataset scale (~1,400 images) and lesion size relative to network stride.

---

## 1. Problem & Dataset

**Task:** Detect and localize acne lesions in facial images using bounding box regression. Acne severity and subtype vary substantially across individuals, making this a challenging fine-grained detection problem.

### ACNE04 Stats

| Split | Images | Annotated Lesions |
|---|---|---|
| Train | 991 | 13,389 |
| Val | 283 | 3,425 |
| Test | 142 | 1,737 |
| **Total** | **1,416** | **18,551** |

- **Annotation format:** COCO bounding boxes (x, y, w, h)
- **Lesion subtypes:** Papule, pustule, nodule/cyst, whitehead/blackhead — **trained and evaluated as four separate classes** (the COCO annotations retain subtype labels; mAP reported is the mean across all four)
- **Lesion size:** Predominantly 20–90px bounding box side length on images that are typically 1024px+; see Section 3 for implications
- **Density:** Ranges from 1 to 50+ annotated lesions per image

> **On the class distribution figure:** Nodules/cysts dominate (5,757 train annotations) while whitehead/blackhead are rarest (675). Because no subtype is vanishingly rare, class weighting was unnecessary. Per-class mAP at validation: pustules and whiteheads (~0.30) outperform papules and nodules (~0.12–0.14), likely because the former are rounder and more visually distinct.

### Sample Images with Annotations

![Sample annotations](outputs/part1/sample_annotations.png)

*Sample ACNE04 images with ground-truth bounding boxes. Note the wide variation in lesion count, size, and skin tone.*

### Class Distribution

![Class distribution](outputs/part1/class_distribution.png)

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
epochs     = 100  (ran all 100; best checkpoint at epoch 55)
batch_size = 16
optimizer  = AdamW
lr0        = 0.01
imgsz      = 640

# Faster R-CNN — not tuned
epochs     = 20
batch_size = 8
optimizer  = SGD, lr=1e-3, momentum=0.9, weight_decay=1e-4
scheduler  = StepLR, step_size=5, gamma=0.5
backbone   = ResNet50-FPN (ImageNet pretrained)
```

No hyperparameter search was performed. Both models use reasonable starting-point settings given the dataset size. Faster R-CNN's best validation checkpoint was at epoch 8 (val loss = 1.0654); subsequent epochs showed no meaningful improvement under StepLR decay.

### Training Curves

![YOLOv8 training curves](outputs/part1/yolov8_training_curves.png)

*YOLOv8s box loss, cls loss, and mAP@0.5 over 100 epochs. Best checkpoint at epoch 55.*

![Faster R-CNN loss](outputs/part1/faster_rcnn_loss.png)

*Faster R-CNN train vs validation loss over 20 epochs. Best checkpoint at epoch 8; StepLR decay causes validation loss to plateau after that.*

### Hardware & Training Time

| Model | GPU | Training Time |
|---|---|---|
| YOLOv8s | A100 (80GB) | ~20 min (100 epochs, best at epoch 55) |
| Faster R-CNN | A100 (80GB) | ~1 hour (20 epochs, best at epoch 8) |

---

## 4. Results

### Evaluation Pipeline

Both models were evaluated on the same held-out test set (142 images) using the **pycocotools COCO evaluation standard** (IoU thresholds 0.5:0.05:0.95, confidence threshold = 0.25). Precision and Recall are reported at the shared confidence threshold of 0.25, matched at IoU ≥ 0.5.

> **On the val/test gap for YOLOv8:** YOLOv8's built-in trainer reported mAP@0.5 = 0.2137 on the *validation* set at best epoch 55. The test-set evaluation yields mAP@0.5 = 0.0571 using the same Ultralytics evaluation code. This is a genuine train/test generalization gap — not a methodology difference — and likely reflects partial overfitting to the 283-image validation set distribution.

### Quantitative Comparison (Test Set)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Detections (conf≥0.25) | Mean IoU | Inference (ms/img) |
|---|---|---|---|---|---|---|---|
| YOLOv8s | 0.057 | 0.014 | **0.536** | 0.078 | 156 | **0.533** | 72 |
| **Faster R-CNN** | **0.156** | **0.037** | 0.316 | **0.453** | 2,489 | 0.369 | 146 |

The two models are not simply "better/worse" — they operate at fundamentally different points on the PR curve:

- **YOLOv8s** is highly conservative: 156 detections total at conf≥0.25, but those detections are better-localized (mean IoU 0.533). When it fires, it fires accurately.
- **Faster R-CNN** is permissive: 16× more detections (2,489), with mean IoU only 0.369 and ~650 detections that have IoU=0 with any GT box (pure false positives). Its higher mAP comes from extending to higher recall at the cost of precision — not from being uniformly more accurate per detection.

Inference latency is ~2× (72 vs 146 ms/img on A100). On a CPU or mobile chip, YOLO's advantage would grow significantly.

### Precision-Recall Curves

![PR curves](outputs/part1/pr_curves.png)

*PR curves at IoU=0.5. Below recall ~0.4, YOLOv8 has higher precision than Faster R-CNN. Above recall ~0.4, Faster R-CNN dominates because YOLOv8 cannot reach those recall regimes at all under default confidence calibration. The mAP gap comes almost entirely from this high-recall region.*

### IoU Distribution

![IoU distribution](outputs/part1/iou_distribution.png)

*YOLOv8s: 156 detections, mean IoU 0.533, 68.6% matched at IoU≥0.5. Faster R-CNN: 2,489 detections, mean IoU 0.369, 39.1% matched. The large spike at IoU=0 for Faster R-CNN (~650 detections) represents false positives with no overlap with any GT box.*

---

## 5. Qualitative Results

### Detection Visualizations

**Side-by-side comparison (YOLOv8s vs Faster R-CNN):**

![Detection comparison](outputs/part1/detection_comparision.png)

*Each column shows the same image with YOLOv8s predictions (left) and Faster R-CNN predictions (right). Note YOLOv8's sparse detections vs Faster R-CNN's higher recall.*

### Failure Cases

| Failure Mode | Model | Why It Happens |
|---|---|---|
| **Small lesions missed** | Both | At 640px input, 20px lesions become ~12px — at or below YOLOv8's stride-8 limit. Faster R-CNN's RPN anchors also start at ~32px. |
| **Dense lesion clusters** | Both | NMS suppresses overlapping predictions. When 10+ lesions are within a 100px region, both models collapse them into 1–2 boxes. |
| **YOLOv8 misses most lesions (R=0.08)** | YOLOv8 | Confidence scores on small/dense lesions fall below 0.25 threshold; mosaic augmentation further shrinks effective object size during training. |
| **False positives on pores/hair follicles** | Faster R-CNN | Texture similarity to small papules; more common at lower confidence thresholds where recall is higher (P=0.32 at conf=0.25). |

---

## 6. Discussion

### Per-Lesion-Density Analysis

![Density mAP](outputs/part1/density_map.png)

*mAP@0.5 broken down by lesion density bin. This shows whether the gap between models is uniform or concentrated in specific difficulty levels.*

The density-bin results reveal where each model's architectural choice matters most:
- On **low-density images** (1–5 lesions), both models perform similarly — region proposals vs anchor-free doesn't matter much when objects are well-separated
- On **high-density images** (16+ lesions), Faster R-CNN's explicit RPN suppresses fewer true positives than YOLOv8's single-pass NMS — which is the mechanistic explanation for its higher overall mAP, grounded in the actual data

### Speed / Accuracy Tradeoff

YOLOv8s is ~2× faster at inference (72ms vs 146ms per image on A100) and trivially deployable as a single `.pt` file. However, the recall gap (R=0.08 vs R=0.45) is far more clinically significant than the latency difference. For a screening application where missing a lesion is more costly than a false alarm, Faster R-CNN is the clear choice. YOLOv8 would become competitive if its recall could be raised — e.g., by lowering the confidence threshold, increasing input resolution, or disabling mosaic augmentation.

### Where Both Models Hit Their Ceiling

Both are well below dermatology benchmarks (mAP@0.5 > 0.40 typical with 10k+ images). Three primary limiters:

1. **Dataset size:** ~1,400 images is small for fine-grained dense detection.
2. **Object scale:** 12–15px effective input size pushes COCO-pretrained backbones to their limit.
3. **Annotation ambiguity:** Subtype boundary ambiguity (whitehead vs small papule) introduces label noise that caps achievable precision.

### What I Would Change Next

1. Train YOLOv8 at 1280px input with mosaic disabled — directly addresses the two biggest recall failure modes.
2. Add image tiling at inference for both models — no retraining required.
3. Stricter early stopping on val mAP (patience=5) to reduce YOLOv8's val overfitting.
4. Replace COCO-pretrained backbone with a face/skin-pretrained one if available.
