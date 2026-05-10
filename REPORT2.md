# Part 2: Cross-Domain Acne Classification — Report

> **Repo:** https://github.com/EvxLee/AcneDetection
> **Author:** Evan Lee · **Date:** May 2026

---

## TL;DR

Trained EfficientNet-B0 on balanced ACNE04 patches (val acc 98.7%). On DermNet (4,002 images, 7.8% acne), the unadapted model achieved AUROC = 0.475, worse than random (predictions inverted). Full domain adaptation (Reinhard color normalization + 2-stage few-shot fine-tuning on 20 DermNet samples + threshold optimization + 6-crop TTA) brought AUROC to 0.662. FaceNet VGGFace2 with only a fine-tuned head reached AUROC = 0.568, recovering most of the gap with far simpler engineering than the full EfficientNet pipeline. F1 scores remain low (~0.12–0.16) throughout due to 92.2% class imbalance; AUROC is the only reliable metric here.

---

## 1. Problem Framing

**Why this is hard:** ACNE04 and DermNet are two fundamentally different datasets despite sharing the label "acne."

- **ACNE04** = tight 224×224 crops of individual lesions from smartphone selfies. Consistent warm lighting, face-only content, uniform background.
- **DermNet** = full clinical photos spanning all body parts (hands, ears, torso, face), varied lighting, diverse skin conditions, and frequently watermarked. The acne folder contains rosacea alongside acne vulgaris.

A model trained on ACNE04 patches learns "warm + close-up + skin texture = acne." In DermNet, that pattern is more common in non-acne clinical photos (psoriasis patches, eczema on body skin) than in DermNet's own acne images. The result: the baseline model is systematically wrong in the wrong direction.

### Domain Gap — Visual

![Domain gap](outputs/part2/domain_gap.png)

*Top row: ACNE04 training patches: tight skin crops, consistent color, face only. Bottom row: DermNet acne images: full clinical photos, varied body parts, watermarks, different lighting. The model had never seen anything like the bottom row during training.*

### Quantified Domain Gap — RGB Channel Statistics

![RGB channel stats](outputs/part2/rgb_channel_stats.png)

| Dataset | R mean | G mean | B mean |
|---|---|---|---|
| ACNE04 | 0.661 | 0.479 | 0.404 |
| DermNet | 0.568 | 0.418 | 0.389 |
| Difference | −0.094 | −0.061 | −0.015 |

DermNet is notably darker and less red. The R-channel gap (−0.094) is the largest single numeric indicator of the domain shift and directly motivates Reinhard color normalization.

---

## 2. Patch Extraction (ACNE04)

**Positive patches (acne):** Crop each bounding box annotation from the ACNE04 source image, resize to 224×224. Boxes smaller than 20px on either side are skipped (too few pixels to contain meaningful lesion signal): 2,304 skipped in train, ~482 in val.

**Negative patches (no-acne):** For each lesion in an image, randomly sample a same-image region of 90×90px with zero overlap with any GT box, then resize to 224×224. Candidates are rejected if they are too dark (mean brightness < 40/255) or near-greyscale (inter-channel spread < 15/255), catching hair, dark backgrounds, and watermarks.

**Result:** Near-perfectly balanced dataset; no class weighting needed.

| Split | Acne patches | No-acne patches | Skipped (tiny) |
|---|---|---|---|
| Train | 11,085 | 11,082 | 2,304 |
| Val | 2,943 | 2,904 | ~482 |

Negatives are sampled from the **same image** as the positives; this forces the model to distinguish lesion texture from the surrounding clear skin under identical lighting and color conditions.

### Sample Patches

![Sample patches](outputs/part2/sample_patches.png)

*Top row: acne patches (positive). Bottom row: no-acne patches (negative). Both come from the same source images; the classifier must learn lesion-specific texture, not image-level statistics.*

---

## 3. Baseline Model

**Architecture:** EfficientNet-B0 (ImageNet pretrained) with a 2-class head: `Dropout(0.3) → Linear(1280, 2)`.

**Why EfficientNet-B0:** Best accuracy/efficiency tradeoff at small model size (~5.3M params). Small enough to fine-tune on 22k patches without severe overfitting; compound scaling (depth × width × resolution) gives better feature resolution for 224×224 skin texture patches than ResNet-18/34.

**Training:** Adam (lr=1e-4), CosineAnnealingLR, 20 epochs, batch 64. Augmentation: RandomResizedCrop (scale 0.7–1.0), ColorJitter, GaussianBlur, RandomHorizontalFlip, all targeted at making features robust to the color/zoom differences between ACNE04 and DermNet.

### Training Curves

![Training curves](outputs/part2/classifier_training_curves.png)

*Train loss drops from 0.194 → 0.023 over 20 epochs. Val loss plateaus at ~0.042 from epoch 10 onward. Best checkpoint at epoch 12: val accuracy = **0.9868**.*

> **98.7% should be read as "memorized this distribution," not "learned acne detection in general."** The patches are near-duplicate crops from the same 991 images; near-perfect accuracy here sets up the domain gap that follows.

### The Gap That Motivates Everything

| Metric | ACNE04 Val | DermNet Test |
|---|---|---|
| Accuracy | **0.9868** | 0.1432 |
| F1 (acne) | — | 0.1460 |
| AUROC | — | 0.4749 |

The model achieves 98.7% on the source domain and **fails worse than random** on the target. AUROC < 0.5 means the model assigns higher acne probability to non-acne DermNet images than to actual acne images; it learned the wrong signal entirely. Every technique in Section 4 is a response to this gap.

> **On the naive baseline:** A model predicting non-acne for every image scores 92.2% accuracy. Accuracy is useless here. AUROC is the only metric that isn't gamed by class imbalance or threshold choice.

---

## 4. Domain Adaptation Experiments

### Color Normalization

**Technique:** Reinhard (2001) LAB-space normalization + per-channel histogram matching. Match DermNet image statistics to ACNE04 reference statistics before inference.

![Color normalization](outputs/part2/color_normalization.png)

*Left → right: Original DermNet image (cooler, darker) → ACNE04 reference patch → after Reinhard normalization → after histogram matching. The shift toward ACNE04's warmer tones is clearly visible.*

### Training Augmentation

Applied during ACNE04 patch training to make the feature extractor robust to DermNet's visual conditions before it ever sees DermNet.

![Augmentation examples](outputs/part2/augmentation_examples.png)

*Same ACNE04 patch, 5 random augmentation draws. ColorJitter shifts color temperature; RandomResizedCrop zooms in/out; GaussianBlur simulates defocus. Each augmentation targets a specific known difference between ACNE04 and DermNet.*

### Full Ablation Table

| Experiment | Technique | Accuracy | F1 (acne) | AUROC |
|---|---|---|---|---|
| Naive baseline | Predict all non-acne | 0.9220 | 0.0000 | — |
| No adaptation | EfficientNet-B0 direct | 0.1432 | 0.1460 | 0.4749 |
| Color norm only | Reinhard + histo-match | 0.4498 | 0.1241 | 0.4794 |
| **Full pipeline** | Color norm + 2-stage FT + thresh (0.05) + TTA | 0.1917 | 0.1542 | **0.6625** |
| FaceNet VGGFace2 | Face-pretrained backbone + color norm + thresh (0.54) | 0.5092 | 0.1571 | 0.5683 |

**Reading the table:** Low accuracy on the full pipeline (19.2%) is not a failure; it's the intended behavior of threshold=0.05, which aggressively calls acne to maximize recall on a 7.8%-prevalence positive class. AUROC is threshold-independent and is the true measure of model quality.

**Full pipeline detail (best AUROC):**
- Stage 1: Freeze backbone, fine-tune head only for 50 epochs on 10 acne + 10 non-acne DermNet samples (lr=5e-3)
- Stage 2: Unfreeze last EfficientNet feature block at 100× lower LR (5e-6 vs 5e-5) for 30 epochs
- Threshold swept 0.05–0.95 on the same 20 samples; 0.05 maximizes F1
- 6-crop TTA: FiveCrop(224) + center horizontal flip, average 6 forward passes

**FaceNet insight:** Frozen VGGFace2 backbone + head fine-tuned on 20 samples = AUROC 0.568, no TTA. Face-specific pretraining provides more transferable skin features than ImageNet alone, with far less engineering.

---

## 5. Final Model Results

![Final model results](outputs/part2/final_model_results.png)

*Left column: baseline (no adaptation). Right column: full pipeline (color norm + 2-stage FT + TTA + threshold=0.05). Top: confusion matrices. Bottom: ROC curves.*

**Confusion matrix interpretation:**

| | Baseline | Full Pipeline |
|---|---|---|
| True Positives (acne caught) | 293 / 312 (94%) | 295 / 312 (95%) |
| Predicted acne for | 3,703 / 4,002 (93% of all images) | 3,513 / 4,002 (88% of all images) |
| False Positives | 3,410 non-acne wrongly flagged | 3,218 non-acne wrongly flagged |
| AUROC | 0.4749 (below random) | **0.6625** |

The 94% recall in the baseline is not a sign of competence: the model predicts acne for 93% of all images regardless of true label. The full pipeline's 1-sample improvement (293 → 295 TP) is noise; the meaningful gain is AUROC 0.475 → 0.663, visible in the ROC curves below the diagonal vs above it.

---

## 6. Grad-CAM Visualizations

![Grad-CAM](outputs/part2/gradcam_dermnet.png)

*10 DermNet test images through the fine-tuned EfficientNet-B0. Row 1: 5 acne samples. Row 2: 5 non-acne samples. Green title = correct prediction, red = wrong. The non-acne row shows the model assigning high acne probability to most non-acne images regardless of threshold, exposing where the domain gap actually bites.*

**What the heatmaps show:**

- **Correct acne predictions (green):** Attention concentrates on central skin texture: papular surfaces, pustule topography. The model is looking at the right place.
- **False positives on non-acne (red):** Heatmaps drift to image borders, backgrounds, and watermarks. The model is triggering on domain artifacts rather than lesion features, a direct Grad-CAM diagnosis of where the domain gap bites.

---

## 7. Reflection

### What Worked

**Few-shot fine-tuning was the biggest lever.** The 18-point AUROC jump (0.479 → 0.662) came almost entirely from 20 labeled DermNet samples and 2-stage fine-tuning, not from preprocessing. Color normalization barely moved AUROC (+0.004), while fine-tuning alone contributed the rest. This confirms that feature-level adaptation requires at least some target-domain supervision, even a tiny amount.

**FaceNet surprised.** Frozen VGGFace2 + linear head + color norm = AUROC 0.568, simpler than the full EfficientNet pipeline. For applications with more labeled target data, scaling this approach would likely outperform EfficientNet fine-tuning.

### What Didn't Work

**Color normalization alone is not enough.** AUROC barely moved from 0.475 → 0.479 despite visually obvious color transfer. The inversion is a feature-level problem, not a color-statistics problem; matching pixel distributions cannot fix what the backbone learned to attend to.

**Accuracy is the wrong metric throughout.** Every experiment that "improves" accuracy either exploits class imbalance (predict all non-acne = 92.2%) or threshold choice. AUROC is the only honest number.

### Honest Assessment of Remaining Gap

Best AUROC achieved: 0.663. Clinical-grade screening typically requires AUROC ≥ 0.85. The remaining gap has three main sources:

1. **Dataset mismatch is structural.** ACNE04 patches ≠ DermNet clinical photos in format, not just statistics. No amount of color normalization fixes a model that has never seen a watermarked clinical photo.
2. **20 samples is too few.** Fine-tuning on 20 samples partially recalibrates the head but can't update the backbone's feature representations safely.
3. **DermNet label noise.** DermNet's acne folder includes rosacea, mislabeled images, and non-face photos. The "acne" signal the model is trying to learn is noisy at the source.

### What I'd Try With More Time

1. **More labeled DermNet samples:** even 100–200 would likely push AUROC past 0.80
2. **ISIC/DermNet pretrained backbone:** start from a backbone already trained on clinical dermatology images rather than ImageNet or VGGFace2
3. **CycleGAN style transfer:** convert DermNet images to ACNE04-patch style before inference, eliminating the need for fine-tuning entirely
4. **Tile-based inference:** split DermNet clinical photos into patches matching ACNE04's crop format, then aggregate predictions; this directly closes the format mismatch without any training
