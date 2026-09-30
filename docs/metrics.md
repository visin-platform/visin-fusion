# Semantic Segmentation Metrics Explanation

This document explains the metrics used to evaluate semantic segmentation models during training and validation.

## Overview

Semantic segmentation is a computer vision task that assigns a class label to every pixel in an image. The metrics below evaluate how well the model performs this task for different object classes (vehicle, sign, human).

## Per-Class Metrics

### IoU (Intersection over Union)
- **Definition**: Measures the overlap between predicted and ground truth segmentation masks
- **Formula**: IoU = (Predicted ∩ Ground Truth) / (Predicted ∪ Ground Truth)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 1.0 = Perfect overlap
  - 0.0 = No overlap
  - IoU is the primary metric for segmentation tasks

### Precision
- **Definition**: Of all pixels predicted as a class, what fraction are actually that class
- **Formula**: Precision = True Positives / (True Positives + False Positives)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Measures how accurate the model's positive predictions are

### Recall
- **Definition**: Of all pixels that are actually a class, what fraction did the model correctly identify
- **Formula**: Recall = True Positives / (True Positives + False Negatives)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Measures how many actual positives the model found

### F1 Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Balances precision and recall, useful when you need both

## Overall Metrics

### Loss
- **Definition**: Training loss value (typically cross-entropy loss for segmentation)
- **Range**: 0.0 to ∞ (lower is better)
- **Interpretation**: Measures how well the model fits the training data

### Mean IoU
- **Definition**: Average IoU across all classes
- **Formula**: Mean IoU = (IoU_vehicle + IoU_sign + IoU_human) / 3
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Overall segmentation performance across classes

### Pixel Accuracy
- **Definition**: Percentage of pixels correctly classified
- **Formula**: Pixel Accuracy = Correct Pixels / Total Pixels
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Simple accuracy metric, but can be misleading for imbalanced classes

### Mean Accuracy
- **Definition**: Average accuracy across all classes
- **Formula**: Mean Accuracy = (Accuracy_vehicle + Accuracy_sign + Accuracy_human) / 3
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Class-wise accuracy average

### Dice Score
- **Definition**: Similar to F1 score, measures overlap between prediction and ground truth
- **Formula**: Dice = 2 × (Predicted ∩ Ground Truth) / (|Predicted| + |Ground Truth|)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Alternative to IoU, often used in medical imaging

### FWIoU (Frequency Weighted IoU)
- **Definition**: Frequency-weighted version of IoU that accounts for class imbalance by weighting each class's IoU by its pixel frequency
- **Formula**: FWIoU = Σ(class_frequency × IoU_class) / Σ(class_frequency)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Gives more importance to common classes, providing a balanced metric for imbalanced datasets

## Train vs Validation

### Training Metrics ("train")
- **Purpose**: Monitor how well the model fits the training data
- **Expected behavior**: Should improve as training progresses
- **Common issues**: If training metrics are poor, the model isn't learning

### Validation Metrics ("val")
- **Purpose**: Monitor how well the model generalizes to unseen data
- **Expected behavior**: Should improve initially, may plateau or decrease (overfitting)
- **Common issues**: Large gap between train/val indicates overfitting

## Example Analysis

Looking at the provided metrics:

**Training Performance:**
- Vehicle: High IoU (0.65) with excellent recall (0.99) but moderate precision (0.66)
- Sign: Low IoU (0.23) with high recall (0.98) but low precision (0.23)
- Human: Moderate IoU (0.32) with high recall (0.98) but low precision (0.32)

**Validation Performance:**
- Similar patterns but lower scores, indicating some overfitting
- Vehicle maintains good performance (IoU 0.60)
- Sign and human show significant drop in precision on validation

**Key Insights:**
- Model is good at finding objects (high recall) but has false positives (low precision)
- Sign detection is particularly challenging
- Validation performance suggests the model may benefit from regularization or data augmentation

## Best Practices

1. **Monitor both train and val**: Watch for overfitting (train improves, val worsens)
2. **Focus on IoU**: Primary metric for segmentation evaluation
3. **Check per-class performance**: Some classes may need more training data
4. **Balance precision/recall**: Depending on your use case requirements
5. **Use validation for early stopping**: Prevent overfitting by stopping when val metrics peak

## Average precision (AP)

Computed per class in `visin_fusion/utils/metrics.py`, the same way for every model:

- For each test image, every pixel that the model **predicts as the class or that is labelled as the
  class** is kept, with the model's softmax probability for the class and whether the label matches.
  Pixels that are neither are left out, so this is AP over the pixels relevant to the class, not
  over every pixel of the image.
- All kept pixels of a test set are ranked by probability (no sampling, so the result is the same on
  every run), and precision and recall are computed at every rank.
- AP is the area under that precision–recall curve with precision made monotonically decreasing
  (the PASCAL VOC 2010 method). `mean_ap` averages it over the evaluated classes.

Before September 2026 the CLFTv2, MaskFormer, Mask2Former and DeepLabV3+ test scripts ranked a
random sample of 100,000 of the kept pixels instead, so their AP moved by up to about 0.02 between
runs of the same checkpoint. CLFT always used every pixel.
