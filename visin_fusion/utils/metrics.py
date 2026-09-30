#!/usr/bin/env python3
"""
Model evaluation metrics Python scripts

Created on June 18th, 2021
"""

import numpy as np
import torch


def find_overlap_exclude_bg_ignore(n_classes, output, anno):
    """
    Correct IoU calculation that excludes background class (0)
    WITHOUT modifying predictions based on ground truth.

    :param n_classes: Total number of classes (including background)
    :param output: Model output batch (B, C, H, W)
    :param anno: Ground truth batch (B, H, W)
    :return: area_overlap, area_pred, area_label, area_union
    """
    # Argmax over channels → predicted class per pixel
    _, pred_indices = torch.max(output, dim=1)

    # Compute overlap (TP): prediction matches annotation
    overlap = pred_indices * (pred_indices == anno).long()

    # Number of foreground classes (1..n_classes-1)
    num_eval_classes = n_classes - 1

    # Histogram for classes 1..n_classes-1
    area_overlap = torch.histc(overlap.float(), bins=num_eval_classes, min=0.5, max=n_classes - 0.5)

    area_pred = torch.histc(pred_indices.float(), bins=num_eval_classes, min=0.5, max=n_classes - 0.5)

    area_label = torch.histc(anno.float(), bins=num_eval_classes, min=0.5, max=n_classes - 0.5)

    # Union = TP + FP + FN
    area_union = area_pred + area_label - area_overlap
    area_union = torch.clamp(area_union, min=1e-6)

    return area_overlap, area_pred, area_label, area_union


# Average precision, shared by every model's test script so all models are scored the same way.
# AP is computed over every stored pixel (no sampling), so it is deterministic.


def store_predictions_for_ap(output_seg, anno, all_predictions, all_targets, eval_classes, eval_indices):
    """Store pixel-wise class probabilities and targets for AP, on the CPU.

    Per class, only pixels predicted as that class or labelled as it are kept, so both
    true and false positives are represented.
    """
    probs = torch.softmax(output_seg, dim=1)  # [batch, classes, H, W]
    preds = torch.argmax(output_seg, dim=1)  # [batch, H, W]

    for cls_name, train_idx in zip(eval_classes, eval_indices):
        cls_probs = probs[:, train_idx].flatten()
        cls_targets = (anno == train_idx).flatten()
        relevant = (preds == train_idx).flatten() | cls_targets
        if relevant.any():
            all_predictions[cls_name].append(cls_probs[relevant].float().cpu())
            all_targets[cls_name].append(cls_targets[relevant].float().cpu())


def compute_ap_for_class(cls_name, all_predictions, all_targets):
    """Average precision of one class over every stored pixel (VOC 2010 method)."""
    if not all_predictions.get(cls_name):
        return 0.0

    pred_probs = torch.cat(all_predictions[cls_name])
    pred_targets = torch.cat(all_targets[cls_name])
    num_positives = pred_targets.sum().item()
    if num_positives == 0:
        return 0.0

    # Sort by confidence, highest first
    pred_targets = pred_targets[torch.argsort(pred_probs, descending=True)]
    tp = torch.cumsum(pred_targets, dim=0)
    fp = torch.cumsum(1 - pred_targets, dim=0)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / num_positives
    return voc_ap(recall, precision)


def voc_ap(recall, precision):
    """Area under the precision-recall curve, VOC 2010 method."""
    if len(recall) == 0:
        return 0.0

    recall = recall.cpu().numpy()
    precision = precision.cpu().numpy()

    # Sentinel values, then make precision monotonically decreasing
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    # Sum over the points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
