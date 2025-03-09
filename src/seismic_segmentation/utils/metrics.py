# src/seismic_segmentation/utils/metrics.py
"""Evaluation metrics for seismic segmentation."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


def get_metrics(y_true, y_pred, num_classes):
    """
    Calculate various evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are numpy arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate metrics
    acc = accuracy_score(y_true_flat, y_pred_flat)

    # Handle binary vs. multiclass cases
    if num_classes == 1 or num_classes == 2:
        precision = precision_score(y_true_flat, y_pred_flat, average="binary", zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average="binary", zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, average="binary", zero_division=0)
        iou = jaccard_score(y_true_flat, y_pred_flat, average="binary", zero_division=0)
    else:
        precision = precision_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)
        iou = jaccard_score(y_true_flat, y_pred_flat, average="macro", zero_division=0)

    # Class-wise IoU (Jaccard Index)
    class_iou = {}
    for cls in range(num_classes):
        if cls in np.unique(y_true_flat):
            mask_true = y_true_flat == cls
            mask_pred = y_pred_flat == cls
            intersection = np.logical_and(mask_true, mask_pred).sum()
            union = np.logical_or(mask_true, mask_pred).sum()
            class_iou[f"iou_class_{cls}"] = intersection / union if union > 0 else 0

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_iou": iou,
        **class_iou,
    }

    return metrics


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient.

    Args:
        y_true: Ground truth
        y_pred: Predictions
        smooth: Smoothing term to avoid division by zero

    Returns:
        Dice coefficient
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    intersection = np.sum(y_true_flat * y_pred_flat)
    dice = (2.0 * intersection + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)

    return dice
