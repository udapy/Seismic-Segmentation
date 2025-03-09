# src/seismic_segmentation/utils/loss.py
"""Loss functions for seismic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""

    def __init__(self, smooth=1.0):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing term to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calculate Dice loss.

        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)

        Returns:
            Dice loss
        """
        num_classes = logits.shape[1]

        if num_classes == 1:
            # Binary case
            probs = torch.sigmoid(logits)
            targets = targets.float().unsqueeze(1)
        else:
            # Multi-class case
            probs = F.softmax(logits, dim=1)
            targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Flatten spatial dimensions
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets.sum(dim=2)

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Return loss (1 - Dice)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""

    def __init__(self, alpha=0.5, gamma=2.0):
        """
        Initialize Focal loss.

        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Calculate Focal loss.

        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)

        Returns:
            Focal loss
        """
        num_classes = logits.shape[1]

        if num_classes == 1:
            # Binary case
            bce = F.binary_cross_entropy_with_logits(
                logits, targets.float().unsqueeze(1), reduction="none"
            )
            probs = torch.sigmoid(logits)
            p_t = probs * targets.float().unsqueeze(1) + (1 - probs) * (
                1 - targets.float().unsqueeze(1)
            )
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * bce

            if self.alpha is not None:
                alpha_t = self.alpha * targets.float().unsqueeze(1) + (1 - self.alpha) * (
                    1 - targets.float().unsqueeze(1)
                )
                loss = alpha_t * loss

            return loss.mean()
        else:
            # Multi-class case
            ce = F.cross_entropy(logits, targets, reduction="none")
            logpt = -ce
            pt = torch.exp(logpt)
            loss = (1 - pt) ** self.gamma * ce

            if self.alpha is not None:
                alpha = torch.tensor([self.alpha] * num_classes, device=logits.device)
                alpha_t = alpha.gather(0, targets.view(-1))
                loss = alpha_t * loss

            return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss (Cross-entropy + Dice)."""

    def __init__(self, alpha=0.5):
        """
        Initialize Combined loss.

        Args:
            alpha: Weight for Cross-entropy loss (1-alpha for Dice loss)
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Calculate Combined loss.

        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)

        Returns:
            Combined loss
        """
        num_classes = logits.shape[1]

        if num_classes == 1:
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float().unsqueeze(1))
        else:
            ce_loss = self.ce_loss(logits, targets)

        dice_loss = self.dice_loss(logits, targets)

        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


def get_loss_fn(config):
    """
    Get loss function based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Loss function
    """
    loss_type = config.get("loss_type", "ce")
    n_classes = config.get("n_classes", 6)

    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "focal":
        alpha = config.get("focal_alpha", 0.5)
        gamma = config.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "combined":
        alpha = config.get("combined_alpha", 0.5)
        return CombinedLoss(alpha=alpha)
    else:  # default: cross entropy
        if n_classes > 1:
            return nn.CrossEntropyLoss()
        else:
            return nn.BCEWithLogitsLoss()
