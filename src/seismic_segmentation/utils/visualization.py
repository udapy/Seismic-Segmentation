# src/seismic_segmentation/utils/visualization.py
"""Visualization utilities for seismic segmentation."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def apply_mask_colormap(mask, num_classes):
    """
    Apply colormap to segmentation mask.

    Args:
        mask: Segmentation mask (H, W)
        num_classes: Number of classes in the mask

    Returns:
        Colored mask (H, W, 3)
    """
    # Define colormap for visualization
    if num_classes <= 4:
        colors = ["black", "red", "green", "blue", "yellow"]
        cmap = ListedColormap(colors[:num_classes])
    else:
        cmap = plt.cm.get_cmap("tab10", num_classes)

    # Apply colormap
    colored_mask = cmap(mask)

    # Remove alpha channel if present
    if colored_mask.shape[-1] == 4:
        colored_mask = colored_mask[..., :3]

    return colored_mask


def visualize_prediction(image, true_mask, pred_mask, num_classes, save_path=None):
    """
    Visualize seismic image with ground truth and prediction masks.

    Args:
        image: Input image (C, H, W) or (H, W)
        true_mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        num_classes: Number of classes
        save_path: Path to save the visualization (optional)
    """
    # Ensure image is 2D
    if len(image.shape) == 3:
        image = image[0]  # Take first channel

    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    im0 = axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Seismic Image")
    axes[0].axis("off")

    # Add colorbar to the first image
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)

    # Plot ground truth mask
    im1 = axes[1].imshow(true_mask, cmap="tab10", vmin=0, vmax=num_classes - 1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Add colorbar to the ground truth mask
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax, ticks=np.arange(num_classes))
    cbar.set_label("Class")

    # Plot predicted mask
    im2 = axes[2].imshow(pred_mask, cmap="tab10", vmin=0, vmax=num_classes - 1)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Add colorbar to the prediction mask
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax, ticks=np.arange(num_classes))
    cbar.set_label("Class")

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_batch(
    batch_images, batch_masks, batch_preds, num_classes, save_path=None, max_samples=4
):
    """
    Visualize a batch of seismic images with their masks and predictions.

    Args:
        batch_images: Batch of images (B, C, H, W) or (B, H, W)
        batch_masks: Batch of ground truth masks (B, H, W)
        batch_preds: Batch of predicted masks (B, H, W)
        num_classes: Number of classes
        save_path: Path to save the visualization (optional)
        max_samples: Maximum number of samples to visualize
    """
    # Limit the number of samples to visualize
    n_samples = min(batch_images.shape[0], max_samples)

    # Create a figure with n_samples rows and 3 columns
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    # If only one sample, wrap axes in a list
    if n_samples == 1:
        axes = np.array([axes])

    for i in range(n_samples):
        # Get sample image, mask, and prediction
        image = batch_images[i]
        mask = batch_masks[i]
        pred = batch_preds[i]

        # Ensure image is 2D
        if len(image.shape) == 3:
            image = image[0]  # Take first channel

        # Plot image
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"Sample {i + 1} - Image")
        axes[i, 0].axis("off")

        # Plot ground truth mask
        axes[i, 1].imshow(mask, cmap="tab10", vmin=0, vmax=num_classes - 1)
        axes[i, 1].set_title(f"Sample {i + 1} - Ground Truth")
        axes[i, 1].axis("off")

        # Plot predicted mask
        axes[i, 2].imshow(pred, cmap="tab10", vmin=0, vmax=num_classes - 1)
        axes[i, 2].set_title(f"Sample {i + 1} - Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
