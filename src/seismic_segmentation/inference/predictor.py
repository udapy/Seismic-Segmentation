# src/seismic_segmentation/inference/predictor.py
"""Prediction class for seismic segmentation models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ..utils.logger import get_logger


class Predictor:
    """Predictor class for seismic segmentation models."""

    def __init__(self, model, dataloader, device, config, output_dir=None):
        """
        Initialize predictor.

        Args:
            model: Model to use for prediction
            dataloader: Data loader
            device: Device to use
            config: Configuration dictionary
            output_dir: Directory for prediction outputs
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.config = config

        # Set up output directory
        if output_dir is None:
            output_dir = Path(config.get("output_dir", "predictions"))

        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.vis_dir = self.output_dir / "visualizations"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)

        # Set up logger
        self.logger = get_logger(str(self.output_dir))

        # Prediction parameters
        self.save_raw = config.get("save_raw", True)
        self.save_vis = config.get("save_vis", True)
        self.n_classes = config.get("n_classes", 6)

        # Load class info if available
        class_info_path = (
            Path(config.get("processed_data_dir", config.get("data_dir", "processed_data")))
            / "class_info.json"
        )
        self.class_info = None
        if class_info_path.exists():
            import json

            with open(class_info_path, "r") as f:
                self.class_info = json.load(f)

    def predict(self):
        """Run prediction on the dataloader."""
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Predicting")):
                # Get data
                images = batch["image"].to(self.device)
                trace_indices = batch["trace_idx"]

                # Forward pass
                if hasattr(self.model, "use_sag") and self.model.use_sag:
                    # For SAG model, generate dummy prompts
                    B = images.size(0)
                    point_prompts = torch.rand(B, 1, 3, device=self.device)
                    box_prompts = torch.rand(B, 1, 4, device=self.device)
                    well_prompts = torch.rand(B, 1, 10, device=self.device)

                    outputs = self.model(
                        images,
                        point_prompts=point_prompts,
                        box_prompts=box_prompts,
                        well_prompts=well_prompts,
                    )
                else:
                    outputs = self.model(images)

                # Get predictions
                if self.n_classes > 1:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()

                # Save predictions
                for i, trace_idx in enumerate(trace_indices):
                    trace_idx = trace_idx.item()
                    pred = preds[i]
                    image = images[i].cpu().numpy()

                    # Save raw prediction
                    if self.save_raw:
                        np.save(str(self.raw_dir / f"trace_{trace_idx}.npy"), pred)

                    # Save visualization
                    if self.save_vis:
                        self._save_visualization(image, pred, trace_idx)

        self.logger.info(f"Prediction complete! Results saved to {self.output_dir}")

    def _save_visualization(self, image, prediction, trace_idx):
        """
        Save visualization of a prediction.

        Args:
            image: Input image
            prediction: Predicted mask
            trace_idx: Trace index
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original image
        if image.shape[0] == 1:
            image = image[0]  # Remove channel
        ax1.imshow(image, cmap="gray", aspect="auto")
        ax1.set_title("Seismic Image")
        ax1.axis("off")

        # Plot prediction
        if self.n_classes > 1 and self.class_info:
            # Create custom colormap based on class_info
            pred_rgb = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

            for cls_idx, cls_data in self.class_info.items():
                mask = prediction == int(cls_idx)
                color = cls_data["color"]

                for c in range(3):
                    pred_rgb[:, :, c][mask] = color[c]

            ax2.imshow(pred_rgb, aspect="auto")
        else:
            # Use default colormap
            ax2.imshow(prediction, cmap="tab10", aspect="auto", vmin=0, vmax=self.n_classes - 1)

        ax2.set_title("Predicted Segmentation")
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(str(self.vis_dir / f"trace_{trace_idx}.png"), dpi=150)
        plt.close()
