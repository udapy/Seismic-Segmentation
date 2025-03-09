# src/seismic_segmentation/evaluation/evaluator.py
"""Evaluation class for seismic segmentation models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.metrics import get_metrics
from ..utils.visualization import visualize_prediction


class Evaluator:
    """Evaluator class for seismic segmentation models."""

    def __init__(self, model, dataloader, device, config, output_dir=None):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            dataloader: Data loader
            device: Device to use
            config: Configuration dictionary
            output_dir: Directory for evaluation outputs
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.config = config

        # Set up output directory
        if output_dir is None:
            experiment_name = config.get(
                "experiment_name", f"{config.get('model_type', 'unet')}_evaluation"
            )
            output_base_dir = config.get("output_dir", "evaluations")
            output_dir = Path(output_base_dir) / experiment_name

        self.output_dir = Path(output_dir)
        self.vis_dir = self.output_dir / "visualizations"
        self.pred_dir = self.output_dir / "predictions"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)
        self.pred_dir.mkdir(exist_ok=True)

        # Set up logger
        self.logger = get_logger(str(self.output_dir))

        # Evaluation parameters
        self.visualize_all = config.get("visualize_all", False)
        self.visualize_n = config.get("visualize_n", 10)
        self.save_predictions = config.get("save_predictions", True)
        self.n_classes = config.get("n_classes", 6)

    def evaluate(self):
        """
        Evaluate the model.

        Returns:
            Evaluation metrics and sample-level results
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        sample_results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                # Get data
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
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
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                # Store predictions and targets for metrics calculation
                all_predictions.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

                # Calculate metrics for each sample in batch
                for i in range(images.size(0)):
                    sample_pred = preds[i].cpu().numpy()
                    sample_mask = masks[i].cpu().numpy()
                    sample_metrics = get_metrics(sample_mask, sample_pred, self.n_classes)

                    sample_results.append(
                        {
                            "trace_idx": trace_indices[i].item(),
                            "accuracy": sample_metrics["accuracy"],
                            "precision": sample_metrics["precision"],
                            "recall": sample_metrics["recall"],
                            "f1_score": sample_metrics["f1_score"],
                            "mean_iou": sample_metrics["mean_iou"],
                        }
                    )

                # Visualize predictions
                if self.visualize_all or (batch_idx == 0 and self.visualize_n > 0):
                    n_to_vis = min(
                        images.size(0),
                        self.visualize_n if not self.visualize_all else images.size(0),
                    )

                    for i in range(n_to_vis):
                        img = images[i].cpu().numpy()
                        mask = masks[i].cpu().numpy()
                        pred = preds[i].cpu().numpy()

                        # Create visualization
                        visualize_prediction(
                            img,
                            mask,
                            pred,
                            self.n_classes,
                            save_path=str(self.vis_dir / f"trace_{trace_indices[i].item()}.png"),
                        )

                # Save predictions if required
                if self.save_predictions:
                    for i in range(images.size(0)):
                        pred = preds[i].cpu().numpy()
                        np.save(str(self.pred_dir / f"trace_{trace_indices[i].item()}.npy"), pred)

        # Concatenate all predictions and targets
        all_predictions_flat = np.concatenate([p.flatten() for p in all_predictions])
        all_targets_flat = np.concatenate([t.flatten() for t in all_targets])

        # Calculate overall metrics
        overall_metrics = get_metrics(all_targets_flat, all_predictions_flat, self.n_classes)

        # Save results
        results = {"overall_metrics": overall_metrics, "sample_results": sample_results}

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save sample results to CSV
        sample_df = pd.DataFrame(sample_results)
        sample_df.to_csv(self.output_dir / "sample_results.csv", index=False)

        self.logger.info(f"Evaluation complete! Results saved to {self.output_dir}")

        return overall_metrics, sample_results
