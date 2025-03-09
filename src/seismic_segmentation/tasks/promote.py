# src/seismic_segmentation/tasks/promote.py
"""Model promotion task for seismic segmentation."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ..utils.logger import get_logger


def run(config):
    """
    Run model promotion task.

    Args:
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    # Set up directories
    models_dir = Path(config.get("models_dir", "output"))
    output_dir = Path(config.get("output_dir", "models/production"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = get_logger(str(output_dir))
    logger.info("Starting model promotion")

    try:
        # Find all models
        model_dirs = []
        for item in models_dir.glob("*"):
            if item.is_dir() and (item / "checkpoints" / "best_model.pth").exists():
                model_dirs.append(item)

        if not model_dirs:
            logger.error(f"No models found in {models_dir}")
            return False

        logger.info(f"Found {len(model_dirs)} model(s)")

        # Evaluate all models
        model_metrics = []
        for model_dir in model_dirs:
            checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
            config_path = model_dir / "config.yaml"

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weight_only=True) # nosec

            # Load config
            if config_path.exists():
                with open(config_path, "r") as f:
                    model_config = yaml.safe_load(f)
            else:
                model_config = checkpoint.get("config", {})

            # Get metrics
            metrics = {}
            if "loss" in checkpoint:
                metrics["loss"] = checkpoint["loss"]

            # Check if evaluation results exist
            eval_results_path = model_dir / "evaluations" / "results.json"
            if eval_results_path.exists():
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)
                metrics.update(eval_results.get("overall_metrics", {}))

            model_metrics.append(
                {
                    "dir": str(model_dir),
                    "name": model_dir.name,
                    "checkpoint": str(checkpoint_path),
                    "config": model_config,
                    "metrics": metrics,
                }
            )

        # Sort models by primary metric
        primary_metric = config.get("primary_metric", "mean_iou")
        threshold = config.get("threshold", {}).get(primary_metric, 0.0)

        # Filter models by threshold
        qualified_models = [
            model
            for model in model_metrics
            if model["metrics"].get(primary_metric, 0.0) >= threshold
        ]

        if not qualified_models:
            logger.error(f"No models meet the threshold ({threshold}) for {primary_metric}")
            return False

        # Sort by primary metric (descending)
        qualified_models.sort(key=lambda x: x["metrics"].get(primary_metric, 0.0), reverse=True)

        # Get best model
        best_model = qualified_models[0]
        logger.info(
            f"Selected best model: {best_model['name']} with {primary_metric}={best_model['metrics'].get(primary_metric, 0.0):.4f}"
        )

        # Create version
        if config.get("version", "auto") == "auto":
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            version = config.get("version")

        # Create version directory
        version_dir = output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        shutil.copy2(best_model["checkpoint"], version_dir / "model.pth")

        # Save model info
        model_info = {
            "version": version,
            "source": best_model["dir"],
            "name": best_model["name"],
            "metrics": best_model["metrics"],
            "config": best_model["config"],
            "promoted_at": datetime.now().isoformat(),
        }

        with open(version_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        # Create latest symlink
        latest_link = output_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)

        # Create relative symlink
        os.symlink(version, latest_link, target_is_directory=True)

        logger.info(f"Model promoted successfully to {version_dir}")
        logger.info(f"Created 'latest' symlink to {version}")

        return True

    except Exception as e:
        logger.error(f"Error during model promotion: {e}")
        if config.get("debug", False):
            import traceback

            logger.error(traceback.format_exc())
        return False
