# src/seismic_segmentation/tasks/infer.py
"""Inference task for seismic segmentation."""

from pathlib import Path

import torch

from ..data.dataset import get_dataloader
from ..inference.predictor import Predictor
from ..models import get_model
from ..utils.logger import get_logger


def run(config):
    """
    Run inference task.

    Args:
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set device
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    # Set up output directory
    output_dir = Path(config.get("output_dir", "predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = get_logger(str(output_dir))
    logger.info("Starting inference")
    logger.info(f"Using device: {device}")

    try:
        # Get checkpoint path
        checkpoint_path = config.get("model_path", config.get("checkpoint", None))
        if checkpoint_path is None:
            logger.error("No checkpoint path provided")
            return False

        # Get split to run inference on
        split = config.get("split", "test")
        logger.info(f"Running inference on {split} split")

        # Create data loader
        dataloader = get_dataloader(config, split=split)
        logger.info(f"Created data loader with {len(dataloader.dataset)} samples")

        # Create and load model
        model = get_model(config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weight_only=True)  # nosec
        if "model_state_dict" not in checkpoint:
            logger.error("Checkpoint does not contain model state dict")
            return False
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

        # Create predictor
        predictor = Predictor(
            model=model, dataloader=dataloader, device=device, config=config, output_dir=output_dir
        )

        # Run prediction
        predictor.predict()

        logger.info(f"Inference completed successfully. Results saved to {output_dir}")

        return True

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        if config.get("debug", False):
            import traceback

            logger.error(traceback.format_exc())
        return False
