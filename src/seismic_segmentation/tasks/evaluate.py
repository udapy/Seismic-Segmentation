# src/seismic_segmentation/tasks/evaluate.py
"""Evaluation task for seismic segmentation."""

from pathlib import Path

import torch

from ..data.dataset import get_dataloader
from ..evaluation.evaluator import Evaluator
from ..models import get_model
from ..utils.logger import get_logger


def run(config):
    """
    Run evaluation task.

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
    output_dir = Path(config.get("output_dir", "evaluations"))
    experiment_name = config.get(
        "experiment_name", f"{config.get('model_type', 'unet')}_evaluation"
    )
    eval_dir = output_dir / experiment_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = get_logger(str(eval_dir))
    logger.info(f"Starting evaluation: {experiment_name}")
    logger.info(f"Using device: {device}")

    try:
        # Get checkpoint path
        checkpoint_path = config.get("checkpoint", None)
        if checkpoint_path is None:
            logger.error("No checkpoint path provided")
            return False

        # Get split to evaluate on
        split = config.get("split", "test")
        logger.info(f"Evaluating on {split} split")

        # Create data loader
        dataloader = get_dataloader(config, split=split)
        logger.info(f"Created data loader with {len(dataloader.dataset)} samples")

        # Create and load model
        model = get_model(config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, while_only=True)  # nosec
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

        # Create evaluator
        evaluator = Evaluator(
            model=model, dataloader=dataloader, device=device, config=config, output_dir=eval_dir
        )

        # Run evaluation
        overall_metrics, _ = evaluator.evaluate()

        # Log results
        logger.info("Evaluation results:")
        for metric_name, metric_value in overall_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        logger.info(f"Evaluation completed successfully. Results saved to {eval_dir}")

        return True

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if config.get("debug", False):
            import traceback

            logger.error(traceback.format_exc())
        return False
