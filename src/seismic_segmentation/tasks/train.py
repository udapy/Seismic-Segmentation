# src/seismic_segmentation/tasks/train.py
"""Training task for seismic segmentation."""

from pathlib import Path

import torch
import yaml

from ..data.dataset import get_dataloader
from ..models import get_model
from ..training.trainer import Trainer
from ..utils.logger import get_logger
from ..utils.loss import get_loss_fn


def run(config):
    """
    Run training task.

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

    # Set up experiment directory
    experiment_name = config.get(
        "experiment_name", f"{config.get('model_type', 'unet')}_experiment"
    )
    output_dir = Path(config.get("output_dir", "output"))
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = get_logger(str(experiment_dir))
    logger.info(f"Starting training experiment: {experiment_name}")
    logger.info(f"Using device: {device}")

    # Save config
    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    try:
        # Create data loaders
        train_loader = get_dataloader(config, split="train")
        val_loader = get_dataloader(config, split="val")

        logger.info(
            f"Created data loaders: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} validation samples"
        )

        # Create model
        model = get_model(config)
        model = model.to(device)

        logger.info(f"Created model: {config.get('model_type', 'unet')}")

        # Create loss function
        criterion = get_loss_fn(config)
        logger.info(f"Using loss function: {config.get('loss_type', 'ce')}")

        # Create optimizer
        lr = config.get("lr", 0.001)
        weight_decay = config.get("weight_decay", 0.0001)

        if config.get("optimizer", "adamw").lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        logger.info(
            f"Using optimizer: {config.get('optimizer', 'adamw')} with lr={lr}, weight_decay={weight_decay}"
        )

        # Create scheduler
        scheduler_type = config.get("scheduler", "cosine")
        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
        elif scheduler_type == "cosine":
            epochs = config.get("epochs", 50)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            scheduler = None

        logger.info(f"Using scheduler: {scheduler_type}")

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            config=config,
            experiment_dir=experiment_dir,
        )

        # Run training
        trainer.train()
        logger.info("Training completed successfully")

        return True

    except Exception as e:
        logger.error(f"Error during training: {e}")
        if config.get("debug", False):
            import traceback

            logger.error(traceback.format_exc())
        return False
