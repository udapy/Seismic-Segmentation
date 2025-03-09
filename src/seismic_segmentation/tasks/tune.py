# src/seismic_segmentation/tasks/tune.py
"""Hyperparameter tuning task for seismic segmentation."""

from pathlib import Path

import optuna
import torch
import yaml
from optuna.trial import Trial

from ..data.dataset import get_dataloader
from ..models import get_model
from ..training.trainer import Trainer
from ..utils.logger import get_logger
from ..utils.loss import get_loss_fn


def run(config):
    """
    Run hyperparameter tuning task.

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

    # Set up tuning directory
    output_dir = Path(config.get("output_dir", "tuning"))
    experiment_name = config.get("experiment_name", f"{config.get('model_type', 'unet')}_tuning")
    tuning_dir = output_dir / experiment_name
    tuning_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    logger = get_logger(str(tuning_dir))
    logger.info(f"Starting hyperparameter tuning: {experiment_name}")
    logger.info(f"Using device: {device}")

    # Save config
    with open(tuning_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Define objective function
    def objective(trial: Trial):
        # Set up trial directory
        trial_dir = tuning_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Get hyperparameters
        trial_config = config.copy()

        # Override with trial-specific hyperparameters
        params = config.get("params", {})
        for param_name, param_config in params.items():
            param_type = param_config.get("type", "float")

            if param_type == "float":
                low = param_config.get("low", 1e-5)
                high = param_config.get("high", 1.0)
                log = param_config.get("log", False)
                value = trial.suggest_float(param_name, low, high, log=log)
            elif param_type == "int":
                low = param_config.get("low", 1)
                high = param_config.get("high", 100)
                value = trial.suggest_int(param_name, low, high)
            elif param_type == "categorical":
                choices = param_config.get("choices", [])
                value = trial.suggest_categorical(param_name, choices)

            trial_config[param_name] = value

        # Create data loaders
        train_loader = get_dataloader(trial_config, split="train")
        val_loader = get_dataloader(trial_config, split="val")

        # Create model
        model = get_model(trial_config)
        model = model.to(device)

        # Create loss function
        criterion = get_loss_fn(trial_config)

        # Create optimizer
        lr = trial_config.get("lr", 0.001)
        weight_decay = trial_config.get("weight_decay", 0.0001)

        if trial_config.get("optimizer", "adamw").lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Create scheduler
        scheduler_type = trial_config.get("scheduler", "cosine")
        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
        elif scheduler_type == "cosine":
            epochs = trial_config.get("epochs", 10)  # Use fewer epochs for tuning
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            scheduler = None

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            config=trial_config,
            experiment_dir=trial_dir,
        )

        # Run training
        trainer.train()

        # Get best validation metric
        metric_name = config.get("metric", "val_mean_iou")
        if metric_name == "val_loss":
            metric_value = min(trainer.val_losses) if trainer.val_losses else float("inf")
        elif metric_name == "val_mean_iou":
            metric_value = (
                max(trainer.val_metrics_history["mean_iou"])
                if trainer.val_metrics_history["mean_iou"]
                else 0.0
            )
        elif metric_name == "val_accuracy":
            metric_value = (
                max(trainer.val_metrics_history["accuracy"])
                if trainer.val_metrics_history["accuracy"]
                else 0.0
            )
        elif metric_name == "val_f1_score":
            metric_value = (
                max(trainer.val_metrics_history["f1_score"])
                if trainer.val_metrics_history["f1_score"]
                else 0.0
            )

        return metric_value

    try:
        # Create study
        direction = config.get("direction", "maximize")
        study_name = experiment_name
        storage_name = f"sqlite:///{tuning_dir}/study.db"

        study = optuna.create_study(
            study_name=study_name, storage=storage_name, direction=direction, load_if_exists=True
        )

        # Run optimization
        n_trials = config.get("n_trials", 20)
        n_jobs = config.get("n_jobs", 1)

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        # Log results
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best {config.get('metric', 'val_mean_iou')}: {best_value:.4f}")
        logger.info("Best parameters:")
        for param_name, param_value in best_params.items():
            logger.info(f"  {param_name}: {param_value}")

        # Save best parameters
        with open(tuning_dir / "best_params.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)

        logger.info(f"Hyperparameter tuning completed successfully. Results saved to {tuning_dir}")

        return True

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        if config.get("debug", False):
            import traceback

            logger.error(traceback.format_exc())
        return False
