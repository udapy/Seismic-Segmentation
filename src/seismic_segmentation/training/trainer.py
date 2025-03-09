# src/seismic_segmentation/training/trainer.py
"""Training class for seismic segmentation models."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.metrics import get_metrics
from ..utils.visualization import visualize_batch


class Trainer:
    """Trainer class for seismic segmentation models."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        config,
        experiment_dir=None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            criterion: Loss function
            device: Device to use
            config: Configuration dictionary
            experiment_dir: Directory for experiment outputs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config

        # Set up experiment directory
        if experiment_dir is None:
            experiment_name = config.get(
                "experiment_name", f"{config.get('model_type', 'unet')}_experiment"
            )
            output_dir = config.get("output_dir", "output")
            experiment_dir = Path(output_dir) / experiment_name

        self.experiment_dir = Path(experiment_dir)
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        self.vis_dir = self.experiment_dir / "visualizations"

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.tensorboard_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)

        # Set up logger
        self.logger = get_logger(str(self.experiment_dir))

        # Set up TensorBoard
        self.writer = SummaryWriter(str(self.tensorboard_dir))

        # Training parameters
        self.epochs = config.get("epochs", 50)
        self.log_interval = config.get("log_interval", 10)
        self.save_interval = config.get("save_interval", 5)
        self.visualize_interval = config.get("visualize_interval", 20)

        # Best validation tracking
        self.best_val_loss = float("inf")

        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = {"mean_iou": [], "accuracy": [], "f1_score": []}

    def train(self):
        """Train model for specified number of epochs."""
        self.logger.info(f"Starting training with {self.epochs} epochs")

        for epoch in range(self.epochs):
            # Train for one epoch
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate(epoch)

            # Update scheduler
            if hasattr(self.scheduler, "step"):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log metrics
            self.logger.info(
                f"Epoch: {epoch + 1}/{self.epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Mean IoU: {val_metrics['mean_iou']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f}"
            )

            # Update tracking
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics_history["mean_iou"].append(val_metrics["mean_iou"])
            self.val_metrics_history["accuracy"].append(val_metrics["accuracy"])
            self.val_metrics_history["f1_score"].append(val_metrics["f1_score"])

            # Save model if it's the best so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"Saved best model with validation loss: {val_loss:.4f}")

            # Save checkpoint at regular intervals
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)

            # Write to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

        self.writer.close()
        self.logger.info("Training complete!")

    def train_one_epoch(self, epoch):
        """Train for one epoch."""

        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Debug prints for the first few batches
            if batch_idx < 3:
                self.logger.info(
                    f"Batch {batch_idx} - Images shape: {images.shape}, Masks shape: {masks.shape}"
                )

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            try:
                # Forward
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

                    # Debug print for model output
                if batch_idx < 3:
                    self.logger.info(f"Model output shape: {outputs.shape}")
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                if hasattr(images, "shape") and hasattr(masks, "shape"):
                    self.logger.error(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
                if "outputs" in locals() and hasattr(outputs, "shape"):
                    self.logger.error(f"Outputs shape: {outputs.shape}")
                raise

            masks = masks.long()  # Ensure masks are long type for loss calculation

            # Calculate loss
            loss = self.criterion(outputs, masks)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # Update running loss
            running_loss += loss.item() * images.size(0)

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log batch loss to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Loss/train_step", loss.item(), global_step)

            # Visualize predictions
            if (batch_idx + 1) % self.visualize_interval == 0:
                self.visualize_predictions(images, masks, outputs, epoch, batch_idx, split="train")

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_masks = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Get data
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Forward
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

                # Calculate loss
                loss = self.criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                # Get predictions
                if self.config.get("n_classes", 1) > 1:
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                # Store predictions and masks for metrics calculation
                all_preds.append(preds.cpu().numpy())
                all_masks.append(masks.cpu().numpy())

                # Visualize first batch
                if batch_idx == 0:
                    self.visualize_predictions(
                        images, masks, outputs, epoch, batch_idx, split="val"
                    )

        # Calculate epoch loss
        epoch_loss = val_loss / len(self.val_loader.dataset)

        # Calculate metrics
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_masks = np.concatenate([m.flatten() for m in all_masks])
        metrics = get_metrics(all_masks, all_preds, self.config.get("n_classes", 1))

        return epoch_loss, metrics

    def visualize_predictions(self, images, masks, outputs, epoch, batch_idx, split="train"):
        """Visualize model predictions."""
        if self.config.get("n_classes", 1) > 1:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = (torch.sigmoid(outputs) > 0.5).float()

        # Generate visualization
        visualize_batch(
            images.cpu(),
            masks.cpu(),
            preds.cpu(),
            self.config.get("n_classes", 1),
            save_path=str(self.vis_dir / f"{split}_epoch{epoch + 1}_batch{batch_idx + 1}.png"),
            max_samples=min(4, images.size(0)),
        )

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if hasattr(self.scheduler, "state_dict")
            else None,
            "loss": self.val_losses[-1] if self.val_losses else float("inf"),
            "config": self.config,
        }

        if is_best:
            checkpoint_path = self.checkpoints_dir / "best_model.pth"
        else:
            checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch{epoch + 1}.pth"

        torch.save(checkpoint, checkpoint_path)
