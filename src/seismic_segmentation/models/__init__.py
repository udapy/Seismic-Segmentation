# src/seismic_segmentation/models/__init__.py
"""Model architectures for seismic segmentation."""

from .sag_model import SAGModel
from .unet import UNetResNet34


def get_model(config):
    """
    Factory function to get the specified model.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized model
    """
    model_type = config.get("model_type", "unet").lower()

    if model_type == "unet":
        return UNetResNet34(config)
    elif model_type == "sag":
        return SAGModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["UNetResNet34", "SAGModel", "get_model"]
