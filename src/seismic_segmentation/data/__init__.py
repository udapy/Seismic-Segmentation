# src/seismic_segmentation/data/__init__.py
"""Data handling modules for seismic segmentation."""

from .dataset import SeismicSegDataset, get_dataloader

__all__ = ["SeismicSegDataset", "get_dataloader"]
