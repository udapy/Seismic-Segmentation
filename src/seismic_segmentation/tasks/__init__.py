# src/seismic_segmentation/tasks/__init__.py
"""Task modules for seismic segmentation pipeline."""

from .evaluate import run as run_evaluate
from .infer import run as run_infer
from .preprocess import run as run_preprocess
from .promote import run as run_promote
from .train import run as run_train
from .tune import run as run_tune

__all__ = ["run_preprocess", "run_train", "run_evaluate", "run_infer", "run_tune", "run_promote"]
