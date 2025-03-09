# src/seismic_segmentation/tasks/preprocess.py
"""Preprocessing task for seismic segmentation."""

import json
from pathlib import Path

import h5py
import numpy as np

from ..data.preprocessing import preprocess_segy_data, preprocess_with_source_splits, save_examples
from ..utils.logger import get_logger


def run(config):
    """
    Run preprocessing task.

    Args:
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    # Set random seed
    seed = config.get("seed", 42)
    np.random.seed(seed)

    # Get paths
    data_dir = Path(config.get("data_dir", "data/NZ_PM_seismic_dataset"))
    output_dir = Path(config.get("output_dir", "processed_data"))

    # Set up logger
    logger = get_logger(str(output_dir))
    logger.info(f"Starting preprocessing with data from {data_dir}")

    try:
        # Define file paths
        seismic_file = data_dir / "parihaka_seismic.sgy"
        labels_file = data_dir / "parihaka_labels.sgy"
        class_info_file = data_dir / "interpretation" / "class_info.json"

        # Check if SEG-Y files exist
        if not seismic_file.exists():
            logger.error(f"Seismic file not found: {seismic_file}")
            return False
        if not labels_file.exists():
            logger.error(f"Labels file not found: {labels_file}")
            return False

        # Check if class info file exists
        if not class_info_file.exists():
            logger.warning(f"Class info file not found: {class_info_file}")
            class_info_file = None

        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train").mkdir(exist_ok=True)
        (output_dir / "val").mkdir(exist_ok=True)
        (output_dir / "test").mkdir(exist_ok=True)
        (output_dir / "train_examples").mkdir(exist_ok=True)
        (output_dir / "val_examples").mkdir(exist_ok=True)
        (output_dir / "test_examples").mkdir(exist_ok=True)

        # Check if source data has training/validation subdirectories
        source_train_dir = data_dir / "interpretation" / "train"
        source_val_dir = data_dir / "interpretation" / "validation"

        use_source_splits = source_train_dir.exists() and source_val_dir.exists()

        if use_source_splits:
            logger.info("Found existing train/validation splits in source data. Using those...")
            # Process data using existing splits
            h5_path = preprocess_with_source_splits(
                seismic_file=seismic_file,
                labels_file=labels_file,
                train_dir=source_train_dir,
                val_dir=source_val_dir,
                output_dir=output_dir,
                class_info_file=class_info_file,
                remove_first_crossline=config.get("remove_first_crossline", True),
                rescale_seismic=config.get("rescale_seismic", True),
                rescale_range=config.get("rescale_range", (-1.0, 1.0)),
            )
        else:
            # Get split ratios for random splitting
            train_ratio = config.get("train_ratio", 0.8)
            val_ratio = config.get("val_ratio", 0.1)
            test_ratio = 1.0 - train_ratio - val_ratio
            split_ratio = (train_ratio, val_ratio, test_ratio)

            # Preprocess data with random splits
            h5_path = preprocess_segy_data(
                seismic_file=seismic_file,
                labels_file=labels_file,
                output_dir=output_dir,
                class_info_file=class_info_file,
                split_ratio=split_ratio,
                remove_first_crossline=config.get("remove_first_crossline", True),
                rescale_seismic=config.get("rescale_seismic", True),
                rescale_range=config.get("rescale_range", (-1.0, 1.0)),
            )

        # Generate example visualizations
        with h5py.File(h5_path, "r") as h5f:
            examples_per_split = config.get("examples_per_split", 5)

            logger.info("Generating example visualizations...")
            # Load class info
            if class_info_file:
                with open(class_info_file, "r") as f:
                    class_info = json.load(f)
            else:
                class_info = None

            # Generate examples for each split
            save_examples(
                h5f,
                h5f["train_indices"][:examples_per_split],
                output_dir / "train_examples",
                class_info,
            )
            save_examples(
                h5f,
                h5f["val_indices"][:examples_per_split],
                output_dir / "val_examples",
                class_info,
            )
            save_examples(
                h5f,
                h5f["test_indices"][:examples_per_split],
                output_dir / "test_examples",
                class_info,
            )

        logger.info(f"Preprocessing complete. Data saved to {h5_path}")
        return True

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        if config.get("debug", False):
            import traceback

            logger.error(traceback.format_exc())
        return False
