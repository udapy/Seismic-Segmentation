# src/seismic_segmentation/data/seismic_dataset.py
"""Dataset classes for seismic segmentation."""

import json
from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class SeismicSegDataset(Dataset):
    """Dataset for seismic segmentation using preprocessed HDF5 data."""

    def __init__(self, config, split="train"):
        """Initialize the dataset with config."""
        self.config = config
        self.split = split

        # Get paths
        self.data_dir = Path(
            config.get("processed_data_dir", config.get("output_dir", "processed_data"))
        )
        self.h5_path = self.data_dir / "data.h5"

        # Get configuration options
        self.patch_size = config.get("patch_size", 256)
        self.use_patches = config.get("use_patches", True)

        # Load class info
        class_info_path = self.data_dir / "class_info.json"
        if class_info_path.exists():
            with open(class_info_path, "r") as f:
                self.class_info = json.load(f)
            self.n_classes = len(self.class_info)
        else:
            self.class_info = None
            self.n_classes = config.get("n_classes", 6)  # Default to 6 classes

        # Load indices for the requested split
        # Open the file temporarily to get the indices
        with h5py.File(self.h5_path, "r") as h5f:
            if split == "train":
                self.indices = h5f["train_indices"][:].copy()
            elif split == "val":
                self.indices = h5f["val_indices"][:].copy()
            else:  # test
                self.indices = h5f["test_indices"][:].copy()

        # Set transforms
        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Get trace index
        trace_idx = self.indices[idx]

        # Open the HDF5 file for this specific access
        with h5py.File(self.h5_path, "r") as h5f:
            # Get seismic and label data
            seismic = h5f["seismic"][trace_idx].astype(np.float32)
            label = h5f["labels"][trace_idx].astype(np.int64)  # Explicitly cast to int64 (Long)

        # Extract a random patch if needed
        if self.use_patches:
            seismic, label = self.extract_random_patch(seismic, label)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=seismic, mask=label)
            seismic = transformed["image"]
            label = transformed["mask"]

        # Ensure proper format (C, H, W) for PyTorch and correct data types
        if isinstance(seismic, np.ndarray):
            seismic = np.expand_dims(seismic, axis=0)  # Add channel dimension
            seismic = torch.from_numpy(seismic).float()
            label = torch.from_numpy(label).long()  # Explicitly convert to long
        elif isinstance(seismic, torch.Tensor) and isinstance(label, torch.Tensor):
            # If already tensors (from transform), ensure correct types
            seismic = seismic.float()
            label = label.long()  # Explicitly convert to long

        return {"image": seismic, "mask": label, "trace_idx": trace_idx}

    def extract_random_patch(self, seismic, label):
        """Extract a random patch from the seismic and label data."""
        # Check dimensions and reshape if necessary
        if len(seismic.shape) == 1:
            # For 1D data, we need a better approach than simple reshaping
            # Let's convert it to a proper 2D format based on our knowledge of seismic data

            # Determine the best rectangle shape close to original data length
            length = seismic.shape[0]
            width = int(np.sqrt(length))
            height = length // width

            # Reshape, padding if necessary
            padded_length = width * height
            if padded_length < length:
                # We need one more row to fit all data
                height += 1
                padded_length = width * height

            # Pad arrays if needed
            if padded_length > length:
                seismic_padded = np.zeros(padded_length, dtype=seismic.dtype)
                label_padded = np.zeros(padded_length, dtype=np.int64)  # Explicitly use int64
                seismic_padded[:length] = seismic
                label_padded[:length] = label
                seismic = seismic_padded
                label = label_padded

            # Reshape to 2D
            seismic = seismic.reshape(height, width)
            label = label.reshape(height, width)

        # Now proceed with patch extraction or resizing
        h, w = seismic.shape

        # Check if image is too small for the patch size
        if h < self.patch_size or w < self.patch_size:
            # Resize to match patch size
            # Use a proper interpolation method for seismic and nearest for labels
            from skimage.transform import resize

            seismic_resized = resize(
                seismic, (self.patch_size, self.patch_size), mode="reflect", anti_aliasing=True
            )

            # For labels (segmentation masks), use nearest neighbor to preserve class values
            label_resized = resize(
                label,
                (self.patch_size, self.patch_size),
                order=0,  # Nearest-neighbor
                mode="constant",
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.int64)  # Explicitly cast to int64

            return seismic_resized, label_resized

        # If image is big enough, extract a random patch
        max_h = h - self.patch_size
        max_w = w - self.patch_size

        # Random top-left corner
        h_start = np.random.randint(0, max_h + 1)
        w_start = np.random.randint(0, max_w + 1)

        # Extract patch
        seismic_patch = seismic[
            h_start : h_start + self.patch_size, w_start : w_start + self.patch_size
        ]
        label_patch = label[
            h_start : h_start + self.patch_size, w_start : w_start + self.patch_size
        ].astype(np.int64)  # Ensure int64

        return seismic_patch, label_patch

    def get_transforms(self):
        """Get appropriate transforms based on split."""
        augmentation = self.config.get("augmentation", True)

        if self.split == "train" and augmentation:
            return A.Compose(
                [
                    A.Resize(self.patch_size, self.patch_size)
                    if not self.use_patches
                    else A.NoOp(),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.GaussNoise(var_limit=(0.001, 0.01), p=0.5),
                    A.Normalize(mean=0.0, std=1.0),
                    ToTensorV2(),
                ],
                is_check_shapes=False,
            )  # Disable shape checking
        else:
            return A.Compose(
                [
                    A.Resize(self.patch_size, self.patch_size)
                    if not self.use_patches
                    else A.NoOp(),
                    A.Normalize(mean=0.0, std=1.0),
                    ToTensorV2(),
                ],
                is_check_shapes=False,
            )  # Disable shape checking

    def __del__(self):
        """Close HDF5 file when done."""
        if hasattr(self, "h5f"):
            self.h5f.close()


def get_dataloader(config, split="train"):
    """Get dataloader for the requested split."""
    dataset = SeismicSegDataset(config, split)

    shuffle = split == "train"
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )

    return dataloader
