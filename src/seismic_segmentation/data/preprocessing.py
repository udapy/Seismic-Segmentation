# src/seismic_segmentation/data/preprocessing.py
"""Preprocessing utilities for seismic data."""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
from tqdm import tqdm


def preprocess_segy_data(
    seismic_file,
    labels_file,
    output_dir,
    class_info_file=None,
    split_ratio=(0.8, 0.1, 0.1),
    remove_first_crossline=True,
    rescale_seismic=True,
    rescale_range=(-1.0, 1.0),
):
    """
    Preprocess SEG-Y seismic data and labels for machine learning.

    Args:
        seismic_file: Path to the seismic SEG-Y file
        labels_file: Path to the labels SEG-Y file
        output_dir: Directory to save processed data
        class_info_file: Path to class info JSON file
        split_ratio: Train/val/test split ratio (default: 0.8/0.1/0.1)
        remove_first_crossline: Whether to remove the first crossline (default: True)
        rescale_seismic: Whether to rescale seismic data (default: True)
        rescale_range: Range to rescale to if rescale_seismic is True (default: (-1.0, 1.0))

    Returns:
        Path to the generated HDF5 file
    """
    print(f"Processing seismic data from {seismic_file}")
    print(f"Processing labels from {labels_file}")

    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class info if provided
    if class_info_file:
        with open(class_info_file, "r") as f:
            class_info = json.load(f)
        print(f"Loaded class info with {len(class_info)} classes")

        # Save class info to output directory
        with open(output_dir / "class_info.json", "w") as f:
            json.dump(class_info, f, indent=2)
    else:
        class_info = None

    # Open seismic and labels files
    with segyio.open(seismic_file, "r") as seismic, segyio.open(labels_file, "r") as labels:
        # Get dimensions
        n_traces = seismic.tracecount
        n_samples = len(seismic.samples)

        print(f"Seismic dimensions: {n_traces} traces x {n_samples} samples")

        # Check if dimensions match
        if n_traces != labels.tracecount:
            raise ValueError("Seismic and label files have different trace counts")

        # Determine start trace (skip first crossline if configured)
        start_trace = 1 if remove_first_crossline else 0

        # Create proper 2D shapes for the data
        # We'll reshape each trace into a square-ish 2D array
        width = int(np.sqrt(n_samples))
        height = n_samples // width
        if width * height < n_samples:
            height += 1
        padded_length = width * height

        print(f"Reshaping 1D traces of length {n_samples} to 2D shape ({height}, {width})")

        # Create HDF5 file for faster access during training
        h5_path = output_dir / "data.h5"
        with h5py.File(h5_path, "w") as h5f:
            # Store both 1D and 2D versions
            h5f.create_dataset(
                "seismic_1d", shape=(n_traces - start_trace, n_samples), dtype="float32"
            )
            h5f.create_dataset(
                "labels_1d", shape=(n_traces - start_trace, n_samples), dtype="int64"
            )

            # Store 2D versions (reshaped)
            h5f.create_dataset(
                "seismic", shape=(n_traces - start_trace, height, width), dtype="float32"
            )
            h5f.create_dataset(
                "labels", shape=(n_traces - start_trace, height, width), dtype="int64"
            )

            # Process traces
            for i in tqdm(range(start_trace, n_traces), desc="Processing traces"):
                # Adjust index for the skipped trace
                adj_idx = i - start_trace

                # Read seismic trace
                seismic_trace = seismic.trace[i].astype(np.float32)

                # Rescale seismic data if configured
                if rescale_seismic:
                    min_val, max_val = rescale_range
                    seismic_min, seismic_max = seismic_trace.min(), seismic_trace.max()
                    if seismic_max > seismic_min:  # Avoid division by zero
                        seismic_trace = (seismic_trace - seismic_min) / (seismic_max - seismic_min)
                        seismic_trace = seismic_trace * (max_val - min_val) + min_val

                # Read label trace
                label_trace = labels.trace[i].astype(np.int64)

                # Store 1D versions
                h5f["seismic_1d"][adj_idx] = seismic_trace
                h5f["labels_1d"][adj_idx] = label_trace

                # Create 2D versions (pad if necessary)
                seismic_2d = np.zeros((height, width), dtype=np.float32)
                label_2d = np.zeros((height, width), dtype=np.int64)

                # Fill in as much as possible
                flat_size = min(n_samples, padded_length)
                seismic_2d.flat[:flat_size] = seismic_trace[:flat_size]
                label_2d.flat[:flat_size] = label_trace[:flat_size]

                # Store 2D versions
                h5f["seismic"][adj_idx] = seismic_2d
                h5f["labels"][adj_idx] = label_2d

            # Create split indices
            n_traces_adjusted = n_traces - start_trace
            indices = np.arange(n_traces_adjusted)
            np.random.shuffle(indices)

            train_size = int(n_traces_adjusted * split_ratio[0])
            val_size = int(n_traces_adjusted * split_ratio[1])

            train_indices = indices[:train_size]
            val_indices = indices[train_size : train_size + val_size]
            test_indices = indices[train_size + val_size :]

            # Store split indices
            h5f.create_dataset("train_indices", data=train_indices)
            h5f.create_dataset("val_indices", data=val_indices)
            h5f.create_dataset("test_indices", data=test_indices)

            print(
                f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
            )

            # Store metadata
            h5f.attrs["n_traces"] = n_traces_adjusted
            h5f.attrs["n_samples"] = n_samples
            h5f.attrs["rescaled"] = rescale_seismic
            h5f.attrs["rescale_range"] = str(rescale_range) if rescale_seismic else "None"
            h5f.attrs["first_crossline_removed"] = remove_first_crossline
            h5f.attrs["original_shape"] = f"1D ({n_samples},)"
            h5f.attrs["reshaped_dimensions"] = f"2D ({height}, {width})"
            h5f.attrs["data_formats"] = "both 1D and 2D versions stored"

    print(f"Preprocessing complete. Data saved to {h5_path}")
    return h5_path


def __getitem__(self, idx):
    """Get a sample from the dataset."""
    # Get trace index
    trace_idx = self.indices[idx]

    # Open the HDF5 file for this specific access
    with h5py.File(self.h5_path, "r") as h5f:
        # Get seismic and label traces
        seismic = h5f["seismic"][trace_idx].astype(np.float32)
        label = h5f["labels"][trace_idx].astype(np.int64)  # Change to int64 (Long)

    # Extract a random patch if needed
    if self.use_patches:
        seismic, label = self.extract_random_patch(seismic, label)

    # Apply transformations
    if self.transform:
        transformed = self.transform(image=seismic, mask=label)
        seismic = transformed["image"]
        label = transformed["mask"]

    # Ensure proper format (C, H, W) for PyTorch
    if isinstance(seismic, np.ndarray):
        seismic = np.expand_dims(seismic, axis=0)  # Add channel dimension
        seismic = torch.from_numpy(seismic).float()
        label = torch.from_numpy(label).long()

    return {"image": seismic, "mask": label, "trace_idx": trace_idx}


def extract_random_patch(self, seismic, label):
    """Extract a random patch from the seismic and label data."""
    # Check dimensions and reshape if necessary
    if len(seismic.shape) == 1:
        # Convert 1D to 2D - simple approach to avoid recursion
        length = seismic.shape[0]
        width = int(np.sqrt(length))
        height = length // width

        # Ensure height*width is sufficient
        if height * width < length:
            height += 1

        # Create new arrays and fill them
        seismic_2d = np.zeros((height, width), dtype=seismic.dtype)
        label_2d = np.zeros((height, width), dtype=label.dtype)

        # Fill only up to the original length
        # This uses flat indexing which is simpler
        idx = min(length, height * width)
        seismic_2d.flat[:idx] = seismic[:idx]
        label_2d.flat[:idx] = label[:idx]

        seismic = seismic_2d
        label = label_2d

    # Get dimensions after potential reshaping
    h, w = seismic.shape

    # Handle case where image is smaller than patch size
    if h < self.patch_size or w < self.patch_size:
        # Simple resize to avoid recursion
        from skimage.transform import resize

        seismic_resized = resize(
            seismic, (self.patch_size, self.patch_size), mode="reflect", anti_aliasing=True
        )

        label_resized = resize(
            label,
            (self.patch_size, self.patch_size),
            order=0,  # Nearest-neighbor
            mode="constant",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.int64)

        return seismic_resized, label_resized

    # If image is big enough, extract a random patch
    max_h = h - self.patch_size
    max_w = w - self.patch_size

    # Avoid potential issues with random values
    h_start = 0 if max_h <= 0 else np.random.randint(0, max_h + 1)
    w_start = 0 if max_w <= 0 else np.random.randint(0, max_w + 1)

    # Extract patch
    seismic_patch = seismic[
        h_start : h_start + self.patch_size, w_start : w_start + self.patch_size
    ]
    label_patch = label[h_start : h_start + self.patch_size, w_start : w_start + self.patch_size]

    return seismic_patch, label_patch


def preprocess_with_source_splits(
    seismic_file,
    labels_file,
    train_dir,
    val_dir,
    output_dir,
    class_info_file=None,
    remove_first_crossline=True,
    rescale_seismic=True,
    rescale_range=(-1.0, 1.0),
):
    """
    Preprocess SEG-Y data using existing train/validation splits from source data.

    Args:
        seismic_file: Path to the seismic SEG-Y file
        labels_file: Path to the labels SEG-Y file
        train_dir: Path to source training directory
        val_dir: Path to source validation directory
        output_dir: Directory to save processed data
        class_info_file: Path to class info JSON file
        remove_first_crossline: Whether to remove the first crossline
        rescale_seismic: Whether to rescale seismic data
        rescale_range: Range to rescale to if rescale_seismic is True

    Returns:
        Path to the generated HDF5 file
    """
    print(f"Processing seismic data from {seismic_file}")
    print(f"Processing labels from {labels_file}")
    print(f"Using train/val splits from: {train_dir} and {val_dir}")

    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class info if provided
    class_info = None
    if class_info_file:
        with open(class_info_file, "r") as f:
            class_info = json.load(f)
        print(f"Loaded class info with {len(class_info)} classes")

        # Save class info to output directory
        with open(output_dir / "class_info.json", "w") as f:
            json.dump(class_info, f, indent=2)

    # Find train and validation indices
    # We'll need to map source data indices to our processed data indices
    train_images = []
    val_images = []

    # Identify training images
    for img_path in train_dir.glob("images/*.png"):
        train_images.append(img_path.stem)

    # Identify validation images
    for img_path in val_dir.glob("images/*.png"):
        val_images.append(img_path.stem)

    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")

    # Open seismic and labels files
    with segyio.open(seismic_file, "r") as seismic, segyio.open(labels_file, "r") as labels:
        # Get dimensions
        n_traces = seismic.tracecount
        n_samples = len(seismic.samples)

        print(f"Seismic dimensions: {n_traces} traces x {n_samples} samples")

        # Check if dimensions match
        if n_traces != labels.tracecount:
            raise ValueError("Seismic and label files have different trace counts")

        # Determine start trace (skip first crossline if configured)
        start_trace = 1 if remove_first_crossline else 0

        # Create HDF5 file for faster access during training
        h5_path = output_dir / "data.h5"
        with h5py.File(h5_path, "w") as h5f:
            # Create datasets
            h5f.create_dataset(
                "seismic", shape=(n_traces - start_trace, n_samples), dtype="float32"
            )
            h5f.create_dataset("labels", shape=(n_traces - start_trace, n_samples), dtype="uint8")

            # Process traces
            for i in tqdm(range(start_trace, n_traces), desc="Processing traces"):
                # Adjust index for the skipped trace
                adj_idx = i - start_trace

                # Read seismic trace
                seismic_trace = seismic.trace[i].astype(np.float32)

                # Rescale seismic data if configured
                if rescale_seismic:
                    min_val, max_val = rescale_range
                    seismic_min, seismic_max = seismic_trace.min(), seismic_trace.max()
                    seismic_trace = (seismic_trace - seismic_min) / (seismic_max - seismic_min)
                    seismic_trace = seismic_trace * (max_val - min_val) + min_val

                # Read label trace
                label_trace = labels.trace[i].astype(np.uint8)

                # Store in HDF5
                h5f["seismic"][adj_idx] = seismic_trace
                h5f["labels"][adj_idx] = label_trace

            # Map source data indices to our processed indices
            # For now, we'll create a simple mapping based on trace names
            # In a real-world scenario, you'd need to use proper trace identifiers

            # Let's assume trace names are numbers 0 to n_traces-1
            trace_names = [str(i) for i in range(start_trace, n_traces)]

            # Identify which traces belong to which split
            train_indices = []
            val_indices = []

            for i, trace_name in enumerate(trace_names):
                if trace_name in train_images:
                    train_indices.append(i)
                elif trace_name in val_images:
                    val_indices.append(i)

            # If we couldn't map all traces, create test indices from remaining traces
            all_indices = set(range(n_traces - start_trace))
            mapped_indices = set(train_indices + val_indices)
            test_indices = list(all_indices - mapped_indices)

            # If we couldn't find any matches (different naming convention), fall back to random splitting
            if not train_indices or not val_indices:
                print(
                    "Warning: Could not match source train/val images to traces. Using random split..."
                )
                # Create a random split
                all_indices = list(range(n_traces - start_trace))
                np.random.shuffle(all_indices)

                train_size = int(0.8 * len(all_indices))
                val_size = int(0.1 * len(all_indices))

                train_indices = all_indices[:train_size]
                val_indices = all_indices[train_size : train_size + val_size]
                test_indices = all_indices[train_size + val_size :]

            # Store split indices
            h5f.create_dataset("train_indices", data=train_indices)
            h5f.create_dataset("val_indices", data=val_indices)
            h5f.create_dataset("test_indices", data=test_indices)

            print(
                f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
            )

    return h5_path


def save_examples(h5f, indices, output_dir, class_info=None, n_examples=5):
    """
    Save example images for visualization.

    Args:
        h5f: HDF5 file handle
        indices: Indices to visualize
        output_dir: Directory to save examples
        class_info: Class info dictionary (optional)
        n_examples: Number of examples to save (default: 5)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit number of examples
    if len(indices) > n_examples:
        indices = indices[:n_examples]

    for i, idx in enumerate(indices):
        try:
            # Try to get the data (detect if it's 1D or 2D)
            if "seismic" in h5f and len(h5f["seismic"].shape) > 2:
                # Regular 2D data
                seismic = h5f["seismic"][idx]
                label = h5f["labels"][idx]
                is_1d = False
            else:
                # 1D data case
                seismic = h5f["seismic"][idx] if "seismic" in h5f else h5f["seismic_1d"][idx]
                label = h5f["labels"][idx] if "labels" in h5f else h5f["labels_1d"][idx]
                is_1d = len(seismic.shape) == 1

            # For 1D data, create a different type of visualization
            if is_1d:
                # Create figure for 1D visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                # Plot seismic as 1D signal
                x = np.arange(len(seismic))
                ax1.plot(x, seismic)
                ax1.set_title(f"Seismic Trace (1D) - Index {idx}")
                ax1.set_xlabel("Sample Index")
                ax1.set_ylabel("Amplitude")
                ax1.grid(True)

                # Plot labels as 1D signal with colors
                unique_labels = np.unique(label)
                for label_value in unique_labels:
                    # Create mask for this label
                    mask = label == label_value
                    # Get color for this label if available
                    if class_info and str(label_value) in class_info:
                        color = [c / 255 for c in class_info[str(label_value)]["color"]]
                        label_name = class_info[str(label_value)]["name"]
                    else:
                        # Default color scheme
                        cmap = plt.cm.get_cmap("tab10")
                        color = cmap(label_value % 10)
                        label_name = f"Class {label_value}"

                    # Plot only the points with this label
                    masked_x = x[mask]
                    if len(masked_x) > 0:  # Only plot if there are points with this label
                        ax2.scatter(
                            masked_x,
                            np.ones_like(masked_x) * label_value,
                            c=[color],
                            marker="|",
                            s=100,
                            label=label_name,
                        )

                ax2.set_title("Label Trace (1D)")
                ax2.set_xlabel("Sample Index")
                ax2.set_ylabel("Label Value")
                ax2.grid(True)

                # Add legend if we have class info
                if class_info:
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                # Regular 2D visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                # Plot seismic
                ax1.imshow(seismic, cmap="gray", aspect="auto")
                ax1.set_title(f"Seismic (2D) - Index {idx}")
                ax1.axis("off")

                # Plot labels with proper colormap if class_info is provided
                if class_info:
                    # Create custom colormap based on class_info
                    label_rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

                    for cls_idx, cls_data in class_info.items():
                        mask = label == int(cls_idx)
                        color = cls_data["color"]

                        for c in range(3):
                            label_rgb[:, :, c][mask] = color[c]

                    ax2.imshow(label_rgb, aspect="auto")
                else:
                    ax2.imshow(label, cmap="tab10", aspect="auto")

                ax2.set_title("Labels (2D)")
                ax2.axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / f"example_{i}.png", dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error saving example {i} (index {idx}): {e}")
            import traceback

            traceback.print_exc()

    print(f"Saved examples to {output_dir}")
