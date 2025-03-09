import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import your dataset class
from seismic_segmentation.data.dataset import SeismicSegDataset


def validate_h5_data(h5_path):
    """Validate the HDF5 file structure and contents."""
    print(f"Validating HDF5 file: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        # Check available datasets
        print("Available datasets:", list(h5f.keys()))

        # Check data shapes
        seismic_shape = h5f["seismic"].shape
        labels_shape = h5f["labels"].shape
        print(f"Seismic data shape: {seismic_shape}")
        print(f"Labels shape: {labels_shape}")

        # Check index arrays
        train_indices = h5f["train_indices"][:]
        val_indices = h5f["val_indices"][:]
        test_indices = h5f["test_indices"][:]
        print(
            f"Train indices: {len(train_indices)} (min={train_indices.min()}, max={train_indices.max()})"
        )
        print(f"Val indices: {len(val_indices)} (min={val_indices.min()}, max={val_indices.max()})")
        print(
            f"Test indices: {len(test_indices)} (min={test_indices.min()}, max={test_indices.max()})"
        )

        # Check for data type issues
        print(f"Seismic data type: {h5f['seismic'].dtype}")
        print(f"Labels data type: {h5f['labels'].dtype}")

        # Check for value ranges
        for name, dataset in [("seismic", h5f["seismic"]), ("labels", h5f["labels"])]:
            sample_idx = train_indices[0]
            sample = dataset[sample_idx]
            print(f"{name} sample shape: {sample.shape}")
            print(f"{name} min: {np.min(sample)}, max: {np.max(sample)}, mean: {np.mean(sample)}")
            unique_values = np.unique(sample)
            print(
                f"{name} unique values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}"
            )


def visualize_sample_and_model_output(config, model_type="unet"):
    """Load a sample batch and pass it through the model to check shapes."""
    from seismic_segmentation.models import get_model

    # Create model
    model = get_model({**config, "model_type": model_type})

    # Create dataloader for a single batch
    dataset = SeismicSegDataset(config, split="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Get a batch
    batch = next(iter(dataloader))
    images = batch["image"]
    masks = batch["mask"]

    print(f"Input images shape: {images.shape}")
    print(f"Ground truth masks shape: {masks.shape}")

    # Try model forward pass
    if model_type == "sag":
        # For SAG model, generate dummy prompts
        B = images.size(0)
        point_prompts = torch.rand(B, 1, 3)
        box_prompts = torch.rand(B, 1, 4)
        well_prompts = torch.rand(B, 1, 10)

        outputs = model(
            images, point_prompts=point_prompts, box_prompts=box_prompts, well_prompts=well_prompts
        )
    else:
        outputs = model(images)

    print(f"Model output shape: {outputs.shape}")

    # Visualize a sample
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    img = images[0, 0].numpy()  # First image, first channel
    mask = masks[0].numpy()  # First mask

    # Model prediction
    if config.get("n_classes", 1) > 1:
        pred = torch.argmax(outputs, dim=1)[0].detach().numpy()
    else:
        pred = (torch.sigmoid(outputs) > 0.5).float()[0, 0].detach().numpy()

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="tab10")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="tab10")
    axes[2].set_title("Model Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("sample_visualization.png")
    plt.close()

    return images.shape, masks.shape, outputs.shape


if __name__ == "__main__":
    # Path to your HDF5 file
    h5_path = "processed_data/data.h5"

    # Validate HDF5 file
    validate_h5_data(h5_path)

    # Basic config for testing
    config = {
        "processed_data_dir": "processed_data",
        "n_classes": 6,
        "patch_size": 64,  # Use smaller patch size for testing
        "use_patches": True,
    }

    # Check UNet model
    try:
        print("\nTesting UNet model:")
        in_shape, gt_shape, out_shape = visualize_sample_and_model_output(config, "unet")
        print(f"UNet shapes - Input: {in_shape}, GT: {gt_shape}, Output: {out_shape}")
    except Exception as e:
        print(f"Error testing UNet: {e}")

    # Check SAG model
    try:
        print("\nTesting SAG model:")
        in_shape, gt_shape, out_shape = visualize_sample_and_model_output(config, "sag")
        print(f"SAG shapes - Input: {in_shape}, GT: {gt_shape}, Output: {out_shape}")
    except Exception as e:
        print(f"Error testing SAG: {e}")
