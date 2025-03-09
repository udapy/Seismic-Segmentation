# debug_sag_model.py
import torch

from seismic_segmentation.models.sag_model import SAGModel


def debug_sag_model():
    # Create a simple config
    config = {
        "img_size": 64,  # Must be divisible by patch_size
        "n_classes": 6,
        "embed_dim": 768,
        "num_layers": 2,  # Smaller for faster testing
        "patch_size": 16,
        "prompt_dim": 256,
        "hidden_dim": 512,
    }

    # Create model
    model = SAGModel(config)

    # Create dummy input
    batch_size = 2
    img = torch.randn(batch_size, 1, config["img_size"], config["img_size"])

    # Create dummy prompts
    point_prompts = torch.rand(batch_size, 1, 3)
    box_prompts = torch.rand(batch_size, 1, 4)
    well_prompts = torch.rand(batch_size, 1, 10)

    # Enable debug prints inside the model
    print("=== Input shapes ===")
    print(f"Image: {img.shape}")
    print(f"Point prompts: {point_prompts.shape}")
    print(f"Box prompts: {box_prompts.shape}")
    print(f"Well prompts: {well_prompts.shape}")

    # Test image encoder
    print("\n=== Testing ViT encoder ===")
    image_tokens = model.image_encoder(img)
    print(f"Image tokens shape: {image_tokens.shape}")

    # Test prompt encoder
    print("\n=== Testing prompt encoder ===")
    prompt_embedding = model.prompt_encoder(point_prompts, box_prompts, well_prompts)
    print(f"Prompt embedding shape: {prompt_embedding.shape}")

    # Test mask decoder
    print("\n=== Testing mask decoder ===")
    mask = model.mask_decoder(image_tokens, prompt_embedding)
    print(f"Mask shape: {mask.shape}")

    # Test full model forward pass
    print("\n=== Testing full model ===")
    output = model(
        img, point_prompts=point_prompts, box_prompts=box_prompts, well_prompts=well_prompts
    )
    print(f"Model output shape: {output.shape}")

    return "SAG model test completed successfully!"


if __name__ == "__main__":
    try:
        result = debug_sag_model()
        print(result)
    except Exception as e:
        print(f"Error testing SAG model: {e}")
        import traceback

        traceback.print_exc()
