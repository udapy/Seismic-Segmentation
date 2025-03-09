"""SAG model implementation for seismic segmentation."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank update (LoRA)."""

    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        """
        Initialize the LoRALinear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            r: Rank of the update matrix
            alpha: Scaling factor
            bias: Whether to use a bias term
        """
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # Freeze original weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA components
        self.A = nn.Parameter(torch.randn(out_features, r))
        self.B = nn.Parameter(torch.randn(r, in_features))
        self.scaling = self.alpha / self.r

        # Initialize weights
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        """
        Forward pass combining the frozen weight with the LoRA update.

        Args:
            x: Input tensor
        """
        base = F.linear(x, self.weight, self.bias)
        lora_update = F.linear(x, self.B.transpose(0, 1))
        lora_update = F.linear(lora_update, self.A.transpose(0, 1))
        return base + self.scaling * lora_update


class TransformerBlock(nn.Module):
    """Transformer block with LoRA-enabled MLP."""

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1, lora_r=4):
        """
        Initialize the TransformerBlock.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio for hidden dimension in MLP
            dropout: Dropout rate
            lora_r: Rank for LoRA layers
        """
        super(TransformerBlock, self).__init__()

        # Ensure embed_dim is divisible by num_heads for multi-head attention
        if embed_dim % num_heads != 0:
            # Find the largest factor of embed_dim that's <= num_heads
            factors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
            num_heads = max([f for f in factors if f <= num_heads])
            print(f"Adjusted num_heads to {num_heads} to be compatible with embed_dim {embed_dim}")

        self.norm1 = nn.LayerNorm(embed_dim)
        # Use a fixed head_dim to ensure compatibility
        # head_dim = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=embed_dim,  # Explicitly set key dimension
            vdim=embed_dim,  # Explicitly set value dimension
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP with LoRA
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_fc1 = LoRALinear(embed_dim, hidden_dim, r=lora_r, alpha=1.0)
        self.mlp_fc2 = LoRALinear(hidden_dim, embed_dim, r=lora_r, alpha=1.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (B, N, D)
        """
        # Print shapes for debugging
        print(f"TransformerBlock input shape: {x.shape}")

        # Self-attention with skip connection
        x_norm = self.norm1(x)

        # Ensure dimensions are compatible for attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # MLP with skip connection
        x_norm = self.norm2(x)
        mlp_out = F.gelu(self.mlp_fc1(x_norm))
        mlp_out = self.dropout(mlp_out)
        mlp_out = self.mlp_fc2(mlp_out)
        x = x + self.dropout(mlp_out)

        print(f"TransformerBlock output shape: {x.shape}")
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for image processing."""

    def __init__(
        self, img_size=256, patch_size=16, in_channels=1, embed_dim=768, num_layers=6, num_heads=8
    ):
        """
        Initialize the ViT encoder.

        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super(ViTEncoder, self).__init__()

        # Ensure img_size is divisible by patch_size
        if img_size % patch_size != 0:
            raise ValueError(
                f"Image size ({img_size}) must be divisible by patch size ({patch_size})"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Convert image to patches and embed
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        """
        Forward pass through the ViT encoder.

        Args:
            x: Input tensor of shape (B, C, H, W)
        """
        # Check input shape
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            # Resize input if needed
            x = F.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )

        # Convert image to patches
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)

        # Reshape to sequence of patches
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        return x  # (B, num_patches, embed_dim)


class PromptEncoder(nn.Module):
    """Encoder for various prompt types (points, boxes, well-logs)."""

    def __init__(self, prompt_dim=256):
        """
        Initialize the prompt encoder.

        Args:
            prompt_dim: Dimension of prompt embeddings
        """
        super(PromptEncoder, self).__init__()

        # Projection layers for different prompt types
        self.point_proj = nn.Linear(3, prompt_dim)  # [x, y, flag]
        self.box_proj = nn.Linear(4, prompt_dim)  # [x_min, y_min, x_max, y_max]
        self.well_proj = nn.Linear(10, prompt_dim)  # 10-d well-log input

    def forward(self, point_prompts=None, box_prompts=None, well_prompts=None):
        """
        Forward pass to encode various prompt types.

        Args:
            point_prompts: Point prompt tensor (B, N_points, 3)
            box_prompts: Box prompt tensor (B, N_boxes, 4)
            well_prompts: Well-log prompt tensor (B, N_wells, 10)
        """
        embeddings = []

        # Process point prompts if provided
        if point_prompts is not None:
            point_emb = F.relu(self.point_proj(point_prompts))  # (B, num_points, prompt_dim)
            embeddings.append(point_emb.mean(dim=1))

        # Process box prompts if provided
        if box_prompts is not None:
            box_emb = F.relu(self.box_proj(box_prompts))
            embeddings.append(box_emb.mean(dim=1))

        # Process well-log prompts if provided
        if well_prompts is not None:
            well_emb = F.relu(self.well_proj(well_prompts))
            embeddings.append(well_emb.mean(dim=1))

        # Combine embeddings if any are provided
        if embeddings:
            prompt_embedding = torch.stack(embeddings, dim=1).mean(dim=1)
        else:
            # Fallback: create zeros based on batch size
            B = 1
            if point_prompts is not None:
                B = point_prompts.shape[0]
            elif box_prompts is not None:
                B = box_prompts.shape[0]
            elif well_prompts is not None:
                B = well_prompts.shape[0]

            device = next(self.parameters()).device
            prompt_embedding = torch.zeros((B, 256), device=device)

        return prompt_embedding  # (B, prompt_dim)


class MaskDecoder(nn.Module):
    """Decoder that fuses image features with prompts and generates segmentation masks."""

    def __init__(self, embed_dim=768, prompt_dim=256, hidden_dim=512, out_size=256, n_classes=1):
        """
        Initialize the mask decoder.

        Args:
            embed_dim: Dimension of image embeddings
            prompt_dim: Dimension of prompt embeddings
            hidden_dim: Hidden dimension for the decoder
            out_size: Output size of the mask
            n_classes: Number of output classes
        """
        super(MaskDecoder, self).__init__()
        self.fuse_fc = nn.Linear(embed_dim + prompt_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_size = out_size
        self.n_classes = n_classes

        # Mask generation head
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_classes, kernel_size=1),
        )

    def forward(self, image_tokens, prompt_embedding):
        """
        Forward pass through the mask decoder.

        Args:
            image_tokens: Image tokens from ViT encoder (B, N, embed_dim)
            prompt_embedding: Prompt embedding (B, prompt_dim)
        """
        B, N, embed_dim = image_tokens.shape
        prompt_dim = prompt_embedding.shape[1]

        # Expand prompt embedding to match image tokens
        prompt_expand = prompt_embedding.unsqueeze(1).expand(B, N, prompt_dim)

        # Concatenate image tokens with prompt embedding
        fused = torch.cat([image_tokens, prompt_expand], dim=-1)

        # Process through MLP
        fused = F.relu(self.fuse_fc(fused))
        fused = self.norm(fused)

        # Reshape to 2D spatial grid
        grid_size = int(np.sqrt(N))
        fused = fused.reshape(B, grid_size, grid_size, -1).permute(0, 3, 1, 2)

        # Generate mask
        mask = self.mask_head(fused)

        # Upsample to target size
        mask = F.interpolate(
            mask, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )

        return mask  # (B, n_classes, out_size, out_size)


class SAGModel(nn.Module):
    """Complete SAG model integrating ViTEncoder, PromptEncoder, and MaskDecoder."""

    def __init__(self, config):
        """
        Initialize the SAG model.

        Args:
            config: Configuration dictionary
        """
        super(SAGModel, self).__init__()
        self.img_size = config.get("img_size", 64)  # Use a smaller default
        self.n_classes = config.get("n_classes", 6)
        self.embed_dim = config.get("embed_dim", 768)
        self.num_layers = config.get("num_layers", 6)
        self.patch_size = config.get("patch_size", 8)  # Use a smaller default
        self.prompt_dim = config.get("prompt_dim", 256)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_heads = config.get("num_heads", 12)  # 768/12 = 64 (evenly divisible)

        # Ensure embed_dim is divisible by num_heads
        if self.embed_dim % self.num_heads != 0:
            # Find the largest factor of embed_dim that is <= num_heads
            factors = [i for i in range(1, self.embed_dim + 1) if self.embed_dim % i == 0]
            self.num_heads = max([f for f in factors if f <= self.num_heads])
            print(
                f"Adjusted num_heads to {self.num_heads} to be compatible with embed_dim {self.embed_dim}"
            )

        # Ensure img_size is divisible by patch_size
        if self.img_size % self.patch_size != 0:
            # Adjust patch_size to be a divisor of img_size
            factors = [i for i in range(1, self.img_size + 1) if self.img_size % i == 0]
            self.patch_size = max([f for f in factors if f <= self.patch_size])
            print(
                f"Adjusted patch_size to {self.patch_size} to be compatible with img_size {self.img_size}"
            )

        # This flag helps identify if this is a SAG model
        self.use_sag = True

        # Image encoder (ViT)
        self.image_encoder = ViTEncoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=1,  # Seismic data has 1 channel
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(prompt_dim=self.prompt_dim)

        # Mask decoder
        self.mask_decoder = MaskDecoder(
            embed_dim=self.embed_dim,
            prompt_dim=self.prompt_dim,
            hidden_dim=self.hidden_dim,
            out_size=self.img_size,
            n_classes=self.n_classes,
        )

    def forward(self, image, point_prompts=None, box_prompts=None, well_prompts=None):
        """
        Forward pass through the SAG model.

        Args:
            image: Input image tensor (B, C, H, W)
            point_prompts: Point prompt tensor (optional)
            box_prompts: Box prompt tensor (optional)
            well_prompts: Well-log prompt tensor (optional)
        """
        # Ensure image has the right dimensions
        B, C, H, W = image.shape
        if H != self.img_size or W != self.img_size:
            # Resize input if needed
            image = F.interpolate(
                image, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )

        # Encode image into tokens
        image_tokens = self.image_encoder(image)  # (B, N, embed_dim)

        # Encode prompts
        prompt_embedding = self.prompt_encoder(point_prompts, box_prompts, well_prompts)

        # Generate mask
        mask = self.mask_decoder(image_tokens, prompt_embedding)

        # Ensure output has the original dimensions
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

        return mask
