# src/seismic_segmentation/models/unet.py
"""UNet implementation with ResNet-34 encoder for seismic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class UNetResNet34(nn.Module):
    """UNet architecture with a ResNet-34 encoder."""

    def __init__(self, config):
        """
        Initialize the UNetResNet34 model.

        Args:
            config: Configuration object with model parameters
        """
        super(UNetResNet34, self).__init__()
        self.n_classes = config.get("n_classes", 6)
        pretrained = config.get("pretrained", True)

        # Use 1 input channel for seismic data (not 3)
        self.modify_first_layer = True

        # Load pretrained ResNet-34 as encoder
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Modify first layer to accept 1 channel
        if self.modify_first_layer:
            new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # Initialize with the average of RGB channels
                with torch.no_grad():
                    new_conv1.weight[:, 0:1, :, :] = torch.mean(
                        resnet.conv1.weight, dim=1, keepdim=True
                    )
            resnet.conv1 = new_conv1

        # Encoder layers
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # (B,64,H/2,W/2)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # (B,64,H/4,W/4)
        self.enc3 = resnet.layer2  # (B,128,H/8,W/8)
        self.enc4 = resnet.layer3  # (B,256,H/16,W/16)
        self.enc5 = resnet.layer4  # (B,512,H/32,W/32)

        # Decoder layers with skip connections
        self.up5 = self._up_block(512, 256)
        self.up4 = self._up_block(256, 128)
        self.up3 = self._up_block(128, 64)
        self.up2 = self._up_block(64, 64)
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final classification layer
        self.out_conv = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        """
        Create an upsampling block with convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, C, H, W)
        """
        # Store original input size
        input_h, input_w = x.shape[2], x.shape[3]

        # Encoder path - standard operations
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder path with skip connections
        d5 = self.up5(e5)
        d5 = d5 + e4  # Changed: adding tensors directly

        d4 = self.up4(d5)
        d4 = d4 + e3  # Changed: adding tensors directly

        d3 = self.up3(d4)
        d3 = d3 + e2  # Changed: adding tensors directly

        d2 = self.up2(d3)
        d2 = d2 + e1  # Changed: adding tensors directly

        d1 = self.up1(d2)

        # Final upsampling to original size
        d1 = F.interpolate(d1, size=(input_h, input_w), mode="bilinear", align_corners=True)

        # Final classification
        out = self.out_conv(d1)

        return out
