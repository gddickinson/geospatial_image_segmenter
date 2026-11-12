"""U-Net network and training dataset used by CNNModel (PyTorch)."""
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation."""

    def __init__(self, n_channels: int, n_classes: int):
        """Initialize U-Net.

        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
        """
        super().__init__()

        # Encoder path
        self.enc1 = self._make_layer(n_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)

        # Bridge
        self.bridge = self._make_layer(512, 1024)

        # Decoder path
        self.dec4 = self._make_layer(1024 + 512, 512)
        self.dec3 = self._make_layer(512 + 256, 256)
        self.dec2 = self._make_layer(256 + 128, 128)
        self.dec1 = self._make_layer(128 + 64, 64)

        # Final layer
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            nn.Sequential: Convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self, m):
        """Initialize layer weights.

        Args:
            m: Layer
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bridge
        bridge = self.bridge(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.dec4(torch.cat([F.interpolate(bridge, size=enc4.shape[2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:]), enc1], 1))

        return self.final(dec1)


class SegmentationDataset(Dataset):
    """Dataset for training segmentation model."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        patch_size: int = 256,
        stride: int = 128
    ):
        """Initialize dataset.

        Args:
            features: Feature array (channels, height, width)
            labels: Label array (height, width)
            patch_size: Size of image patches
            stride: Stride for patch extraction
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.patch_size = patch_size
        self.stride = stride

        # Calculate patch positions
        self.patches = []
        for i in range(0, features.shape[1] - patch_size + 1, stride):
            for j in range(0, features.shape[2] - patch_size + 1, stride):
                self.patches.append((i, j))

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            tuple: (feature patch, label patch)
        """
        i, j = self.patches[idx]
        feature_patch = self.features[:, i:i+self.patch_size, j:j+self.patch_size]
        label_patch = self.labels[i:i+self.patch_size, j:j+self.patch_size]
        return feature_patch, label_patch
