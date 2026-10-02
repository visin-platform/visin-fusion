"""CLFT's segmentation and depth heads."""

import torch.nn as nn


class Interpolate(nn.Module):
    """Upsampling as a module, so it can sit in a ``Sequential``."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """``x`` resized by the configured scale factor and mode."""
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class HeadDepth(nn.Module):
    """Depth head: convolutions and upsampling to one channel, squashed to 0..1 by a sigmoid."""

    def __init__(self, features):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """A depth map in 0..1, ``[B, 1, H, W]``."""
        return self.head(x)


class HeadSeg(nn.Module):
    """Segmentation head: convolutions and upsampling to one channel per class."""

    def __init__(self, features, nclasses=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """Class scores, ``[B, nclasses, H, W]``."""
        return self.head(x)
