"""Turns a ViT's token sequence back into spatial feature maps at four scales (CLFT's reassembly)."""

import torch.nn as nn
from einops.layers.torch import Rearrange

from visin_fusion.models.layers import Read_add, Read_ignore, Read_projection


class MyConvTranspose2d(nn.Module):
    """A transposed convolution that produces a fixed output size."""

    def __init__(self, conv, output_size):
        super().__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        """The transposed convolution of ``x`` at the configured output size."""
        return self.conv(x, output_size=self.output_size)


class Resample(nn.Module):
    """Projects a reassembled map to ``resample_dim`` channels and brings it to the scale ``s``."""

    def __init__(self, p, s, h, emb_dim, resample_dim):
        super().__init__()
        if s not in (4, 8, 16, 32):
            raise ValueError(f"reassemble scale must be one of 4, 8, 16, 32, got {s}")
        self.conv1 = nn.Conv2d(emb_dim, resample_dim, kernel_size=1, stride=1, padding=0)
        if s == 4:
            self.conv2 = nn.ConvTranspose2d(
                resample_dim, resample_dim, kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1
            )
        elif s == 8:
            self.conv2 = nn.ConvTranspose2d(
                resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1
            )
        elif s == 16:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x):
        """``x`` projected and resampled, ``[B, resample_dim, H', W']``."""
        x = self.conv1(x)
        return self.conv2(x)


class Reassemble(nn.Module):
    """Turns a ViT's token sequence into a spatial map at one scale."""

    def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
        """
        p = patch size
        s = coefficient resample
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()
        _channels, image_height, image_width = image_size

        # Read
        self.read = Read_ignore()
        if read == "add":
            self.read = Read_add()
        elif read == "projection":
            self.read = Read_projection(emb_dim)

        # Concat after read
        self.concat = Rearrange("b (h w) c -> b c h w", c=emb_dim, h=(image_height // p), w=(image_width // p))

        # Projection + Resample
        self.resample = Resample(p, s, image_height, emb_dim, resample_dim)

    def forward(self, x):
        """The tokens ``[B, N, C]`` as a map ``[B, resample_dim, H', W']``."""
        x = self.read(x)
        x = self.concat(x)
        return self.resample(x)
