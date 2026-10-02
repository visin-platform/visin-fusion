"""Layers shared by more than one model, defined once.

Checkpoint keys come from attribute names (``conv1``, ``project``...), never from class names, so moving a
layer here does not change any checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualConvUnit(nn.Module):
    """ReLU-conv-ReLU-conv with a skip connection, keeping the channel count and spatial size."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``conv2(relu(conv1(relu(x)))) + x``."""
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class Read_ignore(nn.Module):
    """Drop the class (and distillation) tokens of a ViT token sequence."""

    def __init__(self, start_index: int = 1) -> None:
        super().__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The patch tokens, ``[B, N, C]``."""
        return x[:, self.start_index :]


class Read_add(nn.Module):
    """Add the class token (the mean of two when ``start_index`` is 2) to every patch token."""

    def __init__(self, start_index: int = 1) -> None:
        super().__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The patch tokens plus the readout token, ``[B, N, C]``."""
        readout = (x[:, 0] + x[:, 1]) / 2 if self.start_index == 2 else x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class Read_projection(nn.Module):
    """Concatenate the class token to every patch token and project back to the feature size."""

    def __init__(self, in_features: int, start_index: int = 1) -> None:
        super().__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The projected patch tokens, ``[B, N, C]``."""
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)
