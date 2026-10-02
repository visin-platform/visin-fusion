"""A minimal model added through register_model, for the tests of the registry and the ``plugins`` setting."""

import torch
from torch import nn

from visin_fusion.models import FusionModel, register_model


class TinyBody(nn.Module):
    def __init__(self, num_classes, width):
        super().__init__()
        self.rgb = nn.Conv2d(3, width, 3, padding=1)
        self.lidar = nn.Conv2d(3, width, 3, padding=1)
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, rgb, lidar, modal="rgb"):
        features = self.rgb(rgb) + (self.lidar(lidar) if modal != "rgb" else 0)
        return self.head(torch.relu(features))


class TinyNet(FusionModel):
    def __init__(self, num_classes=4, mode="cross_fusion", width=8, pretrained=False, training_options=None):
        super().__init__(TinyBody(num_classes, width), mode, training_options)


register_model("tinynet", TinyNet, section="TinyNet")
