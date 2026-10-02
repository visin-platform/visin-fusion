"""The input preprocessing a model was trained with, for inference and visualization.

``DatasetPNG`` applies the same resize and normalization during training (plus augmentation);
``tests/unit/test_preprocessing.py`` keeps the two identical.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

ImageInput = str | os.PathLike | Image.Image


class Preprocessor:
    """Turns a camera image and a LiDAR projection into the tensors the models take.

    Built from the ``transforms`` of a config (or of a checkpoint's ``model_info``): ``resize``,
    ``image_mean``, ``image_std`` and, when the dataset has them, ``lidar_mean`` and ``lidar_std``.
    Images are resized to ``resize`` x ``resize``; a LiDAR projection is normalized only when its
    statistics are known. Inputs are file paths or PIL images.
    """

    def __init__(self, transforms_config: Mapping) -> None:
        self.size = transforms_config["resize"]
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=transforms_config["image_mean"], std=transforms_config["image_std"]),
            ]
        )
        mean, std = transforms_config.get("lidar_mean"), transforms_config.get("lidar_std")
        self.lidar_normalize = (
            transforms.Normalize(mean=mean, std=std) if mean is not None and std is not None else None
        )

    @classmethod
    def from_config(cls, config: Mapping) -> Preprocessor:
        """The preprocessing of a pipeline config's ``Dataset.transforms``."""
        return cls(config["Dataset"]["transforms"])

    def load_rgb(self, image: ImageInput) -> torch.Tensor:
        """A camera image as a normalized ``[3, size, size]`` tensor."""
        return self.rgb_transform(_open(image).convert("RGB"))

    def load_lidar(self, image: ImageInput) -> torch.Tensor:
        """A LiDAR projection as a ``[3, size, size]`` tensor, normalized when statistics are known."""
        lidar = _open(image).resize((self.size, self.size), resample=Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(lidar)
        return self.lidar_normalize(tensor) if self.lidar_normalize is not None else tensor


def _open(image: ImageInput) -> Image.Image:
    return image if isinstance(image, Image.Image) else Image.open(image)
