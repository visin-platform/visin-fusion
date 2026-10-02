#!/usr/bin/env python3

"""
Modified dataset loader that uses pre-computed PNG projections instead of pickle files.

PNG Format Specification:
- RGB image where R=X, G=Y, B=Z coordinate projections
- Values: 1-255 (uint8) for normalized coordinates, 0 for empty pixels
- Original dimensions: 1363x768 for ZOD, 1920x1280 for Waymo
- File naming: frame_XXXXXX.png (same as original pickle but .png extension)
"""

import logging
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from visin_fusion.utils.helpers import get_annotation_path, get_lidar_path

logger = logging.getLogger(__name__)


class DatasetPNG(Dataset):
    """
    Dataset class for loading LiDAR projections from PNG files.
    Replaces the original Dataset class but uses PNG files instead of pickle files.
    """

    def __init__(self, config, split=None, path=None):
        """
        Initialize dataset.

        Args:
            config (dict): Configuration dictionary
            split (str): Data split ('train', 'val', 'test')
            path (str): Path to split file (e.g., 'train.txt')
        """
        self.config = config

        # Read the split file to get list of examples
        with open(path) as list_examples_file:
            self.list_examples_cam = np.array(list_examples_file.read().splitlines())

        # Set augmentation probabilities based on split
        if split == "train":  # only augment for training.
            self.p_flip = config["Dataset"]["transforms"]["p_flip"]
            self.p_crop = config["Dataset"]["transforms"]["p_crop"]
            self.p_rot = config["Dataset"]["transforms"]["p_rot"]
        else:
            self.p_flip = 0
            self.p_crop = 0
            self.p_rot = 0

        self.img_size = config["Dataset"]["transforms"]["resize"]
        lidar_mean = config["Dataset"]["transforms"].get("lidar_mean")
        lidar_std = config["Dataset"]["transforms"].get("lidar_std")
        self.lidar_normalize = (
            transforms.Normalize(mean=lidar_mean, std=lidar_std)
            if lidar_mean is not None and lidar_std is not None
            else None
        )
        self.rgb_normalize = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config["Dataset"]["transforms"]["image_mean"], std=config["Dataset"]["transforms"]["image_std"]
                ),
            ]
        )

        self.anno_size = self.img_size

        # Labels are resized with pixel-centre nearest sampling ("nearest-exact"), like the bilinear
        # image resize. Plain "nearest" samples at floor(i * scale), which shifts labels against the
        # image, and a flip reverses that shift (a one-pixel disagreement at every boundary).
        self.anno_resize = transforms.Resize(
            (self.anno_size, self.anno_size), interpolation=transforms.InterpolationMode.NEAREST_EXACT
        )

    def __len__(self):
        """Return dataset length."""
        return len(self.list_examples_cam)

    def __getitem__(self, idx):
        """
        Get item from dataset.

        Args:
            idx (int): Index of the item

        Returns:
            dict: Dictionary containing processed data
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dataroot = self.config["Dataset"]["dataset_root"]

        # Construct paths (generic approach)
        cam_path = os.path.join(dataroot, self.list_examples_cam[idx])

        rgb = Image.open(cam_path).convert("RGB")

        # A frame without an annotation file gets an empty mask, but says so: a wrong
        # annotation_path would otherwise train on empty labels unnoticed
        anno_path = get_annotation_path(cam_path, self.config)
        if os.path.exists(anno_path):
            anno = torch.from_numpy(np.array(Image.open(anno_path))).unsqueeze(0).long()
        else:
            logger.warning("No annotation %s; using an empty mask", anno_path)
            anno = torch.zeros((1, rgb.height, rgb.width), dtype=torch.long)

        png_path = get_lidar_path(cam_path, self.config)

        # RGB-only runs get a black stand-in instead of reading the LiDAR projection
        rgb_only = self.config.get("CLI", {}).get("mode") == "rgb"
        lidar_pil = Image.new("RGB", rgb.size, (0, 0, 0)) if rgb_only else self.load_lidar_png(png_path)

        # Validate filenames match
        rgb_name = os.path.splitext(os.path.basename(cam_path))[0]
        # Only validate annotation name match if we loaded real annotations (not dummy)
        if anno_path and os.path.exists(anno_path):
            anno_name = os.path.splitext(os.path.basename(anno_path))[0]
            if rgb_name != anno_name:
                raise ValueError(f"rgb and annotation names do not match: {rgb_name} vs {anno_name}")
        if not rgb_only:
            png_name = os.path.splitext(os.path.basename(png_path))[0]
            if rgb_name != png_name:
                raise ValueError(f"rgb and LiDAR names do not match: {rgb_name} vs {png_name}")

        # Keep full image (no cropping)
        rgb_orig = rgb.copy()
        rgb_aug = rgb
        anno_aug = anno
        lidar_aug = lidar_pil

        # Apply horizontal flip
        if random.random() < self.p_flip:
            rgb_aug = TF.hflip(rgb_aug)
            anno_aug = TF.hflip(anno_aug)
            lidar_aug = TF.hflip(lidar_aug)

        # Apply random crop
        if random.random() < self.p_crop:
            # Use resized crop for more variation (like DataAugment)
            max_size = max(1, self.img_size - 1)
            random_size = random.randint(min(128, max_size), max_size)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                rgb_aug, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            rgb_aug = TF.resized_crop(
                rgb_aug, i, j, h, w, (random_size, random_size), interpolation=TF.InterpolationMode.BILINEAR
            )
            anno_aug = TF.resized_crop(
                anno_aug, i, j, h, w, (random_size, random_size), interpolation=TF.InterpolationMode.NEAREST_EXACT
            )
            lidar_aug = TF.resized_crop(
                lidar_aug, i, j, h, w, (random_size, random_size), interpolation=TF.InterpolationMode.BILINEAR
            )

        # Apply random rotation
        if random.random() < self.p_rot:
            rotate_range = self.config["Dataset"]["transforms"]["random_rotate_range"]
            angle = (-rotate_range + 2 * rotate_range * torch.rand(1)[0]).item()
            rgb_aug = TF.rotate(rgb_aug, angle)
            anno_aug = TF.rotate(anno_aug, angle, interpolation=TF.InterpolationMode.NEAREST)
            lidar_aug = TF.rotate(lidar_aug, angle)

        # Normalize and resize (same as original)
        rgb = self.rgb_normalize(rgb_aug)  # Tensor [3, 384, 384]
        anno = self.anno_resize(anno_aug).squeeze(0)  # Tensor [384, 384]

        rgb_orig = transforms.ToTensor()(rgb_orig)

        # Resize lidar to match image size and convert to tensor
        lidar_resized = transforms.Resize((self.img_size, self.img_size))(lidar_aug)
        lidar_tensor = TF.to_tensor(lidar_resized)
        if self.lidar_normalize is not None:
            lidar_tensor = self.lidar_normalize(lidar_tensor)

        return {
            "rgb": rgb,
            "rgb_orig": rgb_orig,
            "lidar": lidar_tensor,
            "anno": anno,
        }

    def load_lidar_png(self, png_path):
        """
        Load LiDAR projection from PNG file.

        Args:
            png_path (str): Path to PNG file containing LiDAR projection

        Returns:
            PIL.Image: LiDAR projection as PIL image
        """
        # Load PNG image
        return Image.open(png_path)
