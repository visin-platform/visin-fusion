#!/usr/bin/env python3
"""
DeepLabV3+ implementation for comparison with CLFT.
Supports RGB-only and late-fusion variants.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights, resnet101

logger = logging.getLogger(__name__)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class Fusion(nn.Module):
    def __init__(self, resample_dim, fusion_strategy="residual_average"):
        super().__init__()
        self.resample_dim = resample_dim
        self.fusion_strategy = fusion_strategy

        if self.fusion_strategy != "residual_average":
            raise ValueError(f"Only 'residual_average' fusion strategy is supported, got: {self.fusion_strategy}")

    def forward(self, rgb, lidar, previous_stage=None, modal="cross_fusion"):
        if modal == "cross_fusion":
            # Simple residual average fusion: just average the features
            fused_feat = (rgb + lidar) / 2
            if previous_stage is not None:
                previous_stage = nn.functional.interpolate(
                    previous_stage, size=fused_feat.shape[2:], mode="bilinear", align_corners=False
                )
                fused_feat = fused_feat + previous_stage
            return fused_feat
        raise ValueError(f"Only 'cross_fusion' modal is supported for fusion, got: {modal}")


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # Atrous convolutions with different dilation rates
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.size()[2:]

        # Apply all branches
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode="bilinear", align_corners=True)

        # Concatenate and project
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.project(out)

        return out


class Decoder(nn.Module):
    """DeepLabV3+ decoder module."""

    def __init__(self, low_level_channels, num_classes, aspp_channels=256):
        super().__init__()

        # Low-level feature projection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )

        # Refinement convolutions
        self.refine = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, low_level_feat, input_size):
        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # Upsample high-level features to match low-level
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True)

        # Concatenate and refine
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.refine(x)

        # Final classification
        x = self.classifier(x)

        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        return x


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model for semantic segmentation.
    Supports RGB-only mode.
    """

    def __init__(self, num_classes, backbone="resnet101", pretrained=True):
        super().__init__()

        # Load ResNet backbone
        if backbone == "resnet101":
            if pretrained:
                resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels

        # Apply dilation to layer4 for atrous convolutions
        self.layer4 = self._make_layer4_atrous(resnet.layer4)

        # ASPP module (output channels from layer4 = 2048)
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Decoder (low-level features from layer1 = 256 channels)
        self.decoder = Decoder(low_level_channels=256, num_classes=num_classes)

    def _make_layer4_atrous(self, layer4):
        """Modify layer4 to use atrous convolutions."""
        for n, m in layer4.named_modules():
            if "conv2" in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)
        return layer4

    def forward(self, x):
        input_size = x.size()[2:]

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        low_level_feat = self.layer1(x)  # 1/4 resolution
        x = self.layer2(low_level_feat)  # 1/8 resolution
        x = self.layer3(x)  # 1/16 resolution
        x = self.layer4(x)  # 1/16 resolution (with atrous)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x, low_level_feat, input_size)

        return x


class DeepLabV3PlusLateFusion(nn.Module):
    """
    DeepLabV3+ with late fusion for camera-LiDAR data.
    Uses two separate DeepLabV3+ networks and fuses features using advanced fusion strategies.
    """

    def __init__(
        self, num_classes, fusion_strategy="residual_average", pretrained=True, resample_dim=256, backbone="resnet50"
    ):
        super().__init__()

        self.fusion_strategy = fusion_strategy

        # RGB branch - use lighter backbone
        self.rgb_branch = DeepLabV3Plus(num_classes, backbone=backbone, pretrained=pretrained)

        # LiDAR branch (pretrained on ImageNet but will learn LiDAR features)
        self.lidar_branch = DeepLabV3Plus(num_classes, backbone=backbone, pretrained=pretrained)

        # Remove the final classifier from both branches since we'll fuse before classification
        # We'll add a shared classifier after fusion
        self.rgb_branch.decoder.classifier = nn.Identity()
        self.lidar_branch.decoder.classifier = nn.Identity()

        # Shared classifier after fusion
        self.classifier = nn.Conv2d(256, num_classes, 1)

        # Fusion module using the same strategies as CLFTv2
        self.fusion = Fusion(resample_dim, fusion_strategy=fusion_strategy)

    def forward(self, rgb, lidar, modality="cross_fusion"):
        if modality == "cross_fusion":
            input_size = rgb.size()[2:]

            # Get features from both branches (before final classification)
            rgb_feat = self.rgb_branch(rgb)
            lidar_feat = self.lidar_branch(lidar)

            # Apply fusion
            fused_feat = self.fusion(rgb_feat, lidar_feat, previous_stage=None, modal="cross_fusion")

            # Final classification
            out = self.classifier(fused_feat)

            # Upsample to input size
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)

            # Also compute individual branch predictions for compatibility
            rgb_pred = self.classifier(rgb_feat)
            rgb_pred = F.interpolate(rgb_pred, size=input_size, mode="bilinear", align_corners=True)

            lidar_pred = self.classifier(lidar_feat)
            lidar_pred = F.interpolate(lidar_pred, size=input_size, mode="bilinear", align_corners=True)

            # Return format: (fused_output, rgb_pred, lidar_pred)
            return out, rgb_pred, lidar_pred
        # For single modality, use the appropriate branch
        if modality == "rgb":
            pred = self.rgb_branch(rgb)
            return pred, None, None
        if modality == "lidar":
            pred = self.lidar_branch(lidar)
            return pred, None, None
        raise ValueError(f"Invalid modality: {modality}")

    def forward_single(self, x, modality="rgb"):
        """Forward pass for single modality (for evaluation)."""
        if modality == "rgb":
            return self.rgb_branch(x)
        if modality == "lidar":
            return self.lidar_branch(x)
        raise ValueError(f"Invalid modality: {modality}")


def fusion_strategy_from_config(section):
    """The fusion strategy a DeepLabV3Plus config section asks for, the same for every stage.

    ``fusion_strategy`` is the key; ``fusion_type`` is its older name, still in some configs.
    """
    return section.get("fusion_strategy") or section.get("fusion_type") or "residual_average"


def build_deeplabv3plus(
    num_classes, mode="rgb", fusion_strategy="residual_average", pretrained=True, backbone="resnet101"
):
    """
    Build DeepLabV3+ model based on configuration.

    Args:
        num_classes (int): Number of output classes
        mode (str): 'rgb' for RGB-only, 'lidar' for LiDAR-only, 'fusion' for late fusion
        fusion_strategy (str): Fusion strategy ('residual_average')
        pretrained (bool): Use ImageNet pretrained weights
        backbone (str): Backbone architecture ('resnet50', 'resnet101')

    Returns:
        nn.Module: DeepLabV3+ model
    """
    if mode == "rgb":
        logger.info("Building DeepLabV3+ (RGB-only) with %s classes, backbone: %s", num_classes, backbone)
        return DeepLabV3Plus(num_classes, backbone=backbone, pretrained=pretrained)
    if mode == "lidar":
        logger.info("Building DeepLabV3+ (LiDAR-only) with %s classes, backbone: %s", num_classes, backbone)
        return DeepLabV3Plus(num_classes, backbone=backbone, pretrained=pretrained)
    if mode == "fusion":
        logger.info(
            "Building DeepLabV3+ (Late Fusion - %s) with %s classes, backbone: %s",
            fusion_strategy,
            num_classes,
            backbone,
        )
        return DeepLabV3PlusLateFusion(
            num_classes, fusion_strategy=fusion_strategy, pretrained=pretrained, backbone=backbone
        )
    raise ValueError(f"Unsupported mode: {mode}")
