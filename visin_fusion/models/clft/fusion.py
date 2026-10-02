"""CLFT's residual fusion block, which combines the camera and LiDAR maps with the previous (coarser) stage."""

import torch
import torch.nn as nn

from visin_fusion.models.layers import ResidualConvUnit


class Fusion(nn.Module):
    """Fuses camera and LiDAR feature maps of one scale with the previous stage and upsamples 2x."""

    def __init__(self, resample_dim):
        super().__init__()
        self.res_conv_xyz = ResidualConvUnit(resample_dim)
        self.res_conv_rgb = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)

    def forward(self, rgb, lidar, previous_stage=None, modal="rgb"):
        """The fused map at twice the input resolution; the stream the mode does not use is zeros."""
        if previous_stage is None:
            previous_stage = torch.zeros_like(rgb)

        if modal == "rgb":
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1_lidar = torch.zeros_like(output_stage1_rgb)
        if modal == "lidar":
            output_stage1_lidar = self.res_conv_xyz(lidar)
            output_stage1_rgb = torch.zeros_like(output_stage1_lidar)
        if modal == "cross_fusion":
            output_stage1_rgb = self.res_conv_rgb(rgb)
            output_stage1_lidar = self.res_conv_xyz(lidar)

        output_stage1 = output_stage1_lidar + output_stage1_rgb + previous_stage
        output_stage2 = self.res_conv2(output_stage1)

        return nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
