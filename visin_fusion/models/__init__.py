"""Standalone models: each accepts Python arguments and returns dense logits."""

from .api import CLFT, CLFTv2, DeepLabV3Plus, FusionModel, Mask2FormerFusion, MaskFormerFusion
from .registry import register_model

__all__ = [
    "CLFT",
    "CLFTv2",
    "DeepLabV3Plus",
    "FusionModel",
    "Mask2FormerFusion",
    "MaskFormerFusion",
    "register_model",
]
