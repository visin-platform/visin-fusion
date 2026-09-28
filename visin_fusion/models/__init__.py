"""Standalone models: each accepts Python arguments and returns dense logits."""
from .api import CLFT, CLFTv2, DeepLabV3Plus, Mask2FormerFusion, MaskFormerFusion

__all__ = ['CLFT', 'CLFTv2', 'MaskFormerFusion', 'Mask2FormerFusion', 'DeepLabV3Plus']
