#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced model builder for state-of-the-art transformers.
"""
import torch
from visin_fusion.models.clftv2 import CLFTv2Network


class AdvancedModelBuilder:
    """Handles model creation for advanced transformers."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.num_unique_classes = self._calculate_unique_classes()
    
    def _calculate_unique_classes(self):
        """Calculate number of training classes."""
        return len(self.config['Dataset']['train_classes'])
    
    def build_model(self):
        """Build advanced model based on config."""

        pretrained = self.config['CLFTv2'].get('pretrained', True)
        fusion_strategy = self.config['CLFTv2']['fusion_strategy']
        
        model = CLFTv2Network(
            emb_dims=self.config['CLFTv2'].get('emb_dims', None),
            resample_dim=self.config['CLFTv2']['resample_dim'],
            read=self.config['CLFTv2']['read'],
            reassemble_s=self.config['CLFTv2']['reassembles'],
            nclasses=self.num_unique_classes,
            type=self.config['CLFTv2']['type'],
            model_timm=self.config['CLFTv2']['model_timm'],
            pretrained=pretrained,
            fusion_strategy=fusion_strategy
        )
        
        print(f"Built CLFTv2Network model with {self.num_unique_classes} classes (pretrained: {pretrained}, fusion: {fusion_strategy})")
        return model
    
    def load_checkpoint(self, model, checkpoint_path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Strict: a checkpoint that does not match the model (another fusion strategy, class count or
        # backbone) must fail, not leave the mismatched layers randomly initialised
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
        return model, epoch