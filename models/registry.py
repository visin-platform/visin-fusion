"""The one place that knows how each model is built from a config, called, and trained.

    model = build_model(config)                        # the model CLI.backbone names
    segmap = segment(model, config, rgb, lidar)        # [B, classes, H, W] for any model and mode
    Segmenter(model, config)(rgb, lidar)               # the same, as a module (for profilers)
    setup = training_setup(config, model, device)      # optimizer, schedule, loss, clipping
"""
import copy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from utils.config_schema import BACKBONES
from utils.helpers import calculate_num_classes


def model_section(config):
    """The config section of the model CLI.backbone names, e.g. config['SwinFusion']."""
    return config[BACKBONES[config['CLI']['backbone']][0]]


def build_model(config, pretrained=None):
    """The model for ``config``. ``pretrained`` overrides the config's choice (e.g. False to time a
    model without downloading weights); the config itself is not changed."""
    if pretrained is not None:
        config = copy.deepcopy(config)
        model_section(config)['pretrained'] = pretrained
    backbone, section = config['CLI']['backbone'], model_section(config)
    classes = calculate_num_classes(config)

    if backbone == 'clft':
        from core.model_builder import ModelBuilder
        return ModelBuilder(config, 'cpu').build_model()
    if backbone == 'swin_fusion':
        from core.advanced_model_builder import AdvancedModelBuilder
        return AdvancedModelBuilder(config, 'cpu').build_model()
    if backbone == 'maskformer':
        from models.maskformer_fusion import MaskFormerFusion
        return MaskFormerFusion(backbone=section['model_timm'], num_classes=classes,
                                pixel_decoder_channels=section['pixel_decoder_channels'],
                                transformer_d_model=section['transformer_d_model'],
                                num_queries=section['num_queries'], pretrained=section.get('pretrained', True))
    if backbone == 'mask2former':
        from models.mask2former_fusion import Mask2FormerFusion
        return Mask2FormerFusion(backbone=section['model_timm'], num_classes=classes,
                                 pixel_decoder_channels=section['pixel_decoder_channels'],
                                 transformer_d_model=section['transformer_d_model'],
                                 num_queries=section['num_queries'],
                                 num_decoder_layers=section.get('num_decoder_layers', 9),
                                 n_encoder_layers=section.get('n_encoder_layers', 6),
                                 pretrained=section.get('pretrained', True))
    if backbone == 'deeplabv3plus':
        from models.deeplabv3plus import build_deeplabv3plus, fusion_strategy_from_config
        return build_deeplabv3plus(classes, mode=config['CLI']['mode'],
                                   fusion_strategy=fusion_strategy_from_config(section),
                                   pretrained=section.get('pretrained', True), backbone=section.get('backbone', 'resnet101'))
    raise ValueError(f"no model for CLI.backbone {backbone!r}")


def forward(model, config, rgb, lidar):
    """The model's raw outputs, called the way each model expects its inputs."""
    backbone, mode = config['CLI']['backbone'], config['CLI']['mode']
    if backbone == 'deeplabv3plus':
        return model(rgb, lidar) if mode == 'fusion' else model(rgb if mode == 'rgb' else lidar)
    rgb_in, lidar_in = {'rgb': (rgb, rgb), 'lidar': (lidar, lidar)}.get(mode, (rgb, lidar))
    return model(rgb_in, lidar_in, modal=mode)


def segmentation(outputs, config):
    """The segmentation map ([B, classes, H, W]) in a model's raw outputs."""
    if config['CLI']['backbone'] == 'deeplabv3plus':
        return outputs[0] if isinstance(outputs, tuple) else outputs  # late fusion: (fused, rgb, lidar)
    return outputs[1]  # (depth or None, segmentation, ...)


def segment(model, config, rgb, lidar):
    """The segmentation map ([B, classes, H, W]), called the way the stage scripts call each model."""
    return segmentation(forward(model, config, rgb, lidar), config)


class Segmenter(nn.Module):
    """A model with one calling convention, ``forward(rgb, lidar) -> segmentation map``."""

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, rgb: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        return segment(self.model, self.config, rgb, lidar)


@dataclass
class TrainingSetup:
    """What training a model needs besides its data."""
    optimizer: torch.optim.Optimizer
    scheduler: Optional[object]  # stepped once per epoch, after validation
    loss: Callable  # loss(raw outputs, segmentation map, labels) -> scalar tensor
    clip_grad_norm: Optional[float] = None
    mixed_precision: bool = True


def _class_weighted_cross_entropy(config, device):
    classes = sorted(config['Dataset']['train_classes'], key=lambda c: c['index'])
    weights = torch.Tensor([c['weight'] for c in classes])
    print(f"Class weights {[c['weight'] for c in classes]} for {[c['name'] for c in classes]}")
    ce = nn.CrossEntropyLoss(weight=weights).to(device)
    return lambda outputs, segmap, labels: ce(segmap, labels)


def _warmup_then(optimizer, warmup_epochs, start_factor, main):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0,
                                               total_iters=warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])


def _query_model_schedule(optimizer, section, epochs):
    """MaskFormer / Mask2Former: warmup, then polynomial or step decay (as in their Detectron2 configs)."""
    warmup = section.get('warmup_epochs', 10)
    post_warmup = max(1, epochs - warmup)
    if section.get('lr_scheduler', 'poly') == 'step':
        milestones = [max(1, int(f * post_warmup)) for f in section.get('lr_step_milestones', [0.75, 0.90])]
        main = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                    gamma=section.get('lr_step_gamma', 0.1))
    else:
        main = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=post_warmup,
                                                     power=section.get('lr_poly_power', 0.9))
    return _warmup_then(optimizer, warmup, 1e-6, main)


def _backbone_slower(model, lr):
    """AdamW with the pretrained backbone at a tenth of the learning rate of the rest."""
    backbone = list(model.backbone.parameters())
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{'params': backbone, 'lr': lr * 0.1}, {'params': rest, 'lr': lr}], weight_decay=0.05)


def _deeplab_schedule(optimizer, section, epochs):
    lr = section.get('lr_scheduler')
    if not lr:
        return None
    kind = lr.get('type', 'step')
    schedulers = torch.optim.lr_scheduler
    if kind == 'step':
        return schedulers.StepLR(optimizer, step_size=lr.get('step_size', 30), gamma=lr.get('gamma', 0.1))
    if kind == 'cosine':
        return schedulers.CosineAnnealingLR(optimizer, T_max=lr.get('T_max', epochs), eta_min=lr.get('eta_min', 1e-6))
    if kind == 'exponential':
        return schedulers.ExponentialLR(optimizer, gamma=lr.get('gamma', 0.95))
    if kind == 'plateau':  # stepped with the validation mIoU
        return schedulers.ReduceLROnPlateau(optimizer, mode=lr.get('mode', 'max'), factor=lr.get('factor', 0.1),
                                            patience=lr.get('patience', 10), min_lr=lr.get('min_lr', 1e-6))
    raise ValueError(f"DeepLabV3Plus.lr_scheduler.type {kind!r} is not one of step, cosine, exponential, plateau")


def training_setup(config, model, device):
    """The optimizer, learning-rate schedule, loss and gradient clipping each model is trained with."""
    backbone, section, epochs = config['CLI']['backbone'], model_section(config), config['General']['epochs']

    if backbone == 'clft':
        optimizer = torch.optim.Adam(model.parameters(), lr=section['clft_lr'])
        momentum = section.get('lr_momentum')  # the learning rate is multiplied by it every epoch
        scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: momentum ** epoch)
                     if momentum is not None else None)
        return TrainingSetup(optimizer, scheduler, _class_weighted_cross_entropy(config, device))

    if backbone == 'swin_fusion':
        optimizer = torch.optim.AdamW(model.parameters(), lr=section['clft_lr'],
                                      weight_decay=section.get('weight_decay', 0.05))
        warmup = section.get('warmup_epochs', 10)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
        return TrainingSetup(optimizer, _warmup_then(optimizer, warmup, 0.01, cosine),
                             _class_weighted_cross_entropy(config, device), clip_grad_norm=1.0)

    classes = calculate_num_classes(config)
    if backbone == 'maskformer':
        from models.maskformer_fusion import MaskFormerCriterion
        criterion = MaskFormerCriterion(num_classes=classes, no_object_coef=section.get('eos_coef', 0.1)).to(device)
        optimizer = _backbone_slower(model, section['clft_lr'])
        loss = lambda outputs, segmap, labels: criterion(outputs[2], outputs[3], labels, aux_outputs=outputs[4])  # noqa: E731
        return TrainingSetup(optimizer, _query_model_schedule(optimizer, section, epochs), loss, clip_grad_norm=1.0)

    if backbone == 'mask2former':
        from models.mask2former_fusion import Mask2FormerCriterion
        criterion = Mask2FormerCriterion(num_classes=classes, no_object_coef=section.get('eos_coef', 0.1),
                                         aux_weight=section.get('aux_weight', 1.0)).to(device)
        optimizer = _backbone_slower(model, section['clft_lr'])
        loss = lambda outputs, segmap, labels: criterion(outputs[2], outputs[3], labels)  # noqa: E731
        # 0.01, as in the original Mask2Former Swin configs
        return TrainingSetup(optimizer, _query_model_schedule(optimizer, section, epochs), loss, clip_grad_norm=0.01)

    if backbone == 'deeplabv3plus':
        optimizer = torch.optim.Adam(model.parameters(), lr=section.get('learning_rate', 1e-4))
        return TrainingSetup(optimizer, _deeplab_schedule(optimizer, section, epochs),
                             _class_weighted_cross_entropy(config, device), mixed_precision=False)

    raise ValueError(f"no training setup for CLI.backbone {backbone!r}")
