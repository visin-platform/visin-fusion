"""Uniform public model API. Modes are chosen once, at construction."""

from pathlib import Path
from typing import ClassVar

import torch
from torch import nn


class FusionModel(nn.Module):
    modes = ("rgb", "lidar", "cross_fusion")
    weights: ClassVar[dict] = {}

    def __init__(self, implementation, mode, training_options=None):
        super().__init__()
        if mode not in self.modes:
            raise ValueError(f"mode {mode!r} must be one of {self.modes}")
        self.implementation = implementation
        self.mode = mode
        self.training_options = training_options or {}

    def forward(self, rgb, lidar):
        """Return dense segmentation logits with shape [B, C, H, W]."""
        return self.segment_from_raw(self.raw_forward(rgb, lidar))

    def segment_from_raw(self, output):
        return output[1]

    def raw_forward(self, rgb, lidar):
        return self.implementation(rgb, lidar, modal=self.mode)

    def state_dict(self, *args, **kwargs):
        # Keep the pre-refactor checkpoint key names.
        return self.implementation.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.implementation.load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_pretrained(cls, name, *, map_location="cpu", **kwargs):
        """Load a named or local checkpoint. Publish a name with register_pretrained."""
        source = cls.weights.get(name, name)
        if str(source).startswith(("https://", "http://")):
            checkpoint = torch.hub.load_state_dict_from_url(source, map_location=map_location)
        elif Path(source).is_file():
            checkpoint = torch.load(source, map_location=map_location, weights_only=False)
        else:
            raise ValueError(f"Unknown weights {name!r} for {cls.__name__}; register a URL or pass a checkpoint path")
        model = cls(**kwargs, pretrained=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return model

    @classmethod
    def register_pretrained(cls, name, source):
        cls.weights = {**cls.weights, name: source}


class CLFT(FusionModel):
    def __init__(
        self,
        num_classes=4,
        image_size=384,
        mode="cross_fusion",
        patch_size=16,
        emb_dim=768,
        resample_dim=256,
        read="projection",
        hooks=(2, 5, 8, 11),
        reassembles=(4, 8, 16, 32),
        model_timm="vit_base_patch16_384",
        pretrained=False,
        training_options=None,
    ):
        from .clft.clft import CLFT as Implementation

        impl = Implementation(
            RGB_tensor_size=(3, image_size, image_size),
            XYZ_tensor_size=(3, image_size, image_size),
            patch_size=patch_size,
            emb_dim=emb_dim,
            resample_dim=resample_dim,
            read=read,
            hooks=hooks,
            reassemble_s=reassembles,
            nclasses=num_classes,
            type="segmentation",
            model_timm=model_timm,
            pretrained=pretrained,
        )
        super().__init__(impl, mode, training_options)

    def training_setup(self, class_weights, device="cpu", epochs=100, **options):
        from .training import TrainingSetup, weighted_cross_entropy

        opts = {**self.training_options, **options}
        optimizer = torch.optim.Adam(self.parameters(), lr=opts.get("clft_lr", 8e-5))
        momentum = opts.get("lr_momentum", 0.99)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: momentum**epoch)
        return TrainingSetup(optimizer, scheduler, weighted_cross_entropy(class_weights, device))


class CLFTv2(FusionModel):
    def __init__(
        self,
        num_classes=4,
        mode="cross_fusion",
        emb_dims=None,
        resample_dim=256,
        read="ignore",
        reassembles=(4, 8, 16, 32),
        model_timm="swinv2_tiny_window16_256",
        fusion_strategy="residual_average",
        pretrained=False,
        training_options=None,
    ):
        from .clftv2 import CLFTv2Network

        impl = CLFTv2Network(
            emb_dims=emb_dims,
            resample_dim=resample_dim,
            read=read,
            reassemble_s=reassembles,
            nclasses=num_classes,
            type="segmentation",
            model_timm=model_timm,
            pretrained=pretrained,
            fusion_strategy=fusion_strategy,
        )
        super().__init__(impl, mode, training_options)

    def training_setup(self, class_weights, device="cpu", epochs=100, **options):
        from .training import TrainingSetup, warmup_then, weighted_cross_entropy

        opts = {**self.training_options, **options}
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=opts.get("clft_lr", 8e-5), weight_decay=opts.get("weight_decay", 0.05)
        )
        warmup = opts.get("warmup_epochs", 10)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
        return TrainingSetup(
            optimizer,
            warmup_then(optimizer, warmup, 0.01, cosine),
            weighted_cross_entropy(class_weights, device),
            clip_grad_norm=1.0,
        )


class MaskFormerFusion(FusionModel):
    def __init__(
        self,
        num_classes=4,
        mode="cross_fusion",
        backbone="swinv2_tiny_window16_256",
        pixel_decoder_channels=256,
        transformer_d_model=256,
        num_queries=100,
        pretrained=False,
        training_options=None,
    ):
        from .maskformer_fusion import MaskFormerFusion as Implementation

        impl = Implementation(
            backbone=backbone,
            num_classes=num_classes,
            pixel_decoder_channels=pixel_decoder_channels,
            transformer_d_model=transformer_d_model,
            num_queries=num_queries,
            pretrained=pretrained,
        )
        super().__init__(impl, mode, training_options)
        self.num_classes = num_classes

    @property
    def backbone(self):
        return self.implementation.backbone

    def training_setup(self, class_weights=None, device="cpu", epochs=100, **options):
        from .maskformer_fusion import MaskFormerCriterion
        from .training import TrainingSetup, backbone_slower, query_schedule

        opts = {**self.training_options, **options}
        criterion = MaskFormerCriterion(self.num_classes, no_object_coef=opts.get("eos_coef", 0.1)).to(device)
        optimizer = backbone_slower(self, opts.get("clft_lr", 8e-5))

        def loss(outputs, segmap, labels):
            return criterion(outputs[2], outputs[3], labels, aux_outputs=outputs[4])

        return TrainingSetup(optimizer, query_schedule(optimizer, opts, epochs), loss, clip_grad_norm=1.0)


class Mask2FormerFusion(FusionModel):
    def __init__(
        self,
        num_classes=4,
        mode="cross_fusion",
        backbone="swinv2_tiny_window16_256",
        pixel_decoder_channels=256,
        transformer_d_model=256,
        num_queries=100,
        num_decoder_layers=9,
        n_encoder_layers=6,
        pretrained=False,
        training_options=None,
    ):
        from .mask2former_fusion import Mask2FormerFusion as Implementation

        impl = Implementation(
            backbone=backbone,
            num_classes=num_classes,
            pixel_decoder_channels=pixel_decoder_channels,
            transformer_d_model=transformer_d_model,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            n_encoder_layers=n_encoder_layers,
            pretrained=pretrained,
        )
        super().__init__(impl, mode, training_options)
        self.num_classes = num_classes

    @property
    def backbone(self):
        return self.implementation.backbone

    def training_setup(self, class_weights=None, device="cpu", epochs=100, **options):
        from .mask2former_fusion import Mask2FormerCriterion
        from .training import TrainingSetup, backbone_slower, query_schedule

        opts = {**self.training_options, **options}
        criterion = Mask2FormerCriterion(
            self.num_classes, no_object_coef=opts.get("eos_coef", 0.1), aux_weight=opts.get("aux_weight", 1.0)
        ).to(device)
        optimizer = backbone_slower(self, opts.get("clft_lr", 8e-5))

        def loss(outputs, segmap, labels):
            return criterion(outputs[2], outputs[3], labels)

        return TrainingSetup(optimizer, query_schedule(optimizer, opts, epochs), loss, clip_grad_norm=0.01)


class DeepLabV3Plus(FusionModel):
    modes = ("rgb", "lidar", "fusion")

    def __init__(
        self,
        num_classes=4,
        mode="fusion",
        backbone="resnet101",
        fusion_strategy="residual_average",
        pretrained=False,
        training_options=None,
    ):
        from .deeplabv3plus import build_deeplabv3plus

        impl = build_deeplabv3plus(
            num_classes, mode=mode, backbone=backbone, fusion_strategy=fusion_strategy, pretrained=pretrained
        )
        super().__init__(impl, mode, training_options)

    def segment_from_raw(self, output):
        return output[0]

    def raw_forward(self, rgb, lidar):
        if self.mode == "fusion":
            return self.implementation(rgb, lidar)
        return (self.implementation(rgb if self.mode == "rgb" else lidar),)

    def training_setup(self, class_weights, device="cpu", epochs=100, **options):
        from .training import TrainingSetup, deeplab_schedule, weighted_cross_entropy

        opts = {**self.training_options, **options}
        optimizer = torch.optim.Adam(self.parameters(), lr=opts.get("learning_rate", 1e-4))
        return TrainingSetup(
            optimizer,
            deeplab_schedule(optimizer, opts, epochs),
            weighted_cross_entropy(class_weights, device),
            mixed_precision=False,
        )
