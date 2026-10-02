"""Uniform public model API. Modes are chosen once, at construction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar

import torch
from torch import Tensor, nn

from .clft.clft import CLFT as CLFTNetwork
from .clftv2 import CLFTv2Network
from .deeplabv3plus import build_deeplabv3plus
from .mask2former_fusion import Mask2FormerCriterion
from .mask2former_fusion import Mask2FormerFusion as Mask2FormerNetwork
from .maskformer_fusion import MaskFormerCriterion
from .maskformer_fusion import MaskFormerFusion as MaskFormerNetwork
from .training import (
    TrainingSetup,
    backbone_slower,
    deeplab_schedule,
    query_schedule,
    warmup_then,
    weighted_cross_entropy,
)


class FusionModel(nn.Module):
    """Base of the five public models: one construction-time mode, one call, dense logits.

    ``mode`` is ``"rgb"``, ``"lidar"`` or the two-stream mode, spelled ``"cross_fusion"`` or ``"fusion"``;
    either spelling is accepted by every model and normalized to the model's own (``self.mode``).
    ``forward(rgb, lidar)`` takes ``[B, 3, H, W]`` tensors. A single-stream model needs only its own input;
    the other may be omitted or ``None`` and is ignored.
    """

    modes = ("rgb", "lidar", "cross_fusion")
    weights: ClassVar[dict] = {}

    def __init__(self, implementation: nn.Module, mode: str, training_options: dict | None = None) -> None:
        super().__init__()
        mode = self.canonical_mode(mode)
        self.implementation = implementation
        self.mode = mode
        self.training_options = training_options or {}

    @classmethod
    def canonical_mode(cls, mode: str) -> str:
        """The model's own spelling of ``mode``; ``ValueError`` listing the choices if it is none."""
        if mode in ("fusion", "cross_fusion"):
            mode = next((m for m in cls.modes if m.endswith("fusion")), mode)
        if mode not in cls.modes:
            raise ValueError(f"mode {mode!r} must be one of {cls.modes}")
        return mode

    def forward(self, rgb: Tensor | None = None, lidar: Tensor | None = None) -> Tensor:
        """Return dense segmentation logits with shape [B, C, H, W]."""
        return self.segment_from_raw(self.raw_forward(rgb, lidar))

    def segment_from_raw(self, output):
        """The dense logits inside the architecture-specific ``raw_forward`` output.

        A plain tensor is the logits; a tuple's second element is (the layout of the built-in
        transformer models). Override for another layout.
        """
        return output if isinstance(output, torch.Tensor) else output[1]

    def training_setup(self, class_weights, device="cpu", epochs=100, **options):
        """Default training: Adam at ``learning_rate`` (1e-4) and class-weighted cross-entropy, no schedule.

        The built-in models override this with their own recipes; a model added with ``register_model``
        gets this unless it does the same.
        """

        opts = {**self.training_options, **options}
        optimizer = torch.optim.Adam(self.parameters(), lr=opts.get("learning_rate", 1e-4))
        return TrainingSetup(optimizer, None, weighted_cross_entropy(class_weights, device), mixed_precision=False)

    def raw_forward(self, rgb: Tensor | None = None, lidar: Tensor | None = None):
        """The architecture's own outputs (query models return mask and class predictions too)."""
        rgb, lidar = self._inputs(rgb, lidar)
        return self.implementation(rgb, lidar, modal=self.mode)

    def _inputs(self, rgb: Tensor | None, lidar: Tensor | None) -> tuple[Tensor, Tensor]:
        """Check the streams this mode reads; the unused one stands in as the other so no model sees ``None``."""
        required = {"rgb": ("rgb",), "lidar": ("lidar",)}.get(self.mode, ("rgb", "lidar"))
        streams = {"rgb": rgb, "lidar": lidar}
        missing = [name for name in required if streams[name] is None]
        if missing:
            raise ValueError(f"mode {self.mode!r} needs {' and '.join(required)}; {' and '.join(missing)} is None")
        for name, tensor in streams.items():
            if tensor is not None and (tensor.ndim != 4 or tensor.shape[1] != 3):
                raise ValueError(f"{name} must be shaped [batch, 3, height, width], got {tuple(tensor.shape)}")
        if rgb is not None and lidar is not None and rgb.shape != lidar.shape:
            raise ValueError(f"rgb {tuple(rgb.shape)} and lidar {tuple(lidar.shape)} must have the same shape")
        primary = rgb if rgb is not None else lidar
        if primary is None:
            raise ValueError("give rgb or lidar")
        return (rgb if rgb is not None else primary), (lidar if lidar is not None else primary)

    def state_dict(self, *args, **kwargs):
        """The architecture's weights under the checkpoint key names of earlier releases."""
        return self.implementation.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load weights saved by ``state_dict``; returns torch's missing/unexpected keys result."""
        return self.implementation.load_state_dict(state_dict, strict=strict, assign=assign)

    @classmethod
    def from_pretrained(cls, name: str, *, map_location="cpu", **kwargs):
        """Load a named or local checkpoint. Publish a name with register_pretrained.

        Checkpoints are read with ``weights_only=True``: a state dict, or a dict with ``model_state_dict``.
        A file that holds anything else is refused; for one you trust, load it yourself and call
        ``load_state_dict``.
        """
        source = cls.weights.get(name, name)
        if str(source).startswith(("https://", "http://")):
            checkpoint = torch.hub.load_state_dict_from_url(source, map_location=map_location, weights_only=True)
        elif Path(source).is_file():
            try:
                checkpoint = torch.load(source, map_location=map_location, weights_only=True)
            except pickle.UnpicklingError as exc:
                raise ValueError(
                    f"{source} holds more than weights; load it yourself with torch.load and call load_state_dict"
                ) from exc
        else:
            raise ValueError(f"Unknown weights {name!r} for {cls.__name__}; register a URL or pass a checkpoint path")
        model = cls(**{**kwargs, "pretrained": False})
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return model

    @classmethod
    def register_pretrained(cls, name: str, source: str) -> None:
        """Make ``from_pretrained(name)`` load ``source``, a URL or a file path, for this class."""
        cls.weights = {**cls.weights, name: source}


class CLFT(FusionModel):
    """CLFT: a ViT shared by both streams, with global features fused while decoding."""

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

        impl = CLFTNetwork(
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
        """Adam with a per-epoch exponential learning-rate decay (``lr_momentum``) and class-weighted cross-entropy."""

        opts = {**self.training_options, **options}
        optimizer = torch.optim.Adam(self.parameters(), lr=opts.get("clft_lr", 8e-5))
        momentum = opts.get("lr_momentum", 0.99)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: momentum**epoch)
        return TrainingSetup(optimizer, scheduler, weighted_cross_entropy(class_weights, device))


class CLFTv2(FusionModel):
    """CLFTv2: a hierarchical Swin encoder with a fusion pyramid."""

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
        """AdamW with cosine decay after warmup, class-weighted cross-entropy and gradient clipping at 1.0."""

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
    """MaskFormer with a fusion encoder: class-labelled mask queries over a pixel decoder."""

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

        impl = MaskFormerNetwork(
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
        """The timm backbone, for learning-rate groups that train it more slowly."""
        return self.implementation.backbone

    def training_setup(self, class_weights=None, device="cpu", epochs=100, **options):
        """Hungarian-matched focal and dice mask losses, a slower backbone learning rate and gradient clipping."""

        opts = {**self.training_options, **options}
        criterion = MaskFormerCriterion(self.num_classes, no_object_coef=opts.get("eos_coef", 0.1)).to(device)
        optimizer = backbone_slower(self, opts.get("clft_lr", 8e-5))

        def loss(outputs, segmap, labels):
            return criterion(outputs[2], outputs[3], labels, aux_outputs=outputs[4])

        return TrainingSetup(optimizer, query_schedule(optimizer, opts, epochs), loss, clip_grad_norm=1.0)


class Mask2FormerFusion(FusionModel):
    """Mask2Former with a fusion encoder: multi-scale masked-attention queries."""

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

        impl = Mask2FormerNetwork(
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
        """The timm backbone, for learning-rate groups that train it more slowly."""
        return self.implementation.backbone

    def training_setup(self, class_weights=None, device="cpu", epochs=100, **options):
        """Hungarian-matched mask losses with auxiliary layers, a slower backbone and gradient clipping."""

        opts = {**self.training_options, **options}
        criterion = Mask2FormerCriterion(
            self.num_classes, no_object_coef=opts.get("eos_coef", 0.1), aux_weight=opts.get("aux_weight", 1.0)
        ).to(device)
        optimizer = backbone_slower(self, opts.get("clft_lr", 8e-5))

        def loss(outputs, segmap, labels):
            return criterion(outputs[2], outputs[3], labels)

        return TrainingSetup(optimizer, query_schedule(optimizer, opts, epochs), loss, clip_grad_norm=0.01)


class DeepLabV3Plus(FusionModel):
    """DeepLabV3+ with two convolutional branches and late fusion; its two-stream mode is called ``fusion``."""

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

        mode = self.canonical_mode(mode)
        impl = build_deeplabv3plus(
            num_classes, mode=mode, backbone=backbone, fusion_strategy=fusion_strategy, pretrained=pretrained
        )
        super().__init__(impl, mode, training_options)

    def segment_from_raw(self, output):
        """The dense logits: the first element of the raw output."""
        return output[0]

    def raw_forward(self, rgb=None, lidar=None):
        """Run the branches the mode uses and return the architecture's own output tuple."""
        rgb, lidar = self._inputs(rgb, lidar)
        if self.mode == "fusion":
            return self.implementation(rgb, lidar)
        return (self.implementation(rgb if self.mode == "rgb" else lidar),)

    def training_setup(self, class_weights, device="cpu", epochs=100, **options):
        """Adam with the configured ``lr_scheduler`` (if any) and class-weighted cross-entropy."""

        opts = {**self.training_options, **options}
        optimizer = torch.optim.Adam(self.parameters(), lr=opts.get("learning_rate", 1e-4))
        return TrainingSetup(
            optimizer,
            deeplab_schedule(optimizer, opts, epochs),
            weighted_cross_entropy(class_weights, device),
            mixed_precision=False,
        )
