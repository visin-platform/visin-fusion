"""Model names and the config adapter used by the command line pipeline.

A model joins the pipeline by registering once; no stage knows any model by name:

    register_model("mynet", MyNet, section="MyNet")

``MyNet`` subclasses :class:`~visin_fusion.models.api.FusionModel`. Its config section (``"MyNet": {...}``)
is passed to its constructor as keyword arguments, next to ``num_classes``, ``mode`` and ``pretrained``;
keys it does not take are ignored. Stages run as separate processes, so the module that registers must be
named in the config's ``plugins`` list to be imported in each of them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence

from visin_fusion.config.config_schema import register_backbone

from .api import CLFT, CLFTv2, DeepLabV3Plus, FusionModel, Mask2FormerFusion, MaskFormerFusion

MODEL_CLASSES: dict[str, type[FusionModel]] = {}
SECTIONS: dict[str, str] = {}
OPTIONS: dict[str, Callable[[Mapping, Mapping], dict]] = {}


def register_model(
    name: str,
    cls: type[FusionModel],
    *,
    section: str,
    modes: Sequence[str] | None = None,
    options: Callable[[Mapping, Mapping], dict] | None = None,
) -> None:
    """Make ``CLI.backbone: name`` build ``cls`` from the config section ``section``.

    ``modes`` defaults to ``cls.modes``. ``options(config, section_settings)`` may return more constructor
    arguments derived from the whole config (for example the input size from ``Dataset.transforms``).
    Registering the same name twice with the same arguments is allowed, so a module can be imported twice.
    """
    if not (isinstance(cls, type) and issubclass(cls, FusionModel)):
        raise TypeError(f"{cls!r} must be a subclass of visin_fusion.models.FusionModel")
    register_backbone(name, section, tuple(modes or cls.modes))
    MODEL_CLASSES[name] = cls
    SECTIONS[name] = section
    OPTIONS[name] = options or (lambda config, settings: {})


def model_section(config):
    """The settings of the config's model, from its own section."""
    return config[SECTIONS[config["CLI"]["backbone"]]]


def from_config(config, pretrained=None):
    """Construct a public model from a validated pipeline config.

    The model's section is passed to its constructor; only arguments the constructor takes are forwarded,
    and the whole section is kept as the model's ``training_options``.
    """
    name = config["CLI"]["backbone"]
    cls = MODEL_CLASSES[name]
    section = model_section(config)
    options = dict(section)
    options["num_classes"] = len(config["Dataset"]["train_classes"])
    options["mode"] = config["CLI"]["mode"]
    options["training_options"] = dict(section)
    if pretrained is not None:
        options["pretrained"] = pretrained
    options.update(OPTIONS[name](config, section))
    names = inspect.signature(cls).parameters
    return cls(**{key: value for key, value in options.items() if key in names})


def training_setup(config, model, device):
    """The model's optimizer, schedule and loss, with the config's class weights and epochs."""
    classes = sorted(config["Dataset"]["train_classes"], key=lambda c: c["index"])
    weights = [item["weight"] for item in classes]
    return model.training_setup(weights, device=device, epochs=config["General"]["epochs"])


def _clft_options(config, section):
    return {"image_size": config["Dataset"]["transforms"]["resize"], "reassembles": section["reassembles"]}


def _clftv2_options(config, section):
    return {"reassembles": section["reassembles"]}


def _query_model_options(config, section):
    return {"backbone": section["model_timm"]}


def _deeplab_options(config, section):
    return {"fusion_strategy": section.get("fusion_strategy") or section.get("fusion_type") or "residual_average"}


register_model("clft", CLFT, section="CLFT", options=_clft_options)
register_model("clftv2", CLFTv2, section="CLFTv2", options=_clftv2_options)
register_model("maskformer", MaskFormerFusion, section="MaskFormer", options=_query_model_options)
register_model("mask2former", Mask2FormerFusion, section="Mask2Former", options=_query_model_options)
register_model("deeplabv3plus", DeepLabV3Plus, section="DeepLabV3Plus", options=_deeplab_options)
