"""What a checkpoint records about its model, so it can be rebuilt without the config that trained it.

Kept apart from ``visin_fusion.models`` on purpose: the training stage's helpers import this, and DataLoader
worker processes import those helpers, so it must not pull in the model implementations (timm, scipy).
"""

from __future__ import annotations

from collections.abc import Mapping

from visin_fusion.config.config_schema import BACKBONES

MODEL_INFO_FORMAT = 1


def model_info(config: Mapping) -> dict:
    """The model, its classes, its input preprocessing and its plugins, as plain data for a checkpoint."""
    dataset = config["Dataset"]
    backbone = config["CLI"]["backbone"]
    return {
        "format": MODEL_INFO_FORMAT,
        "backbone": backbone,
        "mode": config["CLI"]["mode"],
        "model": dict(config[BACKBONES[backbone][0]]),
        "train_classes": [dict(c) for c in dataset["train_classes"]],
        "transforms": dict(dataset["transforms"]),
        "layout": dict(dataset.get("layout") or {}),
        "dataset": dataset.get("name", ""),
        "plugins": list(config.get("plugins") or []),
    }


def config_from_info(info: Mapping) -> dict:
    """A minimal config that ``visin_fusion.models.registry.from_config`` can build the model of ``info`` from."""
    backbone = info["backbone"]
    return {
        "CLI": {"backbone": backbone, "mode": info["mode"]},
        BACKBONES[backbone][0]: dict(info["model"]),
        "Dataset": {"train_classes": info["train_classes"], "transforms": info["transforms"], "layout": info["layout"]},
    }
