"""What a run's config may contain, checked when it is loaded (visin_fusion/config/config.py).

The shared sections (CLI, General, Log, Dataset) are strict: an unknown key, usually a typo, is an
error before any training starts. Model sections (CLFTv2, CLFT, ...) are free-form until each
model declares its own options (TODO.md, section 5).

    visin-fusion schema > config.schema.json   # JSON Schema, e.g. for a UI's forms
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = "2.0"

# CLI.backbone -> its model section and the modes it supports. register_backbone() adds a model
# (visin_fusion.models.registry.register_model is the public way).
BACKBONES: dict[str, tuple[str, tuple[str, ...]]] = {
    "clft": ("CLFT", ("rgb", "lidar", "cross_fusion")),
    "clftv2": ("CLFTv2", ("rgb", "lidar", "cross_fusion")),
    "maskformer": ("MaskFormer", ("rgb", "lidar", "cross_fusion")),
    "mask2former": ("Mask2Former", ("rgb", "lidar", "cross_fusion")),
    "deeplabv3plus": ("DeepLabV3Plus", ("rgb", "lidar", "fusion")),
}


BUILTIN_BACKBONES = dict(BACKBONES)  # the models that ship; documentation lists these, not registered extras
RESERVED_KEYS = {"Summary", "description", "tags", "plugins", "CLI", "General", "Log", "Dataset"}


def register_backbone(name: str, section: str, modes: tuple[str, ...]) -> None:
    """Make ``CLI.backbone: name`` valid, with its settings in the config section ``section``."""
    if name in BACKBONES and BACKBONES[name] != (section, modes):
        raise ValueError(f"model {name!r} is already registered with different settings: {BACKBONES[name]}")
    taken = {s for n, (s, _) in BACKBONES.items() if n != name} | RESERVED_KEYS
    if section in taken:
        raise ValueError(f"config section {section!r} is already used")
    BACKBONES[name] = (section, tuple(modes))


class Strict(BaseModel):
    """Base of the schema's sections: an unknown key is an error, which catches typos before a run starts."""

    model_config = ConfigDict(extra="forbid")


class CLI(Strict):
    """The model and its inputs."""

    backbone: str = Field(
        description="The model: clft, clftv2, maskformer, mask2former, deeplabv3plus, or one added with register_model"
    )
    mode: Literal["rgb", "lidar", "fusion", "cross_fusion"] = Field(
        description="Inputs: camera (rgb), LiDAR (lidar) or both (fusion; cross_fusion is the same)"
    )

    @field_validator("backbone")
    @classmethod
    def backbone_is_registered(cls, value):
        """Reject a model name that is not built in or registered, listing the choices."""
        if value not in BACKBONES:
            raise ValueError(
                f"{value!r} is not a known model; choose from {sorted(BACKBONES)}. "
                "A model added with register_model must be imported in every stage: name its module in `plugins`"
            )
        return value

    @model_validator(mode="after")
    def mode_fits_backbone(self):
        """Reject a mode the chosen model does not have."""
        modes = BACKBONES[self.backbone][1]
        if self.mode not in modes:
            raise ValueError(f"mode {self.mode!r} is not one of {self.backbone}'s modes {list(modes)}")
        return self


class General(Strict):
    """Training settings."""

    device: str = Field("cuda:0", description="Torch device; the CPU is used when CUDA is not available")
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    seed: int = 0
    accumulate_batches: int = Field(
        1,
        ge=1,
        description="Batches whose gradients are averaged per optimizer step: effective batch size = batch_size x this",
    )
    num_workers: int | None = Field(
        None,
        ge=0,
        description="DataLoader worker processes; 0 loads in the main process. "
        "Default: CPU count, at most 8 and at most one per batch",
    )
    resume_training: bool = Field(False, description="Continue from the latest checkpoint in Log.logdir")
    reset_lr: bool = Field(False, description="When resuming, start the learning-rate schedule again")
    early_stop_patience: int = Field(ge=1, description="Epochs without a better validation mIoU before stopping")
    max_checkpoints: int = Field(ge=1, description="Best checkpoints kept")
    model_path: str = Field("", description="Checkpoint to start from; empty for the latest in Log.logdir")
    transfer_learning: bool = Field(False, description="Start from model_path trained on other classes")
    source_classes: list | None = Field(None, description="The classes model_path was trained on")
    callbacks: list[str] = Field(
        default_factory=list,
        description='Callback classes to load in every stage, as "package.module:ClassName", built with the config',
    )
    create_new_training: bool = Field(False, description="Start a new Visin run even when resuming")
    hub_repo: str | None = Field(
        None,
        description="Hugging Face model repo (org/name) the best checkpoint is published to when training ends; "
        "empty publishes nothing. Needs the hf extra and your own HF_TOKEN",
    )
    hub_private: bool = Field(True, description="Create hub_repo as a private repo; false makes it public")
    suite: str | None = Field(
        None,
        pattern=r"^[a-z0-9][a-z0-9-]*@[1-9][0-9]*$",
        description="Visin suite version, slug@version, the test stage records its results on as an evaluation "
        "of the tested checkpoint; empty records none. Needs a Visin that has suites, and a visin package "
        "that has evaluate",
    )
    suite_file: str | None = Field(
        None,
        description="The Visin suite file (JSON, or YAML with visin[yaml]) the test stage scores against. Its digest "
        "is sent as the protocol that ran, together with the digest of the test sets' frame lists, which marks the "
        "evaluation observed rather than reported; its slug@version is used when suite is empty",
    )


class Log(Strict):
    """Where a run writes."""

    logdir: str = Field(description="Epoch logs, checkpoints, test results, visualizations and benchmarks")


class DatasetClass(Strict):
    """One label value of the dataset's annotation images."""

    name: str
    index: int = Field(ge=0, description="Label value in the annotation images")


class TrainClass(Strict):
    """One class the model learns, and the dataset values merged into it."""

    name: str
    index: int = Field(ge=0, description="Model output channel; 0..n-1 without gaps, 0 is background")
    weight: float = Field(1.0, gt=0, description="Loss weight")
    dataset_mapping: list[int] = Field(description="Dataset class indices merged into this class")
    color: list[int] = Field(min_length=3, max_length=3, description="RGB color in visualizations")


Triple = list[float]


class Transforms(Strict):
    """Input size, augmentation and normalization."""

    resize: int = Field(ge=1, description="Square input size")
    random_rotate_range: float = Field(0, ge=0, description="Training rotation range, degrees")
    p_flip: float = Field(0, ge=0, le=1)
    p_crop: float = Field(0, ge=0, le=1)
    p_rot: float = Field(0, ge=0, le=1)
    image_mean: Triple = Field(min_length=3, max_length=3)
    image_std: Triple = Field(min_length=3, max_length=3)
    lidar_mean: Triple | None = Field(None, min_length=3, max_length=3)
    lidar_std: Triple | None = Field(None, min_length=3, max_length=3)


class Dataset(Strict):
    """The data and the classes to learn; most of it comes from ``dataset.json`` when there is one."""

    name: str
    dataset_root: str = Field(
        description="Frames in the split files are relative to this; "
        'may use environment variables, e.g. "$DATA_ROOT/zod"'
    )
    train_split: str
    val_split: str
    split_dir: str | None = Field(
        None, description="Where test and visualization splits are (visin_fusion/config/splits.py)"
    )
    test_splits: dict[str, str] | None = Field(None, description="Test sets: name -> split file")
    visualization_split: str | None = None
    annotation_path: str | None = Field(None, description="Annotation folder that replaces a frame's camera folder")
    layout: dict[str, str] | None = Field(
        None, description="Frame folders, e.g. {'camera': 'camera', 'lidar': 'lidar_png'}"
    )
    dataset_classes: list[DatasetClass] = Field(min_length=1)
    train_classes: list[TrainClass] = Field(min_length=1)
    transforms: Transforms

    @model_validator(mode="after")
    def classes_fit(self):
        """Check the training classes are numbered 0..n-1 and only map values the dataset has."""
        indices = sorted(c.index for c in self.train_classes)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"train_classes indices must be 0..{len(indices) - 1} with no gaps or duplicates, got {indices}"
            )
        known = {c.index for c in self.dataset_classes}
        for train_class in self.train_classes:
            unknown = sorted(set(train_class.dataset_mapping) - known)
            if unknown:
                raise ValueError(
                    f"train class {train_class.name!r} maps dataset classes {unknown}, "
                    f"which dataset_classes does not have"
                )
        return self


class Config(BaseModel):
    """A run's config. Model sections are open: the five built-in models' are declared, a registered model's
    is accepted under the section name it registered; any other top-level key is an error."""

    model_config = ConfigDict(extra="allow")

    Summary: str = Field("", description="Run name, e.g. in Visin")
    description: str = Field("", description="What this config or preset is for")
    tags: list[str] = []
    plugins: list[str] = Field(
        default_factory=list,
        description="Modules to import before the config is checked, so they can register models (register_model)",
    )
    CLI: CLI
    General: General
    Log: Log
    Dataset: Dataset
    CLFT: dict | None = None
    CLFTv2: dict | None = None
    MaskFormer: dict | None = None
    Mask2Former: dict | None = None
    DeepLabV3Plus: dict | None = None

    @model_validator(mode="after")
    def has_model_section(self):
        """Reject unknown top-level keys and require the chosen model's section."""
        sections = {section for section, _ in BACKBONES.values()}
        unknown = sorted(set(self.model_extra or {}) - sections)
        if unknown:
            raise ValueError(f"Extra inputs are not permitted: {unknown}")
        section = BACKBONES[self.CLI.backbone][0]
        value = (
            getattr(self, section, None)
            if section in type(self).model_fields
            else (self.model_extra or {}).get(section)
        )
        if value is None:
            raise ValueError(f"CLI.backbone {self.CLI.backbone!r} needs a {section!r} section")
        if not isinstance(value, dict):
            raise ValueError(f"{section!r} must be an object of the model's settings")
        return self


class ConfigError(ValueError):
    """A config that does not match the schema; the message lists every problem."""


def validate_config(config):
    """Raise ConfigError naming every problem in the config, or return it unchanged."""
    try:
        Config.model_validate(config)
    except ValidationError as e:
        problems = "\n".join(f"  {'.'.join(map(str, err['loc'])) or '(config)'}: {err['msg']}" for err in e.errors())
        raise ConfigError(f"Invalid config:\n{problems}") from None
    return config


def export_schema():
    """The config's JSON Schema, with the registered models listed as the allowed ``CLI.backbone`` values."""
    schema = Config.model_json_schema()
    schema["$defs"]["CLI"]["properties"]["backbone"]["enum"] = list(BACKBONES)
    schema["$id"] = f"https://visin.eu/schemas/visin-fusion/{SCHEMA_VERSION}/config.schema.json"
    schema["x-schema-version"] = SCHEMA_VERSION
    return schema


if __name__ == "__main__":
    print(json.dumps(export_schema(), indent=2))
