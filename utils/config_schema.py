"""What a run's config may contain, checked when it is loaded (utils/config.py).

The shared sections (CLI, General, Log, Dataset) are strict: an unknown key, usually a typo, is an
error before any training starts. Model sections (SwinFusion, CLFT, ...) are free-form until each
model declares its own options (TODO.md, section 5).

    python -m utils.config_schema > config.schema.json   # JSON Schema, e.g. for a UI's forms
"""
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# CLI.backbone -> its model section and the modes it supports
BACKBONES = {
    'clft': ('CLFT', ('rgb', 'lidar', 'cross_fusion')),
    'swin_fusion': ('SwinFusion', ('rgb', 'lidar', 'cross_fusion')),
    'maskformer': ('MaskFormer', ('rgb', 'lidar', 'cross_fusion')),
    'mask2former': ('Mask2Former', ('rgb', 'lidar', 'cross_fusion')),
    'deeplabv3plus': ('DeepLabV3Plus', ('rgb', 'lidar', 'fusion')),
}


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class CLI(Strict):
    backbone: Literal['clft', 'swin_fusion', 'maskformer', 'mask2former', 'deeplabv3plus'] = Field(
        description='The model')
    mode: Literal['rgb', 'lidar', 'fusion', 'cross_fusion'] = Field(
        description="Inputs: camera (rgb), LiDAR (lidar) or both (fusion; cross_fusion is the same)")

    @model_validator(mode='after')
    def mode_fits_backbone(self):
        modes = BACKBONES[self.backbone][1]
        if self.mode not in modes:
            raise ValueError(f"mode {self.mode!r} is not one of {self.backbone}'s modes {list(modes)}")
        return self


class General(Strict):
    device: str = Field('cuda:0', description='Torch device; the CPU is used when CUDA is not available')
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    seed: int = 0
    resume_training: bool = Field(False, description='Continue from the latest checkpoint in Log.logdir')
    reset_lr: bool = Field(False, description='When resuming, start the learning-rate schedule again')
    early_stop_patience: int = Field(ge=1, description='Epochs without a better validation mIoU before stopping')
    max_checkpoints: int = Field(ge=1, description='Best checkpoints kept')
    model_path: str = Field('', description='Checkpoint to start from; empty for the latest in Log.logdir')
    transfer_learning: bool = Field(False, description='Start from model_path trained on other classes')
    source_classes: Optional[list] = Field(None, description='The classes model_path was trained on')
    create_new_training: bool = Field(False, description='Start a new Visin run even when resuming')


class Log(Strict):
    logdir: str = Field(description='Epoch logs, checkpoints, test results, visualizations and benchmarks')


class DatasetClass(Strict):
    name: str
    index: int = Field(ge=0, description='Label value in the annotation images')


class TrainClass(Strict):
    name: str
    index: int = Field(ge=0, description='Model output channel; 0..n-1 without gaps, 0 is background')
    weight: float = Field(1.0, gt=0, description='Loss weight')
    dataset_mapping: list[int] = Field(description='Dataset class indices merged into this class')
    color: list[int] = Field(min_length=3, max_length=3, description='RGB color in visualizations')


Triple = list[float]


class Transforms(Strict):
    resize: int = Field(ge=1, description='Square input size')
    random_rotate_range: float = Field(0, ge=0, description='Training rotation range, degrees')
    p_flip: float = Field(0, ge=0, le=1)
    p_crop: float = Field(0, ge=0, le=1)
    p_rot: float = Field(0, ge=0, le=1)
    image_mean: Triple = Field(min_length=3, max_length=3)
    image_std: Triple = Field(min_length=3, max_length=3)
    lidar_mean: Optional[Triple] = Field(None, min_length=3, max_length=3)
    lidar_std: Optional[Triple] = Field(None, min_length=3, max_length=3)


class Dataset(Strict):
    name: str
    dataset_root: str = Field(description='Frames in the split files are relative to this; '
                                          'may use environment variables, e.g. "$DATA_ROOT/zod"')
    train_split: str
    val_split: str
    split_dir: Optional[str] = Field(None, description='Where test and visualization splits are (utils/splits.py)')
    test_splits: Optional[dict[str, str]] = Field(None, description='Test sets: name -> split file')
    visualization_split: Optional[str] = None
    annotation_path: Optional[str] = Field(None, description="Annotation folder that replaces a frame's camera folder")
    layout: Optional[dict[str, str]] = Field(None, description="Frame folders, e.g. {'camera': 'camera', 'lidar': 'lidar_png'}")
    dataset_classes: list[DatasetClass] = Field(min_length=1)
    train_classes: list[TrainClass] = Field(min_length=1)
    transforms: Transforms

    @model_validator(mode='after')
    def classes_fit(self):
        indices = sorted(c.index for c in self.train_classes)
        if indices != list(range(len(indices))):
            raise ValueError(f'train_classes indices must be 0..{len(indices) - 1} with no gaps or duplicates, '
                             f'got {indices}')
        known = {c.index for c in self.dataset_classes}
        for train_class in self.train_classes:
            unknown = sorted(set(train_class.dataset_mapping) - known)
            if unknown:
                raise ValueError(f"train class {train_class.name!r} maps dataset classes {unknown}, "
                                 f"which dataset_classes does not have")
        return self


class Config(Strict):
    Summary: str = Field('', description='Run name, e.g. in Visin')
    description: str = Field('', description='What this config or preset is for')
    tags: list[str] = []
    CLI: CLI
    General: General
    Log: Log
    Dataset: Dataset
    CLFT: Optional[dict] = None
    SwinFusion: Optional[dict] = None
    MaskFormer: Optional[dict] = None
    Mask2Former: Optional[dict] = None
    DeepLabV3Plus: Optional[dict] = None

    @model_validator(mode='after')
    def has_model_section(self):
        section = BACKBONES[self.CLI.backbone][0]
        if getattr(self, section) is None:
            raise ValueError(f"CLI.backbone {self.CLI.backbone!r} needs a {section!r} section")
        return self


class ConfigError(ValueError):
    """A config that does not match the schema; the message lists every problem."""


def validate_config(config):
    """Raise ConfigError naming every problem in the config, or return it unchanged."""
    try:
        Config.model_validate(config)
    except ValidationError as e:
        problems = '\n'.join(f"  {'.'.join(map(str, err['loc'])) or '(config)'}: {err['msg']}" for err in e.errors())
        raise ConfigError(f'Invalid config:\n{problems}') from None
    return config


if __name__ == '__main__':
    print(json.dumps(Config.model_json_schema(), indent=2))
