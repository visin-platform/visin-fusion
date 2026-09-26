# Config reference

Generated from `utils/config_schema.py` by `tools/make_config_reference.py`; do not edit.

Unknown keys in these sections are errors. Model sections (`CLFT`, `SwinFusion`, `MaskFormer`, `Mask2Former`, `DeepLabV3Plus`) take the settings of their model; start from its preset in `configs/presets/`.

| `CLI.backbone` | Model section | Modes |
| --- | --- | --- |
| `clft` | `CLFT` | `rgb`, `lidar`, `cross_fusion` |
| `swin_fusion` | `SwinFusion` | `rgb`, `lidar`, `cross_fusion` |
| `maskformer` | `MaskFormer` | `rgb`, `lidar`, `cross_fusion` |
| `mask2former` | `Mask2Former` | `rgb`, `lidar`, `cross_fusion` |
| `deeplabv3plus` | `DeepLabV3Plus` | `rgb`, `lidar`, `fusion` |

## Top level

The sections of a config and its name.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `Summary` | str | `""` | Run name, e.g. in Visin |
| `description` | str | `""` | What this config or preset is for |
| `tags` | list of str | `[]` |  |
| `CLI` | section | **required** |  |
| `General` | section | **required** |  |
| `Log` | section | **required** |  |
| `Dataset` | section | **required** |  |
| `CLFT` | dict (optional) |  |  |
| `SwinFusion` | dict (optional) |  |  |
| `MaskFormer` | dict (optional) |  |  |
| `Mask2Former` | dict (optional) |  |  |
| `DeepLabV3Plus` | dict (optional) |  |  |

## CLI

The model and its inputs.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `backbone` | `clft`, `swin_fusion`, `maskformer`, `mask2former`, `deeplabv3plus` | **required** | The model |
| `mode` | `rgb`, `lidar`, `fusion`, `cross_fusion` | **required** | Inputs: camera (rgb), LiDAR (lidar) or both (fusion; cross_fusion is the same) |

## General

Training settings.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | str | `"cuda:0"` | Torch device; the CPU is used when CUDA is not available |
| `epochs` | int | **required** |  |
| `batch_size` | int | **required** |  |
| `seed` | int | `0` |  |
| `resume_training` | bool | `False` | Continue from the latest checkpoint in Log.logdir |
| `reset_lr` | bool | `False` | When resuming, start the learning-rate schedule again |
| `early_stop_patience` | int | **required** | Epochs without a better validation mIoU before stopping |
| `max_checkpoints` | int | **required** | Best checkpoints kept |
| `model_path` | str | `""` | Checkpoint to start from; empty for the latest in Log.logdir |
| `transfer_learning` | bool | `False` | Start from model_path trained on other classes |
| `source_classes` | list (optional) |  | The classes model_path was trained on |
| `create_new_training` | bool | `False` | Start a new Visin run even when resuming |

## Log

Where a run writes.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `logdir` | str | **required** | Epoch logs, checkpoints, test results, visualizations and benchmarks |

## Dataset

The data, and the classes to learn. With a `dataset.json` at `dataset_root`, most of this comes from the manifest.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | **required** |  |
| `dataset_root` | str | **required** | Frames in the split files are relative to this; may use environment variables, e.g. "$DATA_ROOT/zod" |
| `train_split` | str | **required** |  |
| `val_split` | str | **required** |  |
| `split_dir` | str (optional) |  | Where test and visualization splits are (utils/splits.py) |
| `test_splits` | dict of str, str (optional) |  | Test sets: name -> split file |
| `visualization_split` | str (optional) |  |  |
| `annotation_path` | str (optional) |  | Annotation folder that replaces a frame's camera folder |
| `layout` | dict of str, str (optional) |  | Frame folders, e.g. {'camera': 'camera', 'lidar': 'lidar_png'} |
| `dataset_classes` | list of section | **required** |  |
| `train_classes` | list of section | **required** |  |
| `transforms` | section | **required** |  |

## Dataset.transforms

Input size, augmentation and normalization.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `resize` | int | **required** | Square input size |
| `random_rotate_range` | float | `0` | Training rotation range, degrees |
| `p_flip` | float | `0` |  |
| `p_crop` | float | `0` |  |
| `p_rot` | float | `0` |  |
| `image_mean` | list of float | **required** |  |
| `image_std` | list of float | **required** |  |
| `lidar_mean` | list of float (optional) |  |  |
| `lidar_std` | list of float (optional) |  |  |

## Dataset.dataset_classes[]

The dataset's label values.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | **required** |  |
| `index` | int | **required** | Label value in the annotation images |

## Dataset.train_classes[]

The classes the model learns.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | **required** |  |
| `index` | int | **required** | Model output channel; 0..n-1 without gaps, 0 is background |
| `weight` | float | `1.0` | Loss weight |
| `dataset_mapping` | list of int | **required** | Dataset class indices merged into this class |
| `color` | list of int | **required** | RGB color in visualizations |
