# Configs

A config is a JSON file describing one run. You rarely write one from scratch: extend a model's
preset and set what is yours.

```json
{
  "extends": "clftv2",
  "Summary": "CLFTv2 on my data, camera only",
  "CLI": {"mode": "rgb"},
  "Dataset": {"dataset_root": "/data/my_dataset"},
  "General": {"epochs": 50}
}
```

## How a config is resolved

Every stage and `visin-fusion run` load a config the same way (`visin_fusion/config/config.py`):

1. **`extends`**: the preset (`visin_fusion/config/presets/<name>.json`) or file (`"./base.json"`, relative to the
   config) it names is loaded first, and the config's own keys are merged over it: dictionaries key by
   key, everything else replaced. A file it extends may extend another.
2. **Paths**: `Dataset.dataset_root` may use environment variables and `~`, e.g. `"$DATA_ROOT/zod"`,
   so the same config runs on every machine (the Docker image sets `DATA_ROOT=/data`).
3. **The dataset's manifest**: `dataset.json` at `Dataset.dataset_root` fills in the dataset's name,
   splits, classes, suggested training classes, folder layout and LiDAR normalization
   ([Datasets](datasets.md)). Keys the config sets win.
4. **Log directory**: `Log.logdir` defaults to `logs/<dataset>/<backbone>-<mode>`.
5. **Check**: the result is checked against the schema (`visin_fusion/config/config_schema.py`). A mistake stops the
   run before any work, with every problem listed:

    ```text
    Invalid config:
      CLI: Value error, mode 'fusion' is not one of clftv2's modes ['rgb', 'lidar', 'cross_fusion']
      General.epcohs: Extra inputs are not permitted
    ```

6. **Defaults**: keys still missing get the schema's defaults.

The [config reference](reference/config.md) lists every key with its type, default and meaning.

## A complete config

This config names everything itself, with no `dataset.json`: where the data and split files are, which annotations to
use and how the dataset's classes map to the model's. Keys you leave out come from the preset.

```json
{
  "extends": "clftv2",
  "Summary": "CLFTv2 on my data",
  "CLI": {"mode": "fusion"},
  "General": {"device": "cuda:0", "epochs": 50, "batch_size": 4, "early_stop_patience": 10, "max_checkpoints": 2},
  "Log": {"logdir": "logs/my_dataset"},
  "Dataset": {
    "name": "my_dataset",
    "dataset_root": "/data/my_dataset",
    "train_split": "/data/my_dataset/train.txt",
    "val_split": "/data/my_dataset/validation.txt",
    "test_splits": {"day": "test_day.txt", "night": "test_night.txt"},
    "visualization_split": "visualizations.txt",
    "annotation_path": "annotation",
    "dataset_classes": [
      {"name": "background", "index": 0},
      {"name": "car", "index": 1},
      {"name": "truck", "index": 2},
      {"name": "pedestrian", "index": 3}
    ],
    "train_classes": [
      {"name": "background", "index": 0, "weight": 0.1, "dataset_mapping": [0], "color": [0, 0, 0]},
      {"name": "vehicle", "index": 1, "weight": 10.0, "dataset_mapping": [1, 2], "color": [128, 0, 128]},
      {"name": "pedestrian", "index": 2, "weight": 20.0, "dataset_mapping": [3], "color": [0, 255, 255]}
    ],
    "transforms": {"lidar_mean": [0.25, 0.49, 0.50], "lidar_std": [0.23, 0.03, 0.17]}
  }
}
```

### Where the files are

| Key | Points at | Relative paths are resolved against |
| --- | --- | --- |
| `Dataset.dataset_root` | the folder with `camera/`, `lidar_png/` and the annotation folder | the working directory; `$VARS` and `~` are expanded |
| `Dataset.train_split`, `val_split` | the files listing training and validation frames, one per line | the working directory; use absolute paths |
| `Dataset.test_splits` | test sets as `{name: file}`; each is evaluated and reported separately | the folder of `val_split` (or `Dataset.split_dir`) |
| `Dataset.visualization_split` | the frames to render | the folder of `val_split` (or `Dataset.split_dir`) |
| `Dataset.annotation_path` | the annotation folder that replaces `camera` in a frame's path | `dataset_root` |
| `Log.logdir` | where logs, checkpoints and results are written | the working directory |

The lines of a split file are frames relative to `dataset_root`, e.g. `camera/000001.png`. If you leave out
`test_splits`, the five weather splits (`test_day_fair.txt` and so on) are used when they exist.

### Class mapping

`dataset_classes` names the pixel values in your annotation images. `train_classes` is what the model learns, and
`dataset_mapping` says which dataset values each one covers:

- **Merge**: `"dataset_mapping": [1, 2]` makes car and truck one `vehicle` class.
- **Fold into background**: put the value in background's mapping, e.g. `[0, 1]` for an "ignore" value `1`.
- **Unmapped values**: a dataset value in no mapping becomes background (`0`), including values above every listed class such as `255`.
- Every number in a `dataset_mapping` must also be listed in `dataset_classes`, or the config is rejected.
- `index` is the output channel: `0, 1, 2, ...` without gaps, and `0` is background. `weight` raises the loss for rare classes,
  `color` is the RGB used in visualizations.

With a manifest, `dataset.json` carries `classes` and `train_classes` and the config does not repeat them
([Datasets](datasets.md#the-manifest)). Set `Dataset.train_classes` in the config to override them.

## Customize an example

The ready-made [model examples](training-examples.md) run one epoch with no pretrained weights, to check the pipeline. To make a real experiment, extend one and change only what is yours. Save this as `configs/my-zod-run.json`:

```json
{
  "extends": "./examples/zod/clftv2/fusion.json",
  "Summary": "ZOD CLFTv2 fusion baseline",
  "Log": {"logdir": "logs/my-zod-baseline"},
  "General": {"epochs": 50},
  "CLFTv2": {"pretrained": true, "warmup_epochs": 5}
}
```

`extends` with a path is relative to the config file. Pretrained backbones need a download or cached weights, and each model has its own settings section (`CLFT`, `CLFTv2`, `DeepLabV3Plus`, `MaskFormer`, `Mask2Former`). To resume a run, keep its `Log.logdir`, set `General.resume_training` to `true` and raise `General.epochs` above the completed count. Use a new `Log.logdir` for every fresh experiment.

## Common recipes

Each is a few lines on top of a preset or an example config:

| I want to... | Set |
| --- | --- |
| Use the camera only, or LiDAR only | `"CLI": {"mode": "rgb"}` or `"lidar"` |
| Run on CPU | `"General": {"device": "cpu"}` (it also falls back to CPU when CUDA is missing) |
| Load data in the main process (debugging) or use fewer cores | `"General": {"num_workers": 0}` (default: the CPU count, at most 8 and at most one per batch) |
| Log to my own tools | `"General": {"callbacks": ["my_package.tracking:Tracking"]}`; see [Callbacks](running.md#callbacks) |
| Train with a larger effective batch than fits | `"General": {"batch_size": 2, "accumulate_batches": 8}` steps once per 8 batches, as a batch of 16 |
| Fit a smaller GPU | `"General": {"batch_size": 2}` |
| Train longer | `"General": {"epochs": 100, "early_stop_patience": 20}` |
| Start from pretrained weights | `"CLFTv2": {"pretrained": true}` (use the model's own section; needs a download or cached weights) |
| Keep runs apart | a new `"Log": {"logdir": "logs/run2"}` for each experiment |
| Continue an interrupted run | same `logdir`, `"General": {"resume_training": true}`, `epochs` above the finished count |
| Repeat with another seed | the train stage's `--seed 1` ([Running](running.md#one-stage)), or `General.seed` |
| Use the same dataset on every machine | `"dataset_root": "$DATA_ROOT/my_dataset"` and export `DATA_ROOT` |
| Test on your own sets | `"test_splits": {"night": "test_night.txt"}` |

## Presets

| Preset | `CLI.backbone` | Default mode | Epochs |
| --- | --- | --- | --- |
| `clft` | `clft` | `cross_fusion` | 300 |
| `clftv2` | `clftv2` | `cross_fusion` | 200 |
| `maskformer` | `maskformer` | `cross_fusion` | 100 |
| `mask2former` | `mask2former` | `cross_fusion` | 100 |
| `deeplabv3plus` | `deeplabv3plus` | `fusion` | 200 |

Set `CLI.mode` to `rgb` (camera only), `lidar` (LiDAR only) or `fusion` (both). DeepLabV3+ calls
fusion `fusion` and the other models `cross_fusion`; either word works for every model.

Learning-rate schedules follow `General.epochs`, so changing the epochs needs no other change:
CLFTv2, MaskFormer, Mask2Former and DeepLabV3+ warm up and then decay over the remaining epochs;
CLFT multiplies its learning rate by `CLFT.lr_momentum` every epoch.

## Training classes

`Dataset.train_classes` says which classes the model learns and how the dataset's classes map onto
them. Each has an `index` (model output channel, `0..n-1` without gaps, `0` is background), the
`dataset_mapping` of dataset class indices merged into it, a loss `weight` and a visualization
`color`. The dataset's manifest suggests a set; set your own to merge or drop classes.

The weights are used by the models trained with cross-entropy (CLFT, CLFTv2, DeepLabV3+).
MaskFormer and Mask2Former are trained with their Hungarian matching loss, which does not use them.
