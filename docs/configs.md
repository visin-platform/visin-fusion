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

Every stage and `run.py` load a config the same way (`utils/config.py`):

1. **`extends`**: the preset (`configs/presets/<name>.json`) or file (`"./base.json"`, relative to the
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
