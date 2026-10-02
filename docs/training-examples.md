# Choose a model

Prepare [ZOD](training/datasets/zod.md), [Waymo](training/datasets/waymo.md),
or [Iseauto](training/datasets/iseauto.md), then open a model page below.
Each page has dataset and input-mode tabs, a copyable complete JSON config,
a downloadable JSON file, and the exact command to run it.

| Model | Example page | Input size |
| --- | --- | --- |
| CLFT | [Configs and commands](training/models/clft.md) | 384 × 384 |
| CLFTv2 | [Configs and commands](training/models/clftv2.md) | 256 × 256 |
| DeepLabV3+ | [Configs and commands](training/models/deeplabv3plus.md) | 256 × 256 |
| MaskFormer | [Configs and commands](training/models/maskformer.md) | 256 × 256 |
| Mask2Former | [Configs and commands](training/models/mask2former.md) | 256 × 256 |

## Choose an input mode

| Mode | Input | ZOD annotation folder | Waymo / Iseauto folder |
| --- | --- | --- | --- |
| RGB | Camera image | `annotation_camera_only` | `annotation` |
| LiDAR | Camera-aligned three-channel LiDAR projection | `annotation_lidar_only` | `annotation` |
| Fusion | Both inputs | `annotation_fusion` | `annotation` |

ZOD examples change the annotation set with the input mode. For a comparison
with identical targets across modes, set the same `Dataset.annotation_path`
in each config. `fusion` works for every model; the loader handles internal
mode names.

## Find a config in the checkout

```text
configs/examples/<dataset>/<model>/<rgb|lidar|fusion>.json
```

There are 45 combinations. Each file is complete and uses a built-in model
preset; no parent config file is required. Dataset manifests provide class
mappings and normalization: ZOD and Waymo train four classes, Iseauto three.

**Next:** [Run and inspect the outputs](running.md#outputs) or [customize a run](configs.md#customize-an-example).
