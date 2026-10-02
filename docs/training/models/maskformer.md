# MaskFormer examples

Each example is a complete JSON file with one epoch, batch size 2, and no pretrained weight download.
[Prepare your dataset](../../download-and-train.md) first, then activate `.venv` and run commands
from the Fusion checkout.

## The config

ZOD with both camera and LiDAR (`configs/examples/zod/maskformer/fusion.json`):

```json title="configs/examples/zod/maskformer/fusion.json"
{
  "extends": "maskformer",
  "Summary": "ZOD: maskformer fusion, one epoch",
  "tags": [
    "zod",
    "maskformer",
    "fusion",
    "one-epoch"
  ],
  "CLI": {
    "mode": "fusion"
  },
  "Dataset": {
    "dataset_root": "$ZOD_DATA_DIR",
    "annotation_path": "annotation_fusion"
  },
  "Log": {
    "logdir": "logs/zod/maskformer/fusion"
  },
  "General": {
    "device": "cuda:0",
    "epochs": 1,
    "batch_size": 2,
    "seed": 0,
    "early_stop_patience": 1,
    "max_checkpoints": 1
  },
  "MaskFormer": {
    "warmup_epochs": 0,
    "pretrained": false
  }
}
```

## Other datasets and inputs

Download the file for your dataset and input. Save it under any name and pass it to `-c`.

| Dataset | Camera only | LiDAR only | Camera + LiDAR |
| --- | --- | --- | --- |
| ZOD (`$ZOD_DATA_DIR`) | [rgb.json](../../assets/configs/zod/maskformer/rgb.json){ download } | [lidar.json](../../assets/configs/zod/maskformer/lidar.json){ download } | [fusion.json](../../assets/configs/zod/maskformer/fusion.json){ download } |
| Waymo (`$WAYMO_DATA_DIR`) | [rgb.json](../../assets/configs/waymo/maskformer/rgb.json){ download } | [lidar.json](../../assets/configs/waymo/maskformer/lidar.json){ download } | [fusion.json](../../assets/configs/waymo/maskformer/fusion.json){ download } |
| Iseauto (`$ISEAUTO_DATA_DIR`) | [rgb.json](../../assets/configs/iseauto/maskformer/rgb.json){ download } | [lidar.json](../../assets/configs/iseauto/maskformer/lidar.json){ download } | [fusion.json](../../assets/configs/iseauto/maskformer/fusion.json){ download } |

The files differ only in these keys:

| Key | Values |
| --- | --- |
| `CLI.mode` | `rgb`, `lidar`, or `fusion` |
| `Dataset.dataset_root` | `$ZOD_DATA_DIR`, `$WAYMO_DATA_DIR`, or `$ISEAUTO_DATA_DIR`: export the folder you prepared |
| `Dataset.annotation_path` | ZOD: `annotation_camera_only`, `annotation_lidar_only`, `annotation_fusion`; Waymo and Iseauto: `annotation` |
| `Log.logdir` | `logs/<dataset>/maskformer/<mode>`, so runs do not overwrite each other |
| `Summary`, `tags` | the run's name and labels in Visin |

## Run

```bash title="Run all four stages"
visin-fusion run -c configs/examples/zod/maskformer/fusion.json --upload --benchmark-device cuda
```

On a CPU host, replace `cuda` with `cpu`. `--upload` enables prediction-image uploads when Visin reporting
is configured. One epoch is a pipeline check, not an accuracy baseline. `extends` names a built-in model
preset, so no parent config file is required.

**Next:** [Run and inspect the outputs](../../running.md#outputs) or [customize a run](../../configs.md#customize-an-example).
