# CLFTv2 examples

Each example is a complete JSON file with one epoch, batch size 2, and no pretrained weight download.
[Prepare your dataset](../../download-and-train.md) first, then activate `.venv` and run commands
from the Fusion checkout.

## The config

ZOD with both camera and LiDAR (`configs/examples/zod/clftv2/fusion.json`):

```json title="configs/examples/zod/clftv2/fusion.json"
{
  "extends": "clftv2",
  "Summary": "ZOD: clftv2 fusion, one epoch",
  "tags": [
    "zod",
    "clftv2",
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
    "logdir": "logs/zod/clftv2/fusion"
  },
  "General": {
    "device": "cuda:0",
    "epochs": 1,
    "batch_size": 2,
    "seed": 0,
    "early_stop_patience": 1,
    "max_checkpoints": 1
  },
  "CLFTv2": {
    "warmup_epochs": 0,
    "pretrained": false
  }
}
```

## Other datasets and inputs

Download the file for your dataset and input. Save it under any name and pass it to `-c`.

| Dataset | Camera only | LiDAR only | Camera + LiDAR |
| --- | --- | --- | --- |
| ZOD (`$ZOD_DATA_DIR`) | [rgb.json](../../assets/configs/zod/clftv2/rgb.json){ download } | [lidar.json](../../assets/configs/zod/clftv2/lidar.json){ download } | [fusion.json](../../assets/configs/zod/clftv2/fusion.json){ download } |
| Waymo (`$WAYMO_DATA_DIR`) | [rgb.json](../../assets/configs/waymo/clftv2/rgb.json){ download } | [lidar.json](../../assets/configs/waymo/clftv2/lidar.json){ download } | [fusion.json](../../assets/configs/waymo/clftv2/fusion.json){ download } |
| Iseauto (`$ISEAUTO_DATA_DIR`) | [rgb.json](../../assets/configs/iseauto/clftv2/rgb.json){ download } | [lidar.json](../../assets/configs/iseauto/clftv2/lidar.json){ download } | [fusion.json](../../assets/configs/iseauto/clftv2/fusion.json){ download } |

The files differ only in these keys:

| Key | Values |
| --- | --- |
| `CLI.mode` | `rgb`, `lidar`, or `fusion` |
| `Dataset.dataset_root` | `$ZOD_DATA_DIR`, `$WAYMO_DATA_DIR`, or `$ISEAUTO_DATA_DIR`: export the folder you prepared |
| `Dataset.annotation_path` | ZOD: `annotation_camera_only`, `annotation_lidar_only`, `annotation_fusion`; Waymo and Iseauto: `annotation` |
| `Log.logdir` | `logs/<dataset>/clftv2/<mode>`, so runs do not overwrite each other |
| `Summary`, `tags` | the run's name and labels in Visin |

## Run

```bash title="Run all four stages"
visin-fusion run -c configs/examples/zod/clftv2/fusion.json --upload --benchmark-device cuda
```

On a CPU host, replace `cuda` with `cpu`. `--upload` enables prediction-image uploads when Visin reporting
is configured. One epoch is a pipeline check, not an accuracy baseline. `extends` names a built-in model
preset, so no parent config file is required.

**Next:** [Run and inspect the outputs](../../running.md#outputs) or [customize a run](../../configs.md#customize-an-example).
