# MaskFormer examples

Select a dataset and input mode below. Each example is a complete JSON file
with one epoch, batch size 2, and no pretrained weight download.

[Prepare your dataset](../setup.md#next-prepare-a-dataset) first, then
activate `.venv` and run commands from the Fusion checkout.

=== "ZOD"

    Export `ZOD_DATA_DIR` using the [ZOD setup](../datasets/zod.md).

    === "RGB"

        Save as `configs/examples/zod/maskformer/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/maskformer/rgb.json){ .md-button download }

        ```json title="configs/examples/zod/maskformer/rgb.json"
        {
          "extends": "maskformer",
          "Summary": "ZOD: maskformer rgb, one epoch",
          "tags": [
            "zod",
            "maskformer",
            "rgb",
            "one-epoch"
          ],
          "CLI": {
            "mode": "rgb"
          },
          "Dataset": {
            "dataset_root": "$ZOD_DATA_DIR",
            "annotation_path": "annotation_camera_only"
          },
          "Log": {
            "logdir": "logs/zod/maskformer/rgb"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/maskformer/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/zod/maskformer/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/maskformer/lidar.json){ .md-button download }

        ```json title="configs/examples/zod/maskformer/lidar.json"
        {
          "extends": "maskformer",
          "Summary": "ZOD: maskformer lidar, one epoch",
          "tags": [
            "zod",
            "maskformer",
            "lidar",
            "one-epoch"
          ],
          "CLI": {
            "mode": "lidar"
          },
          "Dataset": {
            "dataset_root": "$ZOD_DATA_DIR",
            "annotation_path": "annotation_lidar_only"
          },
          "Log": {
            "logdir": "logs/zod/maskformer/lidar"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/maskformer/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/zod/maskformer/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/maskformer/fusion.json){ .md-button download }

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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/maskformer/fusion.json \
          --upload --benchmark-device cuda
        ```

=== "Waymo"

    Export `WAYMO_DATA_DIR` using the [Waymo setup](../datasets/waymo.md).

    === "RGB"

        Save as `configs/examples/waymo/maskformer/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/maskformer/rgb.json){ .md-button download }

        ```json title="configs/examples/waymo/maskformer/rgb.json"
        {
          "extends": "maskformer",
          "Summary": "WAYMO: maskformer rgb, one epoch",
          "tags": [
            "waymo",
            "maskformer",
            "rgb",
            "one-epoch"
          ],
          "CLI": {
            "mode": "rgb"
          },
          "Dataset": {
            "dataset_root": "$WAYMO_DATA_DIR",
            "annotation_path": "annotation"
          },
          "Log": {
            "logdir": "logs/waymo/maskformer/rgb"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/maskformer/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/waymo/maskformer/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/maskformer/lidar.json){ .md-button download }

        ```json title="configs/examples/waymo/maskformer/lidar.json"
        {
          "extends": "maskformer",
          "Summary": "WAYMO: maskformer lidar, one epoch",
          "tags": [
            "waymo",
            "maskformer",
            "lidar",
            "one-epoch"
          ],
          "CLI": {
            "mode": "lidar"
          },
          "Dataset": {
            "dataset_root": "$WAYMO_DATA_DIR",
            "annotation_path": "annotation"
          },
          "Log": {
            "logdir": "logs/waymo/maskformer/lidar"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/maskformer/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/waymo/maskformer/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/maskformer/fusion.json){ .md-button download }

        ```json title="configs/examples/waymo/maskformer/fusion.json"
        {
          "extends": "maskformer",
          "Summary": "WAYMO: maskformer fusion, one epoch",
          "tags": [
            "waymo",
            "maskformer",
            "fusion",
            "one-epoch"
          ],
          "CLI": {
            "mode": "fusion"
          },
          "Dataset": {
            "dataset_root": "$WAYMO_DATA_DIR",
            "annotation_path": "annotation"
          },
          "Log": {
            "logdir": "logs/waymo/maskformer/fusion"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/maskformer/fusion.json \
          --upload --benchmark-device cuda
        ```

=== "Iseauto"

    Export `ISEAUTO_DATA_DIR` using the [Iseauto setup](../datasets/iseauto.md).

    === "RGB"

        Save as `configs/examples/iseauto/maskformer/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/maskformer/rgb.json){ .md-button download }

        ```json title="configs/examples/iseauto/maskformer/rgb.json"
        {
          "extends": "maskformer",
          "Summary": "ISEAUTO: maskformer rgb, one epoch",
          "tags": [
            "iseauto",
            "maskformer",
            "rgb",
            "one-epoch"
          ],
          "CLI": {
            "mode": "rgb"
          },
          "Dataset": {
            "dataset_root": "$ISEAUTO_DATA_DIR",
            "annotation_path": "annotation"
          },
          "Log": {
            "logdir": "logs/iseauto/maskformer/rgb"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/maskformer/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/iseauto/maskformer/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/maskformer/lidar.json){ .md-button download }

        ```json title="configs/examples/iseauto/maskformer/lidar.json"
        {
          "extends": "maskformer",
          "Summary": "ISEAUTO: maskformer lidar, one epoch",
          "tags": [
            "iseauto",
            "maskformer",
            "lidar",
            "one-epoch"
          ],
          "CLI": {
            "mode": "lidar"
          },
          "Dataset": {
            "dataset_root": "$ISEAUTO_DATA_DIR",
            "annotation_path": "annotation"
          },
          "Log": {
            "logdir": "logs/iseauto/maskformer/lidar"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/maskformer/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/iseauto/maskformer/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/maskformer/fusion.json){ .md-button download }

        ```json title="configs/examples/iseauto/maskformer/fusion.json"
        {
          "extends": "maskformer",
          "Summary": "ISEAUTO: maskformer fusion, one epoch",
          "tags": [
            "iseauto",
            "maskformer",
            "fusion",
            "one-epoch"
          ],
          "CLI": {
            "mode": "fusion"
          },
          "Dataset": {
            "dataset_root": "$ISEAUTO_DATA_DIR",
            "annotation_path": "annotation"
          },
          "Log": {
            "logdir": "logs/iseauto/maskformer/fusion"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/maskformer/fusion.json \
          --upload --benchmark-device cuda
        ```

On a CPU host, replace `cuda` with `cpu`. `--upload` enables prediction-image
uploads when Visin reporting is configured. One epoch is a pipeline check, not
an accuracy baseline.

Copied or downloaded configs can be saved under any filename: change `-c` to
that path. `extends` names a built-in model preset, so no parent config file
is required. Keep the dataset environment variable exported.

**Next:** [Run selected stages](../run.md) or [customize a run](../customize.md).
