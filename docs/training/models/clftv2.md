# CLFTv2 examples

Select a dataset and input mode below. Each example is a complete JSON file
with one epoch, batch size 2, and no pretrained weight download.

[Prepare your dataset](../setup.md#next-prepare-a-dataset) first, then
activate `.venv` and run commands from the Fusion checkout.

=== "ZOD"

    Export `ZOD_DATA_DIR` using the [ZOD setup](../datasets/zod.md).

    === "RGB"

        Save as `configs/examples/zod/clftv2/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/clftv2/rgb.json){ .md-button download }

        ```json title="configs/examples/zod/clftv2/rgb.json"
        {
          "extends": "clftv2",
          "Summary": "ZOD: clftv2 rgb, one epoch",
          "tags": [
            "zod",
            "clftv2",
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
            "logdir": "logs/zod/clftv2/rgb"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/clftv2/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/zod/clftv2/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/clftv2/lidar.json){ .md-button download }

        ```json title="configs/examples/zod/clftv2/lidar.json"
        {
          "extends": "clftv2",
          "Summary": "ZOD: clftv2 lidar, one epoch",
          "tags": [
            "zod",
            "clftv2",
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
            "logdir": "logs/zod/clftv2/lidar"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/clftv2/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/zod/clftv2/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/clftv2/fusion.json){ .md-button download }

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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/clftv2/fusion.json \
          --upload --benchmark-device cuda
        ```

=== "Waymo"

    Export `WAYMO_DATA_DIR` using the [Waymo setup](../datasets/waymo.md).

    === "RGB"

        Save as `configs/examples/waymo/clftv2/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/clftv2/rgb.json){ .md-button download }

        ```json title="configs/examples/waymo/clftv2/rgb.json"
        {
          "extends": "clftv2",
          "Summary": "WAYMO: clftv2 rgb, one epoch",
          "tags": [
            "waymo",
            "clftv2",
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
            "logdir": "logs/waymo/clftv2/rgb"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/clftv2/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/waymo/clftv2/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/clftv2/lidar.json){ .md-button download }

        ```json title="configs/examples/waymo/clftv2/lidar.json"
        {
          "extends": "clftv2",
          "Summary": "WAYMO: clftv2 lidar, one epoch",
          "tags": [
            "waymo",
            "clftv2",
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
            "logdir": "logs/waymo/clftv2/lidar"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/clftv2/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/waymo/clftv2/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/clftv2/fusion.json){ .md-button download }

        ```json title="configs/examples/waymo/clftv2/fusion.json"
        {
          "extends": "clftv2",
          "Summary": "WAYMO: clftv2 fusion, one epoch",
          "tags": [
            "waymo",
            "clftv2",
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
            "logdir": "logs/waymo/clftv2/fusion"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/clftv2/fusion.json \
          --upload --benchmark-device cuda
        ```

=== "Iseauto"

    Export `ISEAUTO_DATA_DIR` using the [Iseauto setup](../datasets/iseauto.md).

    === "RGB"

        Save as `configs/examples/iseauto/clftv2/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/clftv2/rgb.json){ .md-button download }

        ```json title="configs/examples/iseauto/clftv2/rgb.json"
        {
          "extends": "clftv2",
          "Summary": "ISEAUTO: clftv2 rgb, one epoch",
          "tags": [
            "iseauto",
            "clftv2",
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
            "logdir": "logs/iseauto/clftv2/rgb"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/clftv2/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/iseauto/clftv2/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/clftv2/lidar.json){ .md-button download }

        ```json title="configs/examples/iseauto/clftv2/lidar.json"
        {
          "extends": "clftv2",
          "Summary": "ISEAUTO: clftv2 lidar, one epoch",
          "tags": [
            "iseauto",
            "clftv2",
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
            "logdir": "logs/iseauto/clftv2/lidar"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/clftv2/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/iseauto/clftv2/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/clftv2/fusion.json){ .md-button download }

        ```json title="configs/examples/iseauto/clftv2/fusion.json"
        {
          "extends": "clftv2",
          "Summary": "ISEAUTO: clftv2 fusion, one epoch",
          "tags": [
            "iseauto",
            "clftv2",
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
            "logdir": "logs/iseauto/clftv2/fusion"
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

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/clftv2/fusion.json \
          --upload --benchmark-device cuda
        ```

On a CPU host, replace `cuda` with `cpu`. `--upload` enables prediction-image
uploads when Visin reporting is configured. One epoch is a pipeline check, not
an accuracy baseline.

Copied or downloaded configs can be saved under any filename: change `-c` to
that path. `extends` names a built-in model preset, so no parent config file
is required. Keep the dataset environment variable exported.

**Next:** [Run selected stages](../run.md) or [customize a run](../customize.md).
