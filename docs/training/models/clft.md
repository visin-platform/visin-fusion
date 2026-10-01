# CLFT examples

Select a dataset and input mode below. Each example is a complete JSON file
with one epoch, batch size 2, and no pretrained weight download.

[Prepare your dataset](../setup.md#next-prepare-a-dataset) first, then
activate `.venv` and run commands from the Fusion checkout.

=== "ZOD"

    Export `ZOD_DATA_DIR` using the [ZOD setup](../datasets/zod.md).

    === "RGB"

        Save as `configs/examples/zod/clft/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/clft/rgb.json){ .md-button download }

        ```json title="configs/examples/zod/clft/rgb.json"
        {
          "extends": "clft",
          "Summary": "ZOD: clft rgb, one epoch",
          "tags": [
            "zod",
            "clft",
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
            "logdir": "logs/zod/clft/rgb"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/clft/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/zod/clft/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/clft/lidar.json){ .md-button download }

        ```json title="configs/examples/zod/clft/lidar.json"
        {
          "extends": "clft",
          "Summary": "ZOD: clft lidar, one epoch",
          "tags": [
            "zod",
            "clft",
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
            "logdir": "logs/zod/clft/lidar"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/clft/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/zod/clft/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/zod/clft/fusion.json){ .md-button download }

        ```json title="configs/examples/zod/clft/fusion.json"
        {
          "extends": "clft",
          "Summary": "ZOD: clft fusion, one epoch",
          "tags": [
            "zod",
            "clft",
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
            "logdir": "logs/zod/clft/fusion"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/zod/clft/fusion.json \
          --upload --benchmark-device cuda
        ```

=== "Waymo"

    Export `WAYMO_DATA_DIR` using the [Waymo setup](../datasets/waymo.md).

    === "RGB"

        Save as `configs/examples/waymo/clft/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/clft/rgb.json){ .md-button download }

        ```json title="configs/examples/waymo/clft/rgb.json"
        {
          "extends": "clft",
          "Summary": "WAYMO: clft rgb, one epoch",
          "tags": [
            "waymo",
            "clft",
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
            "logdir": "logs/waymo/clft/rgb"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/clft/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/waymo/clft/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/clft/lidar.json){ .md-button download }

        ```json title="configs/examples/waymo/clft/lidar.json"
        {
          "extends": "clft",
          "Summary": "WAYMO: clft lidar, one epoch",
          "tags": [
            "waymo",
            "clft",
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
            "logdir": "logs/waymo/clft/lidar"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/clft/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/waymo/clft/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/waymo/clft/fusion.json){ .md-button download }

        ```json title="configs/examples/waymo/clft/fusion.json"
        {
          "extends": "clft",
          "Summary": "WAYMO: clft fusion, one epoch",
          "tags": [
            "waymo",
            "clft",
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
            "logdir": "logs/waymo/clft/fusion"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/waymo/clft/fusion.json \
          --upload --benchmark-device cuda
        ```

=== "Iseauto"

    Export `ISEAUTO_DATA_DIR` using the [Iseauto setup](../datasets/iseauto.md).

    === "RGB"

        Save as `configs/examples/iseauto/clft/rgb.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/clft/rgb.json){ .md-button download }

        ```json title="configs/examples/iseauto/clft/rgb.json"
        {
          "extends": "clft",
          "Summary": "ISEAUTO: clft rgb, one epoch",
          "tags": [
            "iseauto",
            "clft",
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
            "logdir": "logs/iseauto/clft/rgb"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/clft/rgb.json \
          --upload --benchmark-device cuda
        ```

    === "LIDAR"

        Save as `configs/examples/iseauto/clft/lidar.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/clft/lidar.json){ .md-button download }

        ```json title="configs/examples/iseauto/clft/lidar.json"
        {
          "extends": "clft",
          "Summary": "ISEAUTO: clft lidar, one epoch",
          "tags": [
            "iseauto",
            "clft",
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
            "logdir": "logs/iseauto/clft/lidar"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/clft/lidar.json \
          --upload --benchmark-device cuda
        ```

    === "Fusion"

        Save as `configs/examples/iseauto/clft/fusion.json` or use the file already in the checkout.

        [Download JSON](../../assets/configs/iseauto/clft/fusion.json){ .md-button download }

        ```json title="configs/examples/iseauto/clft/fusion.json"
        {
          "extends": "clft",
          "Summary": "ISEAUTO: clft fusion, one epoch",
          "tags": [
            "iseauto",
            "clft",
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
            "logdir": "logs/iseauto/clft/fusion"
          },
          "General": {
            "device": "cuda:0",
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
            "early_stop_patience": 1,
            "max_checkpoints": 1
          },
          "CLFT": {
            "warmup_epochs": 0,
            "pretrained": false
          }
        }
        ```

        ```bash title="Run all four stages"
        visin-fusion run \
          -c configs/examples/iseauto/clft/fusion.json \
          --upload --benchmark-device cuda
        ```

On a CPU host, replace `cuda` with `cpu`. `--upload` enables prediction-image
uploads when Visin reporting is configured. One epoch is a pipeline check, not
an accuracy baseline.

Copied or downloaded configs can be saved under any filename: change `-c` to
that path. `extends` names a built-in model preset, so no parent config file
is required. Keep the dataset environment variable exported.

**Next:** [Run selected stages](../run.md) or [customize a run](../customize.md).
