# Getting started

Everything you need to train, test, visualize and benchmark a model is on this page. Copy the blocks in order.

## 1. Install and run the sample

This trains CLFTv2 for two epochs on a 28-frame sample dataset that ships with the package, on CPU or GPU, and takes a few minutes. You need Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install visin-fusion
visin-fusion quickstart
```

One install covers the models and the whole pipeline (PyTorch, TensorBoard, OpenCV, pandas and the rest). Only Visin reporting is optional: `python -m pip install 'visin-fusion[visin]'`. To work from a source checkout instead, clone the repository and `python -m pip install -e .`.

Everything is written to `logs/quickstart/` in the current directory:

| What | Where |
| --- | --- |
| Epoch logs | `logs/quickstart/epochs/` |
| TensorBoard (`tensorboard --logdir logs/quickstart/tensorboard`) | `logs/quickstart/tensorboard/` |
| Checkpoints | `logs/quickstart/checkpoints/` |
| Test results per test set | `logs/quickstart/test_results/` |
| Prediction images | `logs/quickstart/visualizations/` |
| Speed and memory benchmarks | `logs/quickstart/benchmark/` |

`visin-fusion quickstart` takes the options of `run` (except `-c`), e.g. `--stages train,test`. See [Running](running.md) for Docker and SLURM.

## 2. Train on your own data

Put the data in this layout (the [dataset guide](datasets.md) explains it, and how to make the LiDAR images):

```text
/data/my_dataset/
  camera/000001.png       camera images
  lidar_png/000001.png    LiDAR projected onto the camera image
  annotation/000001.png   one channel; pixel value = class number
  train.txt               one frame per line: camera/000001.png
  validation.txt
  test.txt
  visualizations.txt      frames to render as images
```

Save this as `my_config.json`. It names every path and the class mapping, so nothing else is needed:

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
    "test_splits": {"test": "test.txt"},
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

Then run it, and use the trained model on new images:

```bash
visin-fusion run -c my_config.json
visin-fusion predict --checkpoint logs/my_dataset/checkpoints/<best>.pth --input /data/new_images/camera --output predictions/
```

`predict` writes a class-index mask and an overlay per image; [Use a trained model](library.md#use-a-trained-model) shows the Python API.

The pieces that people most often get wrong:

- **Where the files are.** `dataset_root` is the folder with `camera/`, `lidar_png/` and the annotation folder. Frames listed in the split files are relative to it. `train_split` and `val_split` are file paths. `test_splits` and `visualization_split` are file names looked up next to `val_split`, or full paths.
- **Class mapping.** `dataset_classes` lists the numbers in your annotation images. `train_classes` lists what the model learns: each one merges the dataset numbers in `dataset_mapping` (above, car and truck become `vehicle`), and a dataset number that is in no mapping becomes background. `index` counts `0, 1, 2...` with no gaps and `0` is background.
- **Check your labels.** Annotation PNGs hold class numbers, so they look black in a viewer. `visin-fusion dataset preview --root /data/my_dataset` renders them in color next to the camera images.
- **LiDAR statistics.** `visin-fusion dataset stats --root /data/my_dataset` prints `lidar_mean`, `lidar_std` and suggested class weights.

Prefer a manifest instead? Put a `dataset.json` in the dataset folder and the config shrinks to `{"extends": "clftv2", "Dataset": {"dataset_root": "/data/my_dataset"}}`; see [Datasets](datasets.md#the-manifest). All the options are in [Configs](configs.md) and the [config reference](reference/config.md).

## 3. Use a standard dataset

ZOD, Waymo and iseAuto download from Visin in the right layout, with their manifests:

```bash
python -m pip install -e '.[visin]'
visin download zod        # prints the folder
export ZOD_DATA_DIR=<that folder>
visin-fusion run -c configs/examples/zod/clftv2/fusion.json --benchmark-device cpu
```

[Download and train](download-and-train.md) has the examples for every dataset, model and input mode.

## Call a model from Python

<!-- doctest -->
```python
import torch
from visin_fusion.models import CLFTv2

model = CLFTv2(num_classes=4, mode="cross_fusion", pretrained=False).eval()
rgb = torch.randn(1, 3, 256, 256)
lidar = torch.randn(1, 3, 256, 256)  # projected LiDAR image
with torch.inference_mode():
    logits = model(rgb, lidar)
print(logits.shape)  # [1, 4, 256, 256]
```

For a single stream, set `mode="rgb"` or `mode="lidar"`. See the [library API](library.md) and [model comparison](models.md).

## Optional: Visin reporting

```bash
python -m pip install -e '.[visin]'
cp .env.example .env  # then set VISIN_TOKEN; or export VISIN_ENV_FILE=/path/to/visin.env
```

Without a pipeline key, the run stays local. See the [Visin integration guide](visin.md).
