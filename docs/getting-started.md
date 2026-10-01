# Getting started

For a full dataset and copyable CLI commands, follow [Download and train](download-and-train.md).

## Install from source

```bash
git clone https://github.com/visin-platform/visin-fusion.git
cd visin-fusion
python3 -m venv venv && source venv/bin/activate
python -m pip install -e .
```

The base install provides the model classes. Add `[train]` for the runnable pipeline and `[visin]` for reporting; see [Use as a Python library](library.md#install).

## Call a model

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

The two inputs are image tensors. For a single stream, set `mode="rgb"` or `mode="lidar"`. See the [library API](library.md) for training and checkpoint examples, or [compare the five models](models.md) before choosing one.

## Run the sample pipeline

The repository includes a 28-frame ZOD sample in `tests/data/zod_sample`. Install the pipeline extra and run its short CLFTv2 example:

```bash
python -m pip install -e '.[train]'
visin-fusion run -c configs/quickstart.json
```

The pipeline trains, tests, visualizes and benchmarks. It writes epoch logs, checkpoints, test results, images and benchmark results under `logs/quickstart/`. [Running](running.md) documents stages, Docker and SLURM; [Configs](configs.md) and [Datasets](datasets.md) cover your own data.

## Optional Visin reporting

```bash
python -m pip install -e '.[train,visin]'
cp .env.example .env  # then set VISIN_TOKEN; or export VISIN_ENV_FILE=/path/to/visin.env
```

Without a pipeline key, the run remains local. See the [Visin integration guide](visin.md) for offline nodes and report syncing.
