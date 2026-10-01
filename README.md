# Visin Fusion

A Python library for semantic segmentation from camera images and projected LiDAR. Five research
model families share one call: `model(rgb, lidar)` returns class logits shaped
`[batch, classes, height, width]`.

## Install

Install from a source checkout:

```bash
git clone https://github.com/visin-platform/visin-fusion.git
cd visin-fusion
python -m pip install -e .
```

The base install provides the models. Add `.[train]` for the training and evaluation pipeline,
`.[visin]` for Visin reporting and datasets, or `.[dev]` for tests. See the
[installation and API guide](docs/library.md) for details.

## Use a model

```python
import torch
from visin_fusion.models import CLFTv2

model = CLFTv2(num_classes=4, mode="cross_fusion", pretrained=False).eval()
rgb = torch.randn(1, 3, 256, 256)
lidar = torch.randn(1, 3, 256, 256)  # camera-aligned LiDAR projection
with torch.inference_mode():
    logits = model(rgb, lidar)  # [1, 4, 256, 256]
```

| Model | Main difference |
| --- | --- |
| [CLFT](docs/architecture/CLFT.md) | Global ViT features fused during decoding |
| [CLFTv2](docs/architecture/CLFTv2.md) | Hierarchical Swin fusion pyramid |
| [MaskFormerFusion](docs/architecture/MaskFormer.md) | Class-labelled mask queries |
| [Mask2FormerFusion](docs/architecture/Mask2Former.md) | Multi-scale masked-attention queries |
| [DeepLabV3+](docs/architecture/DeepLabV3Plus.md) | Two convolutional branches with late fusion |

Each model page links its research papers. See [model comparison](docs/models.md) for the fusion points
and [Python API](docs/reference/python-api.md) for modes, training and checkpoints. The LiDAR input is
a three-channel projected image, not a raw point cloud.

## Optional pipeline

For ZOD, Waymo, and Iseauto datasets, follow [Download and train](docs/download-and-train.md):
install, download, and run the pipeline with CLI commands.

Train, test, visualize and benchmark CLFTv2 on the included sample dataset:

```bash
python -m pip install -e '.[train]'
visin-fusion run -c configs/quickstart.json
```

The [pipeline guide](docs/running.md) covers configs, Docker, SLURM and individual stages. Add
`.[train,visin]` and a Visin pipeline key to report runs; see the [Visin integration guide](docs/visin.md).

## Tests

```bash
python -m pip install -e '.[train,visin,dev]'
pytest tests/unit
python tools/coverage.py  # unit + end-to-end; requires >90% package coverage
```

See [tests/README.md](tests/README.md) for focused end-to-end commands. The full documentation is at
[visin-platform.github.io/visin-fusion](https://visin-platform.github.io/visin-fusion/).

## License

[MIT](LICENSE).
