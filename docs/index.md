# Visin Fusion

A Python library for comparing camera and projected LiDAR semantic segmentation models. Import a model into your own code, or use the optional pipeline to train, evaluate, visualize and benchmark it from a config.

```python
from visin_fusion.models import CLFTv2

model = CLFTv2(num_classes=4, mode="cross_fusion")
logits = model(rgb, lidar)  # [batch, classes, height, width]
```

Five research model families share this calling convention. Their fusion points and prediction heads differ:

![Diagram: comparison of CLFT, CLFTv2, MaskFormer, Mask2Former and DeepLabV3+ by encoder, fusion point and prediction head.](assets/models/comparison.svg){ .model-diagram }

## Where to start

| I want to... | Go to |
| --- | --- |
| Try it in five minutes | [Getting started](getting-started.md#1-install-and-run-the-sample) |
| Train on my own data | [Getting started](getting-started.md#2-train-on-your-own-data), then [Configs](configs.md) |
| Train on ZOD, Waymo or iseAuto | [Download and train](download-and-train.md) |
| Call a model from my own code | [Use as a library](library.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) |

## All pages

- [Use as a library](library.md): installation, model API, inference, training and checkpoints.
- [Compare models](models.md): diagrams, implementation differences and paper references.
- [Getting started](getting-started.md): install, run the sample, and train on your own data, on one page.
- [Download and train](download-and-train.md): ZOD, Waymo, and Iseauto examples for all four CLI stages.
- [Training examples](training-examples.md): choose among five models and RGB, LiDAR, or fusion.
- [Configs](configs.md) and [Datasets](datasets.md): run the pipeline on your own data.

One install covers the models and the full pipeline. Visin reporting and `visin:` datasets use the optional `visin` extra.
