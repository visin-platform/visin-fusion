# Visin Fusion

Train and evaluate semantic segmentation models that combine a camera image with a LiDAR projection,
for autonomous driving. Five models share one pipeline:

| Model | Preset |
| --- | --- |
| CLFT (ViT) | `clft` |
| SwinFusion / CLFTv2 (Swin) | `swin` |
| MaskFormer (Swin backbone) | `maskformer` |
| Mask2Former (Swin backbone) | `mask2former` |
| DeepLabV3+ (ResNet-101) | `deeplabv3plus` |

Each takes the camera image (`rgb`), the LiDAR projection (`lidar`) or both (`fusion`).

A run is described by one config. It trains a model, tests it on each test set of the dataset,
renders visualizations and benchmarks speed and memory, and can report all of it to
[Visin](https://app.visin.eu).

```bash
python run.py -c configs/quickstart.json
```

Start with [Getting started](getting-started.md), then [Configs](configs.md) and
[Datasets](datasets.md) to train on your own data.
