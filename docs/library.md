# Use as a Python library

`visin_fusion.models` can be used inside your own data loader, training loop or inference service. A pipeline config, Docker and Visin reporting are optional. The package supplies the five model classes and the pipeline; the `visin` extra adds the Visin integration.

## Install

Install from a source checkout:

```bash
git clone https://github.com/visin-platform/visin-fusion.git
cd visin-fusion
python -m pip install -e .                  # models, Python API and the pipeline
python -m pip install -e '.[visin]'          # also Visin reporting and visin: datasets
```

Python 3.10+ is declared in `pyproject.toml`. The install depends on PyTorch, torchvision, timm and einops for the models, and on NumPy, pydantic, TensorBoard, OpenCV, pandas and a few more for the pipeline (all listed in `pyproject.toml`). The optional `dev` and `docs` extras provide tests and the documentation site.

## Choose a model

```python
from visin_fusion.models import (
    CLFT,
    CLFTv2,
    MaskFormerFusion,
    Mask2FormerFusion,
    DeepLabV3Plus,
)
```

| Class | Both-stream mode | Encoder and prediction style |
| --- | --- | --- |
| `CLFT` | `cross_fusion` | ViT tokens, progressive fusion, dense head |
| `CLFTv2` | `cross_fusion` | Swin pyramid, per-scale fusion, dense head |
| `MaskFormerFusion` | `cross_fusion` | Swin features, FPN, class-labelled mask queries |
| `Mask2FormerFusion` | `cross_fusion` | Swin features, multi-scale pixel decoder, masked queries |
| `DeepLabV3Plus` | `fusion` | Two ResNet-101/ASPP/decoder branches, late fusion |

See [Compare models](models.md) for diagrams, implementation differences and paper links. Every class also accepts `mode="rgb"` or `mode="lidar"`. The mode is chosen when constructing the model, and either spelling of the two-stream mode (`"fusion"` or `"cross_fusion"`) works for every class.

## Inference contract

The call is the same for all five classes: `model(rgb, lidar)`. Both arguments are PyTorch tensors shaped **`[batch, 3, height, width]`**. `lidar` is a camera-aligned *projection* with three channels, not an unordered point cloud or radar tensor. The returned tensor contains **unnormalized semantic logits** shaped `[batch, num_classes, height, width]`. Use `argmax(dim=1)` for integer class predictions. A single-stream model needs only its own input (`model(rgb=image)` or `model(lidar=projection)`); a missing or wrongly shaped input raises a `ValueError` that names it.

<!-- doctest -->
```python
import torch
from visin_fusion.models import CLFTv2

model = CLFTv2(num_classes=4, mode="cross_fusion", pretrained=False).eval()
rgb = torch.randn(1, 3, 256, 256)
lidar = torch.randn(1, 3, 256, 256)
with torch.inference_mode():
    logits = model(rgb, lidar)
    labels = logits.argmax(dim=1)
assert logits.shape == (1, 4, 256, 256)
```

For single-stream modes, pass both arguments to the common call; the model uses the selected stream. `CLFT` has an `image_size` constructor argument (384 by default) because its token reassembly is configured for a fixed square input. The other model constructors and their options are in [`models/api.py`](https://github.com/visin-platform/visin-fusion/blob/main/visin_fusion/models/api.py).

## Training from Python

You can supply your own optimizer and loss. If you want the defaults used by the pipeline, call `training_setup(...)` after placing the model on its device. It returns an optimizer, optional scheduler, loss callable, and optional gradient clipping value.

```python
import torch
from visin_fusion.models import Mask2FormerFusion

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Mask2FormerFusion(num_classes=4, mode="cross_fusion").to(device)
setup = model.training_setup(class_weights=[1.0] * 4, device=device, epochs=20)

for rgb, lidar, labels in loader:
    rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
    setup.optimizer.zero_grad()
    raw = model.raw_forward(rgb, lidar)
    logits = model.segment_from_raw(raw)
    loss = setup.loss(raw, logits, labels)  # labels: [batch, height, width]
    loss.backward()
    if setup.clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), setup.clip_grad_norm)
    setup.optimizer.step()
```

The `raw_forward` and `segment_from_raw` pair matters for MaskFormer and Mask2Former: their training loss uses query class and mask predictions in addition to the dense map. For CLFT, CLFTv2 and DeepLabV3+, the same loop works with weighted cross-entropy. Call `setup.scheduler.step()` according to your epoch/validation schedule when a scheduler is present; PyTorch's `ReduceLROnPlateau` variant needs a validation metric.

## Checkpoints

Save and load the public model's state dictionary as usual. The public wrapper preserves the underlying model's checkpoint key names.

```python
torch.save({"model_state_dict": model.state_dict()}, "model.pth")
restored = Mask2FormerFusion.from_pretrained("model.pth", num_classes=4, mode="cross_fusion")
```

`from_pretrained` also accepts a name registered with `ModelClass.register_pretrained(name, URL)`. This release does **not** ship named pretrained weights; use a local checkpoint or register your own URL. Match the class, class count and architecture options when loading a checkpoint.

## Add your own model

A subclass of `FusionModel` plus one `register_model(...)` call makes a model usable by the whole pipeline; see [Add your own model](add-a-model.md).

## Use a trained model

A checkpoint written by the pipeline's train stage carries the model, its classes and the preprocessing it was trained with, so it needs no config:

```python
from visin_fusion.inference import Predictor

predictor = Predictor.from_checkpoint("logs/my_run/checkpoints/epoch_9_<uuid>.pth")
mask = predictor.predict("camera/000001.png", "lidar_png/000001.png")  # [H, W] uint8, the size of the image
overlay = predictor.overlay("camera/000001.png", mask)  # [H, W, 3] RGB
print(predictor.class_names)  # what each mask value means
```

- `predict` resizes and normalizes the inputs as in training and returns the class index of every pixel at the original image size. Pass a path or a PIL image. A camera-only model needs only the first argument, a LiDAR-only model `lidar=...`.
- `predictor.colorize(mask)` (an RGB array) uses the classes' `color` from the training config; `predictor.logits(...)` gives the scores at the model's input size.
- Checkpoints from before this was added have no `model_info`; pass the config that trained them: `Predictor.from_checkpoint(path, config="my_config.json")`.
- To build tensors for your own loop, `from visin_fusion.data.preprocessing import Preprocessor` and `Preprocessor(transforms)` give `load_rgb` and `load_lidar`, where `transforms` is the checkpoint's `info["transforms"]` or a config's `Dataset.transforms`. Raw images fed to a model without this step give wrong results.

From the command line, for a folder of camera images (the LiDAR projection of `.../camera/x.png` is `.../lidar_png/x.png`, or use `--lidar-dir`):

```bash
visin-fusion predict --checkpoint logs/my_run/checkpoints/epoch_9_<uuid>.pth --input data/camera --output predictions/
```

It writes `<name>_mask.png` and `<name>_overlay.png` for every image, in the same subfolders as under `--input`. The mask is a palette PNG: its pixel values are the class indices (`np.array(Image.open(path))`), and viewers show it in the classes' colors rather than black.

## Use the runnable pipeline from Python

`visin_fusion.run(config)` runs the pipeline from a script or notebook; see [Running](running.md#from-python).

## Use the runnable pipeline

The [CLI pipeline](running.md) supplies datasets, config validation, training, testing, visualization and benchmarking. Run `visin-fusion run -c config.json`. The optional `[visin]` extra adds Visin reporting and `visin:` dataset resolution. Credentials
come from exported variables or an application-owned `.env`/`VISIN_ENV_FILE`; see the
[Visin integration guide](visin.md). See [Configs](configs.md) and [Datasets](datasets.md) for that route.
