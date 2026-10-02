# Add your own model

A model joins the pipeline by registering once. Train, test, visualize, benchmark, `predict` and Visin
reporting then work for it with no change to any stage, because none of them knows a model by name.

## 1. Write the model

Subclass `FusionModel`. It takes the camera image and the LiDAR projection (`[B, 3, H, W]` each) and returns
class scores `[B, num_classes, H, W]`. This one is a single convolution per stream:

<!-- doctest -->
```python
# my_package/models.py
import torch
from torch import nn

from visin_fusion.models import FusionModel, register_model


class TinyBody(nn.Module):
    def __init__(self, num_classes, width):
        super().__init__()
        self.rgb = nn.Conv2d(3, width, 3, padding=1)
        self.lidar = nn.Conv2d(3, width, 3, padding=1)
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, rgb, lidar, modal="rgb"):
        features = self.rgb(rgb) + (self.lidar(lidar) if modal != "rgb" else 0)
        return self.head(torch.relu(features))


class TinyNet(FusionModel):
    def __init__(self, num_classes=4, mode="cross_fusion", width=8, pretrained=False, training_options=None):
        super().__init__(TinyBody(num_classes, width), mode, training_options)


register_model("tinynet", TinyNet, section="TinyNet")
```

- `FusionModel` checks the inputs, accepts `rgb`/`lidar`/`fusion` modes (`cross_fusion` is the same as `fusion`), and calls the wrapped module as `module(rgb, lidar, modal=mode)`. If it returns a tensor, that is the logits. A different output layout (several tensors) overrides `raw_forward` and `segment_from_raw`; the transformer models in `visin_fusion/models/api.py` are examples.
- Training defaults to Adam and class-weighted cross-entropy. Override `training_setup(class_weights, device, epochs, **options)` to return your own optimizer, schedule and loss. It returns a `TrainingSetup`; see `visin_fusion/models/training.py`.
- `register_model(name, cls, section=...)` also takes `modes=` (default: `cls.modes`) and `options=`, a function `(config, settings) -> dict` for constructor arguments that come from elsewhere in the config, such as `Dataset.transforms.resize`.

## 2. Use it from a config

The settings in `"TinyNet"` are passed to the constructor as keyword arguments, next to `num_classes`, `mode` and `pretrained`. Keys the constructor does not take are ignored, and the whole section is available to `training_setup` as `self.training_options`.

```json
{
  "extends": "clftv2",
  "plugins": ["my_package.models"],
  "CLI": {"backbone": "tinynet", "mode": "fusion"},
  "TinyNet": {"width": 16},
  "Dataset": {"dataset_root": "/data/my_dataset"}
}
```

Extending a built-in preset supplies the dataset transforms and training settings. `plugins` lists the modules to import before the config is checked: every stage is a separate process, so a registration made only in your script would be missing there. The checkpoints a run writes record the plugins, so `visin_fusion.inference.Predictor` and `visin-fusion predict` import them too.

Then run it like any other model:

```bash
visin-fusion run -c my_config.json
```

## From Python

```python
import visin_fusion
import my_package.models  # registers "tinynet"

visin_fusion.run("my_config.json")
```

A config that uses a name nobody registered is rejected before anything starts, with the list of known models.
