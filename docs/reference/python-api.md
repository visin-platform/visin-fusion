# Python API reference

The public model classes are exported from `visin_fusion.models`. Their implementations are in `visin_fusion.models.api`; each class wraps its architecture behind the same dense segmentation call.

## Common methods

| Method | Purpose |
| --- | --- |
| `model(rgb, lidar)` | Return unnormalized semantic logits `[B, C, H, W]`. |
| `model.raw_forward(rgb, lidar)` | Return architecture-specific intermediate outputs. Needed for query losses. |
| `model.segment_from_raw(raw)` | Extract dense logits from those intermediate outputs. |
| `model.training_setup(class_weights, device="cpu", epochs=100, **options)` | Build the model's default optimizer, scheduler and loss. |
| `ModelClass.from_pretrained(path_or_name, **constructor_options)` | Load a local checkpoint or a registered URL. |
| `ModelClass.register_pretrained(name, url)` | Register a named checkpoint source for that class. |

`training_setup` returns `TrainingSetup(optimizer, scheduler, loss, clip_grad_norm, mixed_precision)`. Its loss callable takes `(raw_outputs, dense_logits, labels)`. Labels have shape `[B, H, W]` and contain class indices `0` through `num_classes - 1`. See [Training from Python](../library.md#training-from-python) for a loop that works with all five classes.

## Constructors

All constructors accept `num_classes` and `pretrained=False`. The `pretrained` argument controls *backbone* initialization; it does not load a trained fusion checkpoint. Use `from_pretrained` for the latter.

| Class | Default mode | Important constructor options |
| --- | --- | --- |
| [`CLFT`](../architecture/CLFT.md) | `cross_fusion` | `image_size=384`, `patch_size=16`, `model_timm="vit_base_patch16_384"`, `hooks`, `reassembles` |
| [`CLFTv2`](../architecture/CLFTv2.md) | `cross_fusion` | `model_timm="swinv2_tiny_window16_256"`, `fusion_strategy="residual_average"`, `reassembles` |
| [`MaskFormerFusion`](../architecture/MaskFormer.md) | `cross_fusion` | `backbone="swinv2_tiny_window16_256"`, `num_queries=100`, `pixel_decoder_channels=256` |
| [`Mask2FormerFusion`](../architecture/Mask2Former.md) | `cross_fusion` | `backbone="swinv2_tiny_window16_256"`, `num_queries=100`, `num_decoder_layers=9`, `n_encoder_layers=6` |
| [`DeepLabV3Plus`](../architecture/DeepLabV3Plus.md) | `fusion` | `backbone="resnet101"`, `fusion_strategy="residual_average"` |

Each class also accepts `mode="rgb"` or `mode="lidar"`. `DeepLabV3Plus` calls its two-stream mode `"fusion"`; the transformer models call it `"cross_fusion"`. `CLFT` uses `image_size` to configure fixed-square token reassembly. The query models also accept `transformer_d_model`; the full signatures and validation behavior are in [`models/api.py`](https://github.com/visin-platform/visin-fusion/blob/main/visin_fusion/models/api.py).

## Checkpoint format

`state_dict()` and `load_state_dict()` act on the underlying architecture's weights, preserving the repository's checkpoint keys. `from_pretrained` accepts either a plain state dictionary or a dictionary with `model_state_dict`. No named fusion weights are registered in this release.

## Pipeline adapter

The config-driven pipeline converts a validated config into one of these public classes with `visin_fusion.models.registry.from_config(config)`. This adapter is for the CLI; direct Python users can construct a model without a config or Visin dependency.
