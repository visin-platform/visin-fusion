# Customize a run

The examples use one epoch, batch size 2, seed 0, no warmup, and no pretrained
weight download. An epoch covers the full training split. They check the
pipeline; they are not accuracy baselines.

## Change a few settings

Save this as `configs/my-zod-run.json`. It inherits a complete example and
changes only the run name, output directory, and training budget:

```json title="configs/my-zod-run.json"
{
  "extends": "./examples/zod/clftv2/fusion.json",
  "Summary": "ZOD CLFTv2 fusion baseline",
  "Log": {"logdir": "logs/my-zod-baseline"},
  "General": {"epochs": 50},
  "CLFTv2": {"pretrained": true, "warmup_epochs": 5}
}
```

```bash title="Run your config"
visin-fusion run -c configs/my-zod-run.json --upload --benchmark-device cuda
```

Alternatively, copy or download a complete config from a model page and edit
it directly. Those configs extend a built-in preset, so they can be saved
anywhere. Run commands from the checkout and keep your dataset variable exported.

## Model-specific settings

| Model | JSON settings section | Example input size |
| --- | --- | --- |
| CLFT | `CLFT` | 384 × 384 |
| CLFTv2 | `CLFTv2` | 256 × 256 |
| DeepLabV3+ | `DeepLabV3Plus` | 256 × 256 |
| MaskFormer | `MaskFormer` | 256 × 256 |
| Mask2Former | `Mask2Former` | 256 × 256 |

Use the matching section when enabling pretrained weights or changing a
model option. Pretrained backbones require a download or cached weights.

## Keep runs separate or resume

Set a new `Log.logdir` for each fresh experiment. To resume an existing run,
keep its log directory, set `General.resume_training` to `true`, and increase
`General.epochs` beyond the completed epoch count.

For a fixed local dataset path, replace `Dataset.dataset_root` with its folder
instead of an environment variable. See [Configs](../configs.md) for inheritance
and the [config reference](../reference/config.md) for every setting.
