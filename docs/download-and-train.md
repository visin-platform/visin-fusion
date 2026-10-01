# Start training

Pick a dataset, a model, and an input mode. The examples are organized into
short pages so you can copy one config and run its command.

| Step | Page | What you do |
| --- | --- | --- |
| 1 | [Install and connect](training/setup.md) | Install Fusion; optionally add a Visin pipeline key |
| 2 | [ZOD](training/datasets/zod.md), [Waymo](training/datasets/waymo.md), or [Iseauto](training/datasets/iseauto.md) | Download and prepare one dataset |
| 3 | [Choose a model](training-examples.md) | Select RGB, LiDAR, or fusion; copy or download its JSON |
| 4 | [Run and inspect results](training/run.md) | Train, test, visualize, and benchmark |

## First example

After [preparing ZOD](training/datasets/zod.md), run:

```bash title="ZOD + CLFTv2 + fusion"
visin-fusion run -c configs/examples/zod/clftv2/fusion.json --upload --benchmark-device cuda
```

Use `--benchmark-device cpu` on a CPU host. With `.env` configured, Fusion
reports to Visin; otherwise outputs remain local. All examples run one epoch
and use separate output directories. No extra experiments repository is needed.

**Want a longer experiment?** [Customize a run](training/customize.md).
