# Run and inspect results

Prepare a dataset and [choose a config](../training-examples.md) first.
These commands use ZOD, CLFTv2, and fusion; change `-c` for your selection.

## Run all stages

```bash title="Train → test → visualize → benchmark"
visin-fusion run -c configs/examples/zod/clftv2/fusion.json --upload --benchmark-device cuda
```

Use `--benchmark-device cpu` on a CPU host. Omit the option to benchmark CPU
and GPU when available. `--upload` sends rendered images when reporting is
configured; other reports do not need that flag. The pipeline stops when a stage fails.

## Inspect outputs

Each config writes to `logs/<dataset>/<model>/<mode>/`:

| Folder | Contents |
| --- | --- |
| `epochs/` | Training and validation metrics |
| `checkpoints/` | Best retained checkpoints |
| `test_results/` | Scores for each test split |
| `visualizations/` | Segment, overlay, comparison, and correct-only images |
| `benchmark/` | Model size, speed, and memory measurements |

Testing and visualization use the trained checkpoint. Benchmarking measures
the same architecture with random weights. Visin reports share the training run.

## Train only

```bash title="Only training and validation"
visin-fusion run -c configs/examples/zod/clftv2/fusion.json --stages train
```

## Retry later stages

```bash title="Reuse the trained checkpoint"
visin-fusion run -c configs/examples/zod/clftv2/fusion.json \
  --stages test,visualize,benchmark --upload --benchmark-device cuda
```

## Reporting modes

```bash title="Local run without reporting"
VISIN_MODE=disabled visin-fusion run -c configs/examples/zod/clftv2/fusion.json
```

For a compute node without network access, queue reports and sync later from
a connected machine with the same report directory and exported Visin credentials:

```bash title="Offline run, then sync"
VISIN_DIR="$PWD/.visin" VISIN_MODE=offline \
  visin-fusion run -c configs/examples/zod/clftv2/fusion.json --upload
VISIN_DIR="$PWD/.visin" visin sync
```

**Next:** [Customize a longer run](customize.md). [Running](../running.md)
also covers Docker, SLURM, and individual stage options.
