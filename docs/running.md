# Running

## The whole pipeline

```bash
python run.py -c <config.json>                          # train, test, visualize, benchmark
python run.py -c <config.json> --stages test,visualize  # some stages, in order
python run.py -c <config.json> --upload                 # also upload visualizations to Visin
python run.py -c <config.json> --benchmark-device cpu   # benchmark on one device only
```

`run.py` checks the config first, then runs each stage as its own process and stops at the first that
fails, with its exit code.

## One stage

Each stage is a module, the same for every model (the config's `CLI.backbone` picks the model). Run
them from the repository root:

```bash
python -m stages.train.common -c <config.json> [--seed N]
python -m stages.test.common -c <config.json> [--checkpoint <file.pth>]
python -m stages.visualize.common -c <config.json> [-p <frames.txt>] [--upload]
python -m stages.benchmark.common -c <config.json> [<config.json> ...]
```


### Benchmarking

Every model shares one benchmark, `python -m stages.benchmark.common` (the per-model benchmark modules
call it). It builds each config's model with random weights (speed does not depend on them, and
nothing is downloaded) and records parameters, FLOPs, inference time and memory:

```bash
python -m stages.benchmark.common -c <config.json> [<config.json> ...]   # CPU and GPU
python -m stages.benchmark.common -c configs/ --single --device cuda       # every config in a folder
```

It times 100 forward passes after 10 warm-up passes (`--num-runs`, `--warmup-runs`). GPU memory is
reported twice: `gpu_memory_*` is what stays allocated after a pass (mostly the weights), and
`gpu_memory_peak_mb` the peak during the passes, activations included. A config that fails is named
and makes the run exit non-zero; the others are still saved.

Every stage is one implementation for every model (`stages/<stage>/common.py`), built on
`models/registry.py`:

- **Train** builds the model and its training setup from the registry (optimizer, learning-rate
  schedule, loss, gradient clipping, mixed precision), trains `General.epochs` epochs on the train split
  and validates on every frame of the validation split each epoch. Epoch logs go to `<logdir>/epochs/`,
  checkpoints (with the schedule's state, so a resumed run continues it) to `<logdir>/checkpoints/`,
  keeping the best `General.max_checkpoints` by validation mIoU. `--seed N` repeats a config with
  another seed into `<logdir>_seed<N>`.

- **Test** evaluates the best checkpoint in `Log.logdir` (or `--checkpoint`) on every frame of every test
  set of the dataset, and averages the sets into `overall`. Results go to `<logdir>/test_results/`.
- **Visualize** renders the best checkpoint's predictions on the visualization split (or `-p`) into
  `<logdir>/visualizations/{segment,overlay,compare,correct_only}/`; `--upload` sends them to Visin.
- **Benchmark** is described below.

Each exits non-zero when anything fails, so `run.py` and the SLURM jobs stop.

## Docker

The image holds the code and its dependencies; data, outputs and downloaded weights are mounted:

| In the container | On the host (default) | |
| --- | --- | --- |
| `/data` | `./data` (`DATA_DIR`) | datasets; `visin:` datasets are downloaded here |
| `/outputs` | `./outputs` (`OUTPUT_DIR`) | logs and checkpoints |
| `/cache` | `./.cache` (`CACHE_DIR`) | pretrained backbone weights |

```bash
docker compose run --rm fusion -c configs/quickstart.json                                # GPU
docker compose run --rm fusion-cpu -c configs/quickstart.json --benchmark-device cpu     # CPU
```

A config's relative `Log.logdir` is placed under `/outputs`. The GPU service needs the NVIDIA container
toolkit on the host. Images are published by `.github/workflows/image.yml` to
`ghcr.io/visin-platform/visin-fusion` (`:<sha>`, `:latest`, `-cpu` variants).

## SLURM

`slurms/run.slurm` runs a config through `run.py` on a SLURM cluster, from the venv or, with `IMAGE` set,
from the published image through Apptainer. Submit from the repository root; adjust its `#SBATCH`
resources to your cluster.

```bash
sbatch slurms/run.slurm my_config.json
sbatch slurms/run.slurm my_config.json --stages test,visualize
IMAGE=docker://ghcr.io/visin-platform/visin-fusion:latest sbatch slurms/run.slurm my_config.json
```

On nodes without internet, set `VISIN_MODE=offline` and send the reports later with `visin sync`
from a node that has it; download `visin:` datasets beforehand with `visin download <name>`.
