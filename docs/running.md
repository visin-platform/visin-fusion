# Running

## The whole pipeline

```bash
visin-fusion run -c <config.json>                          # train, test, visualize, benchmark
visin-fusion run -c <config.json> --stages test,visualize  # some stages, in order
visin-fusion run -c <config.json> --upload                 # also upload visualizations to Visin
visin-fusion run -c <config.json> --benchmark-device cpu   # benchmark on one device only
```

`visin-fusion run` checks the config first, then runs each stage as its own process and stops at the first that
fails, with its exit code.

## Outputs

Each config writes to its `Log.logdir` (by default `logs/<dataset>/<model>-<mode>`):

| Folder | Contents |
| --- | --- |
| `epochs/` | Training and validation metrics, one file per epoch |
| `checkpoints/` | The best `General.max_checkpoints` checkpoints, with the model's description for `visin-fusion predict` |
| `tensorboard/` | `tensorboard --logdir <logdir>/tensorboard` |
| `test_results/` | Scores for each test set |
| `visualizations/` | Segment, overlay, comparison and correct-only images |
| `benchmark/` | Model size, speed and memory |

Testing and visualization use the best checkpoint. Benchmarking measures the same architecture with random weights. Visin reports share the training run.

## Retrying and reporting modes

Stages after training can be repeated without training again, as long as `Log.logdir` is unchanged:

```bash
visin-fusion run -c my_config.json --stages test,visualize,benchmark --benchmark-device cpu
```

Reporting to Visin follows the environment. For a local run without reporting, `VISIN_MODE=disabled visin-fusion run -c my_config.json`. On a compute node without network access, queue reports and send them later from a connected machine that shares the report directory and has the Visin credentials exported:

```bash
VISIN_DIR="$PWD/.visin" VISIN_MODE=offline visin-fusion run -c my_config.json --upload
VISIN_DIR="$PWD/.visin" visin sync
```

## From Python

```python
import visin_fusion

logdir = visin_fusion.run("my_config.json")                        # all four stages
logdir = visin_fusion.run({"extends": "clftv2", "General": {"epochs": 5}, ...}, stages=["train", "test"])
```

`run(config, stages=..., upload=False, benchmark_device=None, output_dir=None)` does what `visin-fusion run` does and returns the log directory. `config` is a file or a dict; it is checked before anything starts, and a mistake raises `ConfigError` or `ValueError`. Stages still run as separate processes, in order; a failing one raises `visin_fusion.pipeline.StageFailed` with its `stage` and `returncode`. This makes parameter sweeps a loop:

```python
for lr in (1e-4, 8e-5, 5e-5):
    visin_fusion.run({"extends": "clftv2", "CLFTv2": {"clft_lr": lr}, "Log": {"logdir": f"logs/lr_{lr}"}, ...})
```

## Callbacks

To log to your own tools (Weights & Biases, MLflow, a database), subclass `Callback`, override the events you need and name the class in the config. Every stage loads it:

```python
# my_package/tracking.py
from visin_fusion.engine.callbacks import Callback, EpochEnd


class Tracking(Callback):
    def __init__(self, config):
        self.config = config

    def on_epoch_end(self, event: EpochEnd) -> None:
        print(event.epoch, event.results["val"]["mean_iou"])
```

```json
"General": {"callbacks": ["my_package.tracking:Tracking"]}
```

The package must be importable where the stages run. Each `on_*` method receives one event object (a frozen dataclass from `visin_fusion.engine.callbacks`), so editors and type checkers know its fields:

| Event method | Event class | Fields |
| --- | --- | --- |
| `on_run_start` | `RunStart` | `config`, `state` |
| `on_epoch_end` | `EpochEnd` | `config`, `epoch`, `epoch_uuid`, `results` (`train`, `val`, `system_info`), `learning_rate`, `epoch_time` |
| `on_checkpoint` | `Checkpoint` | `config`, `epoch`, `epoch_uuid` |
| `on_run_end` | `RunEnd` | `config`, `error` (`None` on success) |
| `on_test_end` | `TestEnd` | `config`, `epoch`, `epoch_uuid`, `results`, `test_uuid` |
| `on_visualization` | `Visualization` | `config`, `epoch`, `epoch_uuid`, `output_dir`, `image_name` |
| `on_benchmark` | `Benchmark` | `results`, `system_info`, `training_uuid`, `epoch`, `epoch_uuid` |

A later release may add fields to an event, so read the ones you need by name. A callback that raises stops the stage.

## One stage

Each stage is a module, the same for every model (the config's `CLI.backbone` picks the model). Run
them from the repository root:

```bash
python -m visin_fusion.engine.stages.train.common -c <config.json> [--seed N]
python -m visin_fusion.engine.stages.test.common -c <config.json> [--checkpoint <file.pth>]
python -m visin_fusion.engine.stages.visualize.common -c <config.json> [-p <frames.txt>] [--upload]
python -m visin_fusion.engine.stages.benchmark.common -c <config.json> [<config.json> ...]
```


Every stage is one implementation for every model (`visin_fusion/engine/stages/<stage>/common.py`), built on
`visin_fusion/models/registry.py`:

- **Train** builds the model and its training setup from the registry (optimizer, learning-rate
  schedule, loss, gradient clipping, mixed precision), trains `General.epochs` epochs on the train split
  and validates on every frame of the validation split each epoch. Epoch logs go to `<logdir>/epochs/`,
  checkpoints (with the schedule's state, so a resumed run continues it) to `<logdir>/checkpoints/`,
  keeping the best `General.max_checkpoints` by validation mIoU. `--seed N` repeats a config with
  another seed into `<logdir>_seed<N>`.

- **Test** evaluates the best checkpoint in `Log.logdir` (or `--checkpoint`) on every frame of every test
  set of the dataset, and averages the sets into `overall`. Results go to `<logdir>/test_results/`.
- **Visualize** renders the best checkpoint's predictions on the visualization split (or `-p`) into
  `<logdir>/visualizations/{mask,segment,confidence,overlay,compare,correct_only}/`; `--upload` sends them to Visin.
- **Benchmark** measures the model without trained weights; see below.

Each exits non-zero when anything fails, so `visin-fusion run` and the SLURM jobs stop.

### Benchmark

The benchmark builds each config's model with random weights (speed does not depend on them, and
nothing is downloaded) and records parameters, FLOPs, inference time and memory:

```bash
python -m visin_fusion.engine.stages.benchmark.common -c <config.json> [<config.json> ...]   # CPU and GPU
python -m visin_fusion.engine.stages.benchmark.common -c configs/ --single --device cuda       # every config in a folder
```

It times 100 forward passes after 10 warm-up passes (`--num-runs`, `--warmup-runs`). GPU memory is
reported twice: `gpu_memory_*` is what stays allocated after a pass (mostly the weights), and
`gpu_memory_peak_mb` the peak during the passes, activations included. A config that fails is named
and makes the run exit non-zero; the others are still saved.

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

Visin credentials come from `.env` beside the Compose file or an external path set with
`VISIN_ENV_FILE=/path/to/visin.env`; the file is never copied into the image.

A config's relative `Log.logdir` is placed under `/outputs`. The GPU service needs the NVIDIA container
toolkit on the host. Images are published by `.github/workflows/image.yml` to
`ghcr.io/visin-platform/visin-fusion` (`:<sha>`, `:latest`, `-cpu` variants).

## SLURM

`slurms/run.slurm` runs a config through `visin-fusion run` on a SLURM cluster, from the venv or, with `IMAGE` set,
from the published image through Apptainer. Submit from the repository root; adjust its `#SBATCH`
resources to your cluster.

```bash
sbatch slurms/run.slurm my_config.json
sbatch slurms/run.slurm my_config.json --stages test,visualize
IMAGE=docker://ghcr.io/visin-platform/visin-fusion:latest sbatch slurms/run.slurm my_config.json
```

On nodes without internet, set `VISIN_MODE=offline` and send the reports later with `visin sync`
from a node that has it; download `visin:` datasets beforehand with `visin download <name>`.

To train, evaluate on a suite, publish to Hugging Face and report to Visin as one chain of jobs, with the
downloads and uploads done from the login node, see `slurms/hpc/README.md`.


## Downloadable models on Visin

Install the optional integration (`pip install -U 'visin-fusion[visin]'`), configure `VISIN_URL` and a
project pipeline key in `VISIN_TOKEN`, and add these settings to your config:

```json
{
  "General": {
    "publish_model": "visin",
    "model_license": "cc-by-4.0",
    "model_credit": "Tallinn University of Technology (TalTech)"
  }
}
```

Training publishes its best checkpoint before closing the run. It retains an inference copy under
`<logdir>/published/<checkpoint stem>/checkpoint.pth` with weights, `model_info`, and the epoch; optimizer
and scheduler state remain in the training checkpoint. This inference file has one SHA-256 shared by
its Visin download, suite evaluations, and ONNX exports. It survives checkpoint rotation. An unset
licence produces a warning: the model can be stored but Visin refuses its download with 409.

Offline runs copy the model into the SDK spool, so rotation cannot remove an unsent upload. This needs
space for the training checkpoint, its inference copy, and the spool copy. `visin sync` sends staged
files later; transient retries reuse the same reservation and claim. A slow upload that exceeds the
run's finish timeout stays queued for sync.

Load a published PyTorch model using its run UUID and model id:

```python
from visin_fusion.inference import Predictor

predictor = Predictor.from_pretrained("visin://<run UUID>/<model id>", device="cpu")
```

The SDK checks project access and verifies the download's SHA-256 before PyTorch reads it. Public models
need only `VISIN_URL`; private models also need an authorized credential. The same reference works in
`visin-fusion predict --checkpoint visin://<run UUID>/<model id> --input camera/ --output predictions/`.

## Confidence and output geometry

`Predictor.probabilities()` returns class probabilities at the original image size. It resizes logits
bilinearly, with `align_corners=False`, then applies softmax. `predict()` takes argmax of the same resized
logits. `confidence()` encodes the maximum probability as an 8-bit image, `round(255 * p)`; request it from
the command with `predict --confidence`. The mask is a palette PNG retaining class indices.

The visualize stage writes matching `mask/`, `segment/`, and `confidence/` images at the original frame
size. With `--upload`, each records its frame, dimensions, and confidence encoding in `metadata.prediction`.
The dataset marker surrounding the supplied `--path` frame list identifies the prediction dataset, which
can differ from the training dataset. Old or unmarked frame lists leave prediction provenance unknown.

## ONNX export and parity

Install `pip install 'visin-fusion[onnx]'` for the optional ONNX and CPU Runtime dependencies:

```bash
visin-fusion export --checkpoint logs/run/published/epoch_99/checkpoint.pth \
  --format onnx --output model.onnx --frames sample/all.txt --dataset-root sample/
```

The graph includes softmax and outputs `[B, C, H, W]` probabilities. Inputs are named `rgb`, `lidar`, or
both according to mode. The spatial size is static and comes from `model_info.transforms.resize`, with
an optional `--size`; batch size is dynamic and the opset is 17. Metadata carries the model description,
preprocessing, and source inference-file digest. Exporting a full training checkpoint first creates or
reuses that same inference copy.

Parity checks every supplied camera/LiDAR frame against PyTorch at the model's input size: maximum
probability difference must be below `1e-4`, and differing argmax pixels at most `0.01%`. A failed check
removes the export. Without a frame list, seeded synthetic inputs check the graph, with `parity.frames=0`;
publishing requires an actual frame list. Add `--publish --run <run UUID> --licence cc-by-4.0 --credit TalTech`
to save the ONNX file under the same checkpoint identity. This must be the run that produced the weights.

To reproduce input-image decoding from ONNX, take log probabilities, resize them bilinearly to the frame,
then softmax and argmax. Log probabilities equal logits up to a per-pixel common offset, so this preserves
Predictor's decoder; directly resizing probabilities uses a different decoder. Float underflow to zero
requires clipping to the smallest positive value before taking logs.

CLFT and CLFTv2 are the initial supported targets. MaskFormer and Mask2Former may fail export because of
model-specific attention operations. A single ONNX protobuf cannot hold 2 GiB or more of weights: exports
requiring external tensor files are refused, rather than publishing a download missing those tensors.
Use a smaller model; fp16 and multi-file packaging are not implemented.
