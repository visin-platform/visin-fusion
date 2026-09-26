# Visin Fusion

Train and compare camera + LiDAR semantic segmentation models for autonomous driving. One config
describes a run: pick a model's preset, point at a dataset, and `run.py` trains, tests on every test set,
renders visualizations and benchmarks speed and memory, reporting all of it to
[Visin](https://github.com/visin-platform/visin).

| Model | Preset | |
| --- | --- | --- |
| CLFT | `clft` | ViT camera + LiDAR fusion |
| SwinFusion (CLFTv2) | `swin` | hierarchical Swin fusion with an FPN-style decoder |
| MaskFormer | `maskformer` | query-based decoder on a Swin backbone |
| Mask2Former | `mask2former` | masked-attention decoder on a Swin backbone |
| DeepLabV3+ | `deeplabv3plus` | ResNet-101, late fusion of camera and LiDAR branches |

Every model trains on the camera image, the LiDAR projection or both. Datasets (ZOD, Waymo, iseAuto, or
your own in the same layout) come from [Visin](https://app.visin.eu/datasets) or a folder.

Documentation: https://visin-platform.github.io/visin-fusion/

## Quick start

Train, test, visualize and benchmark SwinFusion (CLFTv2) on the small sample dataset in the repo:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run.py -c configs/quickstart.json
```

Or in Docker (`fusion` uses the GPU, `fusion-cpu` does not; outputs land in `./outputs`):

```bash
docker compose run --rm fusion-cpu -c configs/quickstart.json --benchmark-device cpu
```

Put a Visin project token in `integrations/.env` (see `integrations/.env.example`) to follow the run in
[Visin](https://app.visin.eu); without one, everything runs and nothing is reported.

## Repository layout

```text
run.py              run a config through its pipeline: python run.py -c <config.json>
stages/             train/, test/, visualize/, benchmark/: common.py for every model (and
                    visualize/ground_truth.py to render labels)
models/             the architectures, and registry.py: how each is built, called and trained
core/               shared engines: training, testing, metrics, model builders, dataset loader
utils/              config loading and schema, dataset manifests and downloads, split files, metrics
integrations/       reporting to Visin
configs/            presets/ (one per model) and quickstart.json
tools/              command-line tools: download datasets, sample dataset, manifests, LiDAR projection
tests/              unit and end-to-end tests, and the sample dataset (tests/data/zod_sample)
docs/               the documentation site
slurms/             run.slurm: a config through run.py on a SLURM cluster
```

## Usage

A config (JSON) describes one run. Start from a model's preset in `configs/presets/` (`clft`, `swin`,
`maskformer`, `mask2former`, `deeplabv3plus`) and set only what is yours:

```json
{
  "extends": "swin",
  "Summary": "SwinFusion on my data, camera only",
  "CLI": {"mode": "rgb"},
  "Dataset": {"dataset_root": "/data/my_dataset"},
  "General": {"epochs": 50}
}
```

A preset holds the model's settings, training defaults and augmentation, in fusion mode unless
`CLI.mode` says `rgb` or `lidar`. The dataset's `dataset.json` supplies its splits, classes and
layout (see Dataset below). Anything can be overridden: dictionaries are merged key by key, other
values replaced. `extends` also takes a path to another config (`"./base.json"`). Logs and
checkpoints go to `Log.logdir`, by default `logs/<dataset>/<backbone>-<mode>`. A config is checked
before anything runs, and every mistake is listed.

`run.py` runs a config through every stage, or only some:

```bash
python run.py -c <config.json>                          # train, test, visualize, benchmark
python run.py -c <config.json> --stages test,visualize  # some stages
python run.py -c <config.json> --upload                 # also upload visualizations to Visin
```

Each stage is also a module of its own, the same for every model (`CLI.backbone` picks the model:
`clft`, `swin_fusion`, `maskformer`, `mask2former`, `deeplabv3plus`). Run them from the repository root:

```bash
python -m stages.train.common -c <config.json>
python -m stages.test.common -c <config.json>
python -m stages.visualize.common -c <config.json> [-p <frames.txt>]
python -m stages.benchmark.common -c <config.json> [<config.json> ...]
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/unit                  # fast
pytest tests/e2e --device cpu      # every model and mode through all four stages on the sample dataset
```

See [tests/README.md](tests/README.md).

## Dataset

Datasets are hosted at [https://app.visin.eu/datasets](https://app.visin.eu/datasets); a config that
says `"dataset_root": "visin:zod"` downloads ZOD the first time (`visin datasets`
shows what is there):

- **ZOD** – original frames plus SAM-generated semantic masks
- **Iseauto** – manually annotated frames for the autonomous vehicle platform
- **Waymo** – labeled images from the Waymo Open Dataset

Each archive holds the frames (`camera/`, `lidar_png/`), annotation folders and split files. A dataset
can describe itself in a `dataset.json` at its root (splits, classes, folder layout, LiDAR
normalization); a config then only needs `Dataset.dataset_root`. Write one with
`python tools/make_manifest.py --root <dataset> --config <a config for it>`; see
`utils/dataset_manifest.py` for the format and `tests/data/zod_sample/dataset.json` for an example.

## Papers using Visin Fusion

Each paper keeps its configs and results in its own repository.

- CLFTv2: Efficient Camera-LiDAR Fusion for Semantic Segmentation via Hierarchical Feature Pyramids

## License

See LICENSE file for details.
