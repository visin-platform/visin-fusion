# TODO

Goal: a generic, reusable camera + LiDAR segmentation training tool. Anyone trains any of the models
on a Visin dataset or their own, from a small config, locally, in Docker or on a cluster, and follows
the run in Visin. Each paper lives in its own repository; nothing paper-specific stays here.
What changed in results along the way is in `CHANGELOG.md`.

## 1. Move to `visin-platform/visin-fusion`

The repo is generic now: paper configs, analysis scripts, ensembles and per-experiment SLURM jobs are
gone (they are in git history and the paper repositories), references use the new name, one generic
`slurms/run.slurm` runs any config, and each stage is one module (`stages/<stage>/common.py`).

- [ ] Create `visin-platform/visin-fusion` from the current tree as a fresh first commit: the old history
      has no secrets, but does have a personal HPC username, home paths (`/media/tom`, `/gpfs`) and HPC
      host names in 6-14 commits. The old repository keeps the full history
- [ ] After the first push: turn on Pages (Settings -> Pages -> Source: GitHub Actions), make the GHCR
      package public, add the repo description and topics, watch the first CI / image / docs runs
      (CPU time per model, GPU image size vs runner disk)
- [ ] List the paper repositories, with links, under "Papers using Visin Fusion" in the README

## 2. Datasets

The unzipped ZOD, Waymo and iseAuto folders have a `dataset.json` (checked frame by frame); a config
says `"dataset_root": "visin:zod"` and the `visin` package downloads it on first use (`visin.Datasets`,
`visin download`).

- [ ] Re-zip ZOD, Waymo and iseAuto with their `dataset.json` and upload them to Visin
- [ ] Licences: the Waymo zip is public on Visin, but the Waymo Open Dataset's terms restrict
      redistribution; check what they allow (and ZOD's and iseAuto's); add `license` and `url` to the
      manifests so users see them
- [ ] Data gaps: iseAuto `annotation_enhanced` lacks 112 frames; Waymo lacks one annotation
      (`labeled/day/not_rain/annotation/segment-10023947602400723454_..._0000000073.png`)
- [ ] The three datasets encode `lidar_png` differently (`docs/datasets.md`); re-project them from the
      raw point clouds with `tools/project_lidar.py`, so models transfer between datasets. Then retire
      `tools/waymo_pickle_to_png.py` and `tools/xod_pickle_to_png.py`
- [ ] Publish the sample dataset (`tests/data/zod_sample`) on Visin, for the quick start
- [ ] Release `visin` 0.2.0 with `visin.Datasets` (written in `visin-py`, not yet committed): this repo
      now requires `visin>=0.2`, so CI and fresh installs need it on PyPI. `visin-py` also has a failing
      test from before this change (`test_the_real_changelog_can_be_released`)
- [ ] The dataset service has no OpenAPI spec, so `visin-py`'s contract tests cannot check its dataset
      requests the way they check the vision service's
- [ ] A dataset version or content hash on Visin, recorded on each run
- [ ] Pass the Visin dataset id to `visin.init` so runs link to their dataset (only the name is sent)
- [ ] Image normalization in the manifest too (the presets use ImageNet statistics)
- [ ] Check at startup that the split files' frames exist, naming the missing folder, before any GPU
      work; an option to fail on a missing annotation instead of using an empty mask
- [ ] Worked example in `docs/datasets.md`: point clouds -> `project_lidar` -> `make_manifest` ->
      `dataset_stats --write` -> a config extending a preset
- [ ] Visin bug: the MCP `list_datasets` fails on a dataset without `archive.size`

## 3. Training behaviour

- [ ] SwinFusion and MaskFormer training is not reproducible run to run, even on CPU with fixed seeds
      (same code twice: metrics differ by up to several percent after one epoch); CLFT and DeepLabV3+
      are. Find the non-deterministic operations; optionally a config flag for deterministic algorithms
- [ ] AP is computed over the pixels predicted as or labelled as a class, not every pixel
      (`docs/metrics.md`); decide whether to switch every model to standard pixel AP
- [ ] Stage settings in their own config sections (`Test`, `Visualize`, `Benchmark`), not in `General`
- [ ] Inference on new frames: `predict -c cfg.json --checkpoint ... --input <frames>` (visualize only
      reads split files)

## 4. Containers and deployment

The image builds (CPU and GPU variants, `.github/workflows/image.yml`) and the quick start runs in
the CPU container.

- [ ] GPU in Docker: the NVIDIA Container Toolkit goes on the host, not in the image (the image already
      has CUDA through torch). On the workstation, with sudo:
      `sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`,
      then `docker compose run --rm fusion -c configs/quickstart.json`
- [ ] Pin versions (`pyproject.toml` or a lock file); torch and the rest are unpinned
- [ ] Pretrained backbone weights for nodes without internet: pre-fetch into `/cache` (`HF_HOME`,
      `TORCH_HOME`) or bake into the image
- [ ] Apptainer on the cluster: `apptainer pull docker://ghcr.io/visin-platform/visin-fusion:<sha>`
- [ ] `train.yml` (`workflow_dispatch`: config, stages, target) launching on a self-hosted GPU runner or
      submitting to SLURM over SSH; secrets `VISIN_TOKEN` and the SSH key as GitHub secrets
- [ ] GPU smoke test on the self-hosted runner, nightly and before releases
- [ ] Log the image tag, git SHA and the config's origin (preset, overrides) on each Visin run

## 5. Code

- [ ] Fold `core/model_builder.py` and `core/advanced_model_builder.py` into `models/registry.py`
- [ ] Option schemas for the model sections (`SwinFusion`, `CLFT`, ...), from the registry; today they
      are free-form
- [ ] `print()` -> `logging` (204 prints; only `integrations/` and the visin package log)
- [ ] Type annotations and `mypy`, starting with `utils/`, `models/registry.py` and `stages/`
      (`visin-py` is strictly typed; this repo has ~1,000 missing annotations)
- [ ] 26 unused variables (`F841`): check each (some are calls whose result is ignored), then enforce
- [ ] `ruff format` in one commit, then enforce formatting
- [ ] Break up the longest functions: `DatasetPNG.__getitem__` (99 lines), `TestingEngine.test` (79),
      `manage_checkpoints_by_miou` (102), `get_epoch_system_snapshot` (104)
- [ ] Raise the coverage floor (CI 35%, now 48%): unit tests for the training and testing engines, the
      visualizer and checkpoint management, or coverage collected from the e2e subprocesses
- [ ] Visin integration tests with `visin` mocked (resume finds the same run)

## 6. Docs and open source

- [ ] Pages: adding a model (in detail), the `lidar_png` encoding, handover notes (what is settled,
      open work, how SLURM and Visin fit together), CLI reference from `--help`, API reference from
      docstrings (`mkdocstrings`) with an `interrogate` threshold in CI
- [ ] Fold `integrations/README.md` into the docs
- [ ] Versioned docs per release (`mike`)
- [ ] `CITATION.cff` (for the tool), `CODE_OF_CONDUCT.md`, release workflow (image, `CHANGELOG.md`)
- [ ] Repo settings: description, topics, docs URL, Discussions; README badges (CI, docs, coverage, image)
- [ ] Pretrained weights per preset and dataset, published with model cards (dataset, classes, metrics
      from Visin, data licence); hardware notes per preset from the Visin benchmarks
- [ ] Access and ownership for a newcomer: GitHub org membership, Visin account and token, HPC account;
      repo, images, Visin project and datasets owned by the group, not a personal account

## 7. Definition of done

- [ ] README opens with a quick start of at most five commands (clone, token, `run.py` or
      `docker compose run` on the sample, open the run in Visin)
- [ ] Someone new follows only the README and docs on a clean machine: the sample in under 30 minutes,
      then a preset on a Visin dataset, then their own dataset; every place they get stuck becomes a fix.
      The same on the cluster

## Later: scale (many models x 100 datasets x 1M experiments)

The repo grows only with models; datasets, experiments and results are data, kept in Visin.

- [ ] Sweep definitions (presets x datasets x parameters) expanded by an orchestrator (SLURM job arrays,
      Kubernetes), not files in git; GitHub Actions stays CI/CD
- [ ] Config schema version in every config and the image tag on every run, with migrations when keys
      change, so stored configs stay runnable
- [ ] `run.py --config visin:<id>`; skip a run whose config, image and dataset version already have results
- [ ] Comparisons as Visin queries (model x dataset leaderboards)
