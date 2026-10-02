# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

These follow the standards of the sibling client library, `/mnt/ml/projects/visin-py`. Where this repo does not meet one yet, new and touched code meets it; don't reformat unrelated files inside a feature commit.

- AVOID inline comments in new code. Reasoning a reader needs goes in the docstring of the function or module, or in `docs/`. Existing comments stay until the code around them is touched. Delete commented-out code.
- Every public module, class, function and method has a docstring. It says what the caller gets and any surprising contract, not what the code does line by line.
- Annotate new and touched functions. Use `from __future__ import annotations`. Use `X | None` and built-in generics, not `Optional` and `typing.List`.
- No `print` outside `visin_fusion/cli.py` (ruff enforces it). Use `logger = logging.getLogger(__name__)` with lazy arguments (`logger.info("Epoch %d", epoch)`, never an f-string). Only entry points call `visin_fusion.logging_setup.configure_logging()`; library code never adds handlers. A model or layer never logs during `forward`.
- Configuration is read through the validated schema in `visin_fusion/config/config_schema.py`. Don't add new `config["A"]["b"]` lookups for keys the schema doesn't declare, and don't add a setting without a schema entry, a default and a line in `docs/reference/config.md` (`python tools/make_config_reference.py`).
- Imports live at the top of the file. A lazy import is for an optional dependency (the `visin` extra) and needs a reason in the docstring.
- No mutable default arguments. `zip(..., strict=True)` unless lengths legitimately differ. Catch specific exceptions; a broad `except Exception` needs a logged reason, and `except: pass` is never acceptable.
- `torch.load(..., weights_only=True)` for state dicts. `weights_only=False` only where a checkpoint carries optimizer or config objects, and only for checkpoints this project wrote.
- Keep functions small. As a guide, flag anything over 50 statements, 12 branches or 5 parameters. Group related parameters into a config dataclass instead of adding a sixth.
- Share, don't copy. A layer or loss used by more than one model lives once in `visin_fusion/models/` and is imported. Before writing a helper, grep for it.
- Public names are the ones exported from `visin_fusion/__init__.py` and `visin_fusion.models`. Anything else may change freely.
- Don't edit `CHANGELOG.md` release sections or `visin_fusion/_version.py` by hand. The Release workflow builds both from Conventional Commit messages (`tools/release.py`). Only add a line under `## [Unreleased]` for something a commit subject cannot say.
- Python 3.10+ at runtime. Core dependencies are everything the `visin-fusion` command and its stages import (the models' `torch`, `torchvision`, `timm`, `einops`, plus `numpy`, `pydantic`, `tensorboard` and so on), so a plain `pip install visin-fusion` works; add a new import to `dependencies` in the same commit. Only `visin` integration packages go behind an extra.

## Commands

```bash
make install            # venv with the dev tools and the pre-commit hooks
make test               # unit tests (seconds)
make e2e MODEL=clftv2   # end-to-end on the sample dataset, CPU, fusion mode
make lint               # ruff check and format check
make typecheck         # mypy on the annotated modules (listed in pyproject.toml)
make format             # ruff format and ruff check --fix
make coverage           # unit + fusion e2e coverage, fails under the floor
make docs               # mkdocs build --strict, plus config reference and diagram checks
make check              # what CI runs, minus the nightly all-modes e2e
```

Run `make lint typecheck test` before every commit, and `make e2e` for each model you touched. Never lower the coverage floor or add an ignore to pass a check: fix the finding, or widen a per-file ignore with a reason.

```bash
visin-fusion quickstart                              # the pipeline on the bundled sample
visin-fusion run -c configs/quickstart.json          # train, test, visualize, benchmark
visin-fusion run -c cfg.json --stages test,visualize
visin-fusion schema                                  # JSON Schema of a config
python -m visin_fusion.engine.stages.train.common -c cfg.json   # one stage
```

## Architecture Overview

`visin_fusion` trains and evaluates camera, LiDAR and camera+LiDAR semantic segmentation models, and optionally reports the runs to Visin through the `visin` package.

### Layers

1. **Config** (`config/`): `config.py` builds a config from a preset plus `extends`, dataset manifest and overrides. `config_schema.py` (pydantic) validates it. `dataset_manifest.py` and `splits.py` describe datasets. Presets are `config/presets/*.json`.
2. **Data** (`data/`): `dataset_png.py` and `data_loader.py` load frames in every mode; `visin_datasets.py` downloads datasets from Visin.
3. **Models** (`models/`): CLFT, CLFTv2, DeepLabV3+, MaskFormer and Mask2Former behind one interface (`api.py`, `registry.py`). `registry.py` is the only place that knows how to build, call and train each model; a model joins it with `register_model`.
4. **Engine** (`engine/`): `training_engine.py`, `testing_engine.py`, `visualizer.py`, `metrics_calculator.py` and the event `Callback` system (`callbacks.py`).
5. **Stages** (`engine/stages/{train,test,visualize,benchmark}/`): one implementation per stage for every model, run as separate processes by `cli.py`.
6. **Integrations** (`integrations/`): the optional Visin callback. It is the only code that imports `visin`.
7. **Dataset tools** (`dataset_tools/`, `visin-fusion dataset ...`) and **inference** (`inference.py`, `visin-fusion predict`): part of the installed package. **Repo tools** (`tools/`): docs generators, sample builder, release script; not installed.

### Key Design Decisions

**One pipeline for every model.** A model is added through the registry and a preset, never through a per-model stage. If a stage needs `if backbone == ...`, the registry is missing a hook.

**Modes are chosen at construction.** `rgb`, `lidar` and `cross_fusion` are a model argument, not something `forward` switches on per call in user code.

**Reporting never breaks training.** The `visin` package swallows delivery failures. The integration must not add code paths that raise into the training loop; with no `VISIN_TOKEN` nothing is reported and every stage runs unchanged.

**Training creates the run; later stages attach.** Test, visualize and benchmark find the training UUID in the epoch logs and never change run status. Epoch ids are deterministic (`engine/epoch_ids.py`), so retries are idempotent.

**Importing a module has no side effects.** Loading `.env`, setting `VISIN_URL` and configuring logging happen when a stage starts, not on import.

**The installed package never stores credentials.** Tokens come from the environment or a caller-owned `.env` / `VISIN_ENV_FILE`.

### Testing Strategy

Tests import from `visin_fusion.*` only. `tests/unit/` runs in seconds on CPU and needs no network; `tests/e2e/` runs every model through all four stages on the sample dataset in `visin_fusion/sample/zod_sample`. Test names are sentences stating the promise. Fix a bug by writing a failing test at the boundary that failed, preferring the public API over internals. Tests never touch the network, `~/.visin`, or a real Visin project unless `--visin online` is passed explicitly.

### Important Files for Common Tasks

- **Adding a model**: architecture in `models/`, a `register_model(...)` call in `models/registry.py` (it adds the schema's `BACKBONES` entry; `docs/add-a-model.md`), a preset in `config/presets/`, a line in `PRESETS` in `tests/e2e/test_pipelines.py`, then `python tools/make_config_reference.py` and `python tools/generate_model_diagrams.py`.
- **A config option**: `config/config_schema.py`, regenerate `docs/reference/config.md`, document in `docs/configs.md`.
- **A new callback event**: a dataclass, its `on_*` method and an `EVENT_HANDLERS` entry in `engine/callbacks.py`, the emitting engine, and every callback that cares (`docs/running.md#callbacks`).
- **Visin reporting**: `integrations/visin.py`; see `docs/visin.md`. Look at `/mnt/ml/projects/visin-py` (`CLAUDE.md`, `src/visin/__init__.py`) before changing how this repo calls `visin`.
- **A dataset**: no code; a `dataset.json` manifest (`docs/datasets.md`).

## Source of Truth Elsewhere

- **The Visin client**: `/mnt/ml/projects/visin-py`. Its public API is what `integrations/` may call. A signature change there is a dependency bump here.
- **The Visin server and its OpenAPI spec**: `/mnt/ml/projects/visin`.

## Making a Change

1. Read the code you are changing and the test that covers it.
2. Write the test first when fixing a bug.
3. Keep the patch scoped to the change. Cleanup of unrelated code is its own `refactor:` or `style:` commit, so review and `git blame` stay useful.
4. A user-facing change updates `docs/` in the same commit.
5. `make lint typecheck test`, plus `make e2e` for touched models, then commit with a Conventional Commit message: `feat:` (minor), `fix:` (patch), `feat!:` (breaking), or `docs:`, `test:`, `ci:`, `chore:`, `refactor:`, `build:`, `style:`.
6. Commit or push only when asked.

## Known Debt (work through it, don't add to it)

- `pyproject.toml` no longer ignores any rule family that has findings in `visin_fusion/`; what remains is listed with its reason. `tests/` and `tools/` still have per-file ignores (`D`, `T201`). `mypy` covers only the modules listed under `[tool.mypy]` in `pyproject.toml`; add a path there once a module is annotated and clean.
- Config is passed around as nested dicts; move consumers to the validated pydantic model.
