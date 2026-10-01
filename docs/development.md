# Development

## Layout

```text
visin_fusion/models/          public models and config adapter
visin_fusion/engine/          callbacks, training, testing, visualization and stages
visin_fusion/data/            dataset loaders and resolvers
visin_fusion/config/          config schema, presets, manifests and splits
visin_fusion/integrations/    optional Visin callback
configs/                      quickstart.json and example presets
tools/                        sample data, dataset manifests, LiDAR projection
slurms/                       HPC job script
tests/                        unit and end-to-end tests, sample dataset
```

## Tests

```bash
pip install -e '.[train,visin,dev]'
pytest tests/unit                                  # seconds
pytest tests/e2e --device cpu                      # every model x mode, all four stages
pytest tests/e2e --models clftv2 --modes fusion      # one case
pytest tests/e2e --visin online                    # report to the VISIN_TOKEN's project
python tools/coverage.py                               # full-package coverage gate (>90%)
```

The end-to-end tests write configs the way a user does (a preset, the sample dataset, a few
overrides) and run the stage scripts as subprocesses. See `tests/README.md`.

## Checks

CI (`.github/workflows/ci.yml`) runs on every pull request:

- every file compiles (`python -m compileall`)
- `ruff check` and `ruff format --check`, with the rules in `pyproject.toml`
- the unit tests and end-to-end tests on CPU, one job per model (fusion mode; all modes nightly)
- combined coverage for every module in `visin_fusion`, including models and CLI stages, above 90%
- the package: the sdist and wheel build, and the wheel works installed alone, with only the core
  dependencies and with every extra
- the docs build (`mkdocs build --strict`), with the config reference and model diagrams up to date

The docs site and the Docker images are published by releases, not by pushes to `main`.

## Releasing

A release is one run of `.github/workflows/release.yml`, started by hand: Actions -> Release -> Run
workflow, on `main`. Nothing is edited by hand:

1. CI runs; nothing is published that it has not passed.
2. `tools/release.py` picks the version from the [Conventional Commits](https://www.conventionalcommits.org)
   since the last release: a breaking change is major (minor before 1.0), a `feat` is minor, anything
   else a patch. `release-as` overrides it with `patch`, `minor`, `major` or an exact version.
3. It writes `visin_fusion/_version.py` and the `CHANGELOG.md` entry: the `feat`, `fix` and `perf`
   commits, after anything written by hand under `## [Unreleased]`. Then it commits
   `chore(release): <version>` and tags `v<version>`.
4. The package goes to PyPI, the GitHub release gets the changelog entry as its notes, and the docs
   site and Docker images are published for the tag.

`target=testpypi` rehearses a release: the same build goes to TestPyPI, and nothing is committed,
tagged or released. To see what a release would contain without running it:

```bash
python tools/release.py --dry-run
```

One-time setup: add a trusted publisher on PyPI and TestPyPI (project `visin-fusion`, workflow
`release.yml`, environments `pypi` and `testpypi`), create those two environments under the
repository's Settings -> Environments, and set Settings -> Pages -> Source to GitHub Actions. No token
is stored.

## Docs

```bash
pip install --group docs
python tools/make_config_reference.py   # regenerates docs/reference/config.md from the schema
mkdocs serve
```

## Copyable training examples

The model pages and downloadable JSON files are generated from
`configs/examples/<dataset>/<model>/<mode>.json`. After changing an example:

```bash
python tools/make_training_examples.py
python tools/make_training_examples.py --check
```

CI and docs deployment check that the displayed configs and downloads match
their source files. Edit the source configs, rather than the generated pages
in `docs/training/models/` or downloads in `docs/assets/configs/`.

## Architecture diagrams

The six model diagrams are SVGs in `docs/assets/models/`. They stay crisp when zoomed and remain
readable on a phone by scrolling horizontally. Edit the stage summaries and colors in
`tools/generate_model_diagrams.py`, then regenerate and check them:

```bash
python tools/generate_model_diagrams.py
python tools/generate_model_diagrams.py --check
mkdocs build --strict
```

The files are standard vector SVGs and can also be opened in Inkscape. Keep the model pages' paper
references and descriptions aligned with their actual fusion paths.
