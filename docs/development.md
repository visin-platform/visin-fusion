# Development

## Layout

```text
visin_fusion/models/          public models and config adapter
visin_fusion/engine/          callbacks, training, testing, visualization and stages
visin_fusion/data/            dataset loaders and resolvers
visin_fusion/config/          config schema, presets, manifests and splits
visin_fusion/integrations/    optional Visin callback
run.py                        compatibility launcher for existing jobs
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
- `ruff check` with the rules in `ruff.toml`
- the unit tests and end-to-end tests on CPU, one job per model (fusion mode; all modes nightly)
- combined coverage for every module in `visin_fusion`, including models and CLI stages, above 90%

`.github/workflows/image.yml` builds the Docker images, and `.github/workflows/docs.yml` publishes this
site.

## Releasing

A version tag publishes the package to PyPI, a GitHub release and the Docker images. Once CI passes on
`main`:

```bash
git tag v1.2.3 && git push origin v1.2.3
```

The version comes from the tag (setuptools-scm), so nothing in the repository is edited. Between
releases, a checkout reports a development version such as `1.2.4.dev3+g1a2b3c4`.

`.github/workflows/release.yml` builds the sdist and wheel, installs the wheel with only the core
dependencies and with every extra, runs the unit tests, then publishes to PyPI and creates the GitHub
release. Its notes are the `## 1.2.3` section of `CHANGELOG.md` if there is one, otherwise GitHub's notes
generated from the pull requests and commits since the previous tag. Tags like `v1.2.0rc1` are marked as
pre-releases.

PyPI publishing uses a trusted publisher, so no token is stored. Set it up once on PyPI (project
`visin-fusion`, workflow `release.yml`, environment `pypi`) and create the `pypi` environment under the
repository's Settings -> Environments.

## Docs

```bash
pip install --group docs
python tools/make_config_reference.py   # regenerates docs/reference/config.md from the schema
mkdocs serve
```

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
