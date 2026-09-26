# Development

## Layout

```text
run.py          run a config through its pipeline
stages/         train/, test/, visualize/, benchmark/: common.py for every model
models/         the architectures; registry.py builds, calls and trains each
core/           training and testing engines, metrics, model builders, dataset loader
utils/          config loading and schema, dataset manifests, split files, metrics
integrations/   reporting to Visin
configs/        presets/, quickstart.json, experiments/ (past experiments)
tools/          sample dataset, dataset manifests, LiDAR projection
slurms/         run.slurm: a config through run.py on a SLURM cluster
tests/          unit/ and e2e/ tests, the sample dataset
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/unit                                  # seconds
pytest tests/e2e --device cpu                      # every model x mode, all four stages
pytest tests/e2e --models swin --modes fusion      # one case
pytest tests/e2e --visin online                    # report to the VISIN_TOKEN's project
```

The end-to-end tests write configs the way a user does (a preset, the sample dataset, a few
overrides) and run the stage scripts as subprocesses. See `tests/README.md`.

## Checks

CI (`.github/workflows/ci.yml`) runs on every pull request:

- every file compiles (`python -m compileall`)
- `ruff check` with the rules in `ruff.toml`
- the unit tests, with coverage
- the end-to-end tests on CPU, one job per model (fusion mode; all modes nightly)

`.github/workflows/image.yml` builds the Docker images, and `.github/workflows/docs.yml` publishes this
site.

## Docs

```bash
pip install -r requirements-docs.txt
python tools/make_config_reference.py   # regenerates docs/reference/config.md from the schema
mkdocs serve
```
