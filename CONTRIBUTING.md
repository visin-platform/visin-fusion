# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Set up

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e '.[train,visin,dev]'
pytest tests/unit
pip install pre-commit && pre-commit install   # optional: the CI checks before each commit
```

## Before a pull request

- `ruff check .` and `pytest tests/unit` pass.
- Run the end-to-end test of the models you touched on CPU:
  `pytest tests/e2e --models <model> --modes fusion --device cpu` (a few minutes; the sample dataset is in
  the repo). CI runs every model.
- A change to the config schema (`visin_fusion/config/config_schema.py`) comes with
  `python tools/make_config_reference.py` and an update of the presets in `visin_fusion/config/presets/` if needed.
- Docs in `docs/` change in the same pull request as the behaviour they describe.

## Where things go

See [docs/development.md](docs/development.md) for the layout. In short: one module per stage in
`visin_fusion/engine/stages/<stage>/common.py`, models and their registry in `visin_fusion/models/`, shared engines in `visin_fusion/engine/`, helpers in
`visin_fusion/utils/`, command-line tools in `tools/`.

## Adding a model or a dataset

- A dataset needs no code: put it in the documented layout with a `dataset.json`
  ([docs/datasets.md](docs/datasets.md)).
- A model needs its architecture in `visin_fusion/models/`; in `visin_fusion/visin_fusion/models/registry.py`, how to build it, call it and
  train it (`build_model`, `forward`, `segmentation`, `training_setup`); a preset in `visin_fusion/config/presets/`;
  an entry in `visin_fusion/config/config_schema.py`'s `BACKBONES`; and a line in the e2e tests' `PRESETS`. Every
  stage (`visin_fusion/engine/stages/<stage>/common.py`) and the model tests pick it up from the registry.
