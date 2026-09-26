## What and why

## Checks

- [ ] `pytest tests/unit` passes
- [ ] End-to-end on the affected models: `pytest tests/e2e --models <model> --modes fusion --device cpu`
- [ ] Docs updated (`docs/`); if the config schema changed, `python tools/make_config_reference.py`
- [ ] No private paths, tokens or personal notes in the diff
