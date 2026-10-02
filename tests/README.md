# Tests

```bash
pip install -e '.[visin,dev]'
pytest                                   # everything
pytest tests/e2e --models clftv2           # one model, all modes
pytest tests/e2e --modes fusion          # one mode, all models
pytest tests/e2e --device cpu            # force CPU (default: cuda:0 if available)
python tools/coverage.py                # unit + fusion end-to-end coverage, >90% of package
```

The end-to-end runs write checkpoints (CLFT's are about 1.3 GB) into pytest's temporary directory,
`/tmp` by default, and keep them only for failed tests. Where `/tmp` is small or held in memory (tmpfs),
point it at a disk: `pytest tests/e2e --basetemp /data/scratch/e2e`.

## End-to-end (`tests/e2e`)

`test_pipelines.py` runs every model (CLFT, CLFTv2, MaskFormer, Mask2Former, DeepLabV3+) in every
mode (rgb, lidar, fusion) through the same four scripts the SLURM jobs run: train (one epoch), test,
visualize and benchmark. Each config extends the model's preset, as a user's would, on the sample dataset
and written to a temporary directory together with all logs, checkpoints and images. A failing stage
shows the end of its output and the path of its full log.

Visin reporting is checked too, chosen with `--visin`:

| `--visin` | What happens |
| --- | --- |
| `offline` (default) | Reports are kept on disk; the test checks they are all there under the run's UUID. No network needed |
| `online` | Reports go to the Visin project of `VISIN_TOKEN` (environment, current-directory `.env`, or `VISIN_ENV_FILE`); the test reads the run back. Use a test project |
| `disabled` | Nothing is reported |

## Sample dataset (`visin_fusion/sample/zod_sample`)

28 ZOD frames downscaled 4x (3.3 MB), in the same layout as the full dataset: `camera/`, `lidar_png/`,
three annotation folders, and split files including all five weather test splits and the
visualization list. Rebuild it with:

```bash
python tools/make_sample_dataset.py --source <unzipped ZOD> --splits <unzipped ZOD> \
    --output visin_fusion/sample/zod_sample   # then: visin-fusion dataset manifest --root visin_fusion/sample/zod_sample ...
```
