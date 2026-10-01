# Prepare Iseauto

[Install Fusion](../setup.md) first. Run from the checkout with `.venv` active.

## Download and prepare

Copy this block into your terminal:

```bash title="Prepare Iseauto"
export VISIN_DATASET_URL=https://dataset-api.visin.eu
export VISIN_DATA_DIR=./data
ISEAUTO_DATA_DIR="$(visin download iseauto)"
export ISEAUTO_DATA_DIR
if [ ! -f "$ISEAUTO_DATA_DIR/dataset.json" ]; then
  python tools/make_manifest.py \
    --root "$ISEAUTO_DATA_DIR" \
    --config configs/datasets/iseauto.json
fi
```

The download prints the extracted folder; progress goes to stderr. It resumes
interrupted downloads, unpacks under `data/`, and deletes the ZIP after successful
extraction. Completed data is reused. Allow room for the ZIP and extracted files.
Public datasets need no token; export your credential first for private ones.

The manifest command handles archives without `dataset.json`. It uses this
dataset's baseline label mappings, weights, and LiDAR normalization; it does
not recompute statistics. Iseauto uses the `annotation` folder and three classes: background, vehicle, and human. The SDK returns the inner `xod_dataset/` folder. The manifest tool recognizes `visualization.txt` and skips the empty snow split.

| Training frames | Validation frames | Test splits | Visualization frames |
| --- | --- | --- | --- |
| 1,200 | 600 | 4 | 12 |

If manifest generation reports missing files, inspect the affected entries.
Missing annotations become empty masks during training. Restore them or remove
the frame from the affected split before an accuracy experiment. Warnings for
unused annotation folders do not affect your chosen config.

## First run

```bash title="One epoch: CLFTv2 fusion"
visin-fusion run -c configs/examples/iseauto/clftv2/fusion.json --upload --benchmark-device cuda
```

Use `--benchmark-device cpu` on a CPU host. Outputs go to
`logs/iseauto/clftv2/fusion/`. [Configure Visin](../setup.md#optional-connect-to-visin)
to upload the run, metrics, and images. One epoch is a pipeline check.

In a new terminal, activate `.venv` and repeat the preparation block; downloaded
files are reused.

**Next:** [Choose another model or mode](../../training-examples.md).
See [Datasets](../../datasets.md) for layout, normalization, and your own data.
