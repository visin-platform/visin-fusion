# Hugging Face

Trained models and datasets can live on the [Hugging Face Hub](https://huggingface.co) instead of on one
machine's disk. Visin keeps a pointer to the exact commit a run used; the bytes stay on the Hub.

```bash
pip install 'visin-fusion[hf]'
export HF_TOKEN=hf_...      # or huggingface-cli login; only for private or gated repos, and to publish
```

Your own token is read by `huggingface_hub`. Nothing in this package stores it or sends it to Visin.

## Use a published model

```python
from visin_fusion.inference import Predictor

predictor = Predictor.from_pretrained("hf://org/clftv2-zod@3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433")
mask = predictor.predict("camera/000001.png", "lidar_png/000001.png")
```

```bash
visin-fusion predict --checkpoint 'hf://org/clftv2-zod@<commit>' --input camera/ --output out/
```

A reference is `hf://org/name`, optionally `@<commit>` and a trailing file or folder name. Pin a commit: a branch
moves, so a result that names one is not reproducible. The repo is read for `checkpoint.pth`, or for its
only `.pth`/`.pt` file; name the file (`hf://org/name/best.pth`) when there are several. The download is
cached by the Hub client. A folder reference such as `hf://org/name@<commit>/runs/one` selects
`checkpoint.pth` or the only `.pth`/`.pt` file inside that folder.

## Publish the best checkpoint after training

```json
"General": {"hub_repo": "org/clftv2-zod", "hub_private": true}
```

When the train stage finishes, the checkpoint with the best validation mIoU goes to that model repo as
`checkpoint.pth` (weights and `model_info`, without the optimizer state) and `config.json` (the same
`model_info`). The repo is created private unless `hub_private` is `false`. A failed upload is logged and
never fails the run: the checkpoints are still on disk.

With Visin reporting on and a `visin` that can link models, the upload goes through `Run.log_model`: the run
shows `org/clftv2-zod @ 3f2a1c9` with a link, and Visin writes the repo's model card, with the scores as the
Hub's `model-index`, when the repo has no README. The project's storage must be set to Hugging Face in
Visin, or the link is refused.

## Train on a Hub dataset

```json
"Dataset": {"dataset_root": "hf:org/zod-png@3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"}
```

The repo must hold a `dataset.json` manifest at its root ([Datasets](datasets.md)). It is downloaded into the
Hub cache on first use. An unpinned `hf:org/zod-png` works but logs a warning, for the reason above.

For a dataset Visin lists and has pointed at the Hub, `"visin:zod"` fetches it from the Hub at the commit
Visin names, and falls back to Visin's own zip when it has one. `visin push zod --repo org/zod-png` (from
the `visin` package) publishes a Visin dataset to the Hub.

## A demo anyone can try

```bash
visin-fusion space --model hf://org/clftv2-zod --space org/clftv2-zod-demo
```

Creates a public [Space](https://huggingface.co/docs/hub/spaces) (`--private` for a private one) with a Gradio
app: upload a camera image, and a LiDAR projection if the model reads one, and see the predicted mask over
it. The demo loads the model at the commit it has now, so it keeps showing that model if the repo changes.
File and folder paths in the model reference are preserved, so a demo uses the same checkpoint as
the platform's model-loading snippets. It runs on the Hub's free CPU hardware, a few seconds per image. For a private model repo, add an `HF_TOKEN`
secret to the Space in its settings, or the Space cannot load the model.
