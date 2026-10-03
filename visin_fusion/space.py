"""A Gradio demo of a published model, as a Hugging Face Space: ``visin-fusion space``.

    visin-fusion space --model hf://org/clftv2-zod --space org/clftv2-zod-demo

Creates the Space (public unless ``--private``) with an ``app.py`` that loads the model with
``Predictor.from_pretrained``, pinned to the commit the model has now, and shows the predicted mask over
the camera image. Anyone with the link can try the model without installing anything or having a Visin
account; the Space runs on the Hub's free CPU hardware, so expect seconds per image.

A Space reads its model with its own credentials. For a private model repo, add an ``HF_TOKEN`` secret to
the Space in its settings, or the Space cannot load the model.

Your Hugging Face token (``HF_TOKEN`` or ``huggingface-cli login``) creates the Space and is never stored.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from visin_fusion import __version__
from visin_fusion.hub import HubRef, huggingface_module, parse_ref

logger = logging.getLogger(__name__)

APP = '''"""Try {repo}: upload a camera image (and a LiDAR projection, if the model reads one)."""

import gradio as gr

from visin_fusion.inference import Predictor

MODEL = {model}
predictor = Predictor.from_pretrained(MODEL, device="cpu")


SWATCH = '<span style="display:inline-block;width:1em;height:1em;background:rgb({{}},{{}},{{}})"></span> {{}}&emsp;'


def legend():
    classes = zip(predictor.class_names, predictor.palette.tolist(), strict=True)
    return "<p>" + "".join(SWATCH.format(*color, name) for name, color in classes) + "</p>"


def run(camera, lidar):
    if predictor.mode != "lidar" and camera is None:
        raise gr.Error("This model needs a camera image.")
    if predictor.mode != "rgb" and lidar is None:
        raise gr.Error("This model also needs a LiDAR projection image.")
    rgb = None if predictor.mode == "lidar" else camera
    mask = predictor.predict(rgb, None if predictor.mode == "rgb" else lidar)
    return predictor.overlay(camera, mask) if camera is not None else predictor.colorize(mask)


demo = gr.Interface(
    run,
    [
        gr.Image(type="pil", label="Camera image"),
        gr.Image(type="pil", label="LiDAR projection (if the model uses one)"),
    ],
    gr.Image(label="Prediction"),
    title="{repo}",
    description="Semantic segmentation with a model trained with Visin. Mode: " + predictor.mode + ". " + legend(),
)

if __name__ == "__main__":
    demo.launch()
'''

README = """---
title: {title}
emoji: 🛰️
sdk: gradio
app_file: app.py
pinned: false
models:
  - {repo}
---

A demo of [{repo}](https://huggingface.co/{repo}), written by `visin-fusion space`.
"""


def build_space(model: HubRef, destination: str | Path) -> Path:
    """Write the files of a demo Space for ``model`` (which should name its commit) into ``destination``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pinned = f"hf://{model.repo}" + (f"@{model.revision}" if model.revision else "")
    if model.filename:
        pinned += f"/{model.filename}"
    (destination / "app.py").write_text(APP.format(repo=model.repo, model=json.dumps(pinned)))
    (destination / "requirements.txt").write_text(f"visin-fusion[hf]>={__version__}\ngradio\n")
    (destination / "README.md").write_text(README.format(repo=model.repo, title=model.repo.split("/")[1]))
    return destination


def publish_space(model_ref: str, space_repo: str, *, private: bool = False) -> str:
    """Create the Space ``space_repo`` for the model ``model_ref`` and return its URL.

    A model named without a commit is pinned to its newest one, so the demo keeps showing the model it was
    made for if the repo changes later.
    """
    model = parse_ref(model_ref)
    hub = huggingface_module()
    api = hub.HfApi()
    if not model.revision:
        model = HubRef(model.repo, api.model_info(model.repo).sha, model.filename)
    with tempfile.TemporaryDirectory() as staging:
        folder = build_space(model, staging)
        api.create_repo(repo_id=space_repo, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
        api.upload_folder(repo_id=space_repo, repo_type="space", folder_path=str(folder))
    logger.info("Created the Space %s for %s", space_repo, model.repo)
    return f"https://huggingface.co/spaces/{space_repo}"
