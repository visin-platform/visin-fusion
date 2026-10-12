"""Uploading the stages/visualize/*.py scripts' rendered frames to Visin.

    run = attach_to_training(config['Log']['logdir'])
    for image_name in images:
        ...render...
        queue_visualizations(run, epoch, epoch_uuid, output_base, image_name)
    run.finish()      # waits for the uploads and prints what was sent

Uploads go out in the background while the next image renders. Each file is
copied at the call, so the next image may reuse the same output paths.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# The kinds each stages/visualize/*.py script renders, one subdirectory each.
VISUALIZATION_KINDS = ("segment", "mask", "confidence", "overlay", "compare", "correct_only")


def queue_visualizations(
    run: Any, epoch: int, epoch_uuid: str, output_base: str, image_name: str, *, predicted_on: Mapping | None = None
) -> int:
    """Queue every kind rendered for one image. Returns how many were found to send."""
    queued = 0
    for kind in VISUALIZATION_KINDS:
        path = os.path.join(output_base, kind, image_name)
        if not os.path.exists(path):
            logger.warning("Skipping %s - file not found: %s", kind, path)
            continue
        metadata: dict[str, Any] = {"image_name": image_name, "source": "visualization_script"}
        if kind in ("mask", "segment", "confidence"):
            with Image.open(path) as image:
                width, height = image.size
            prediction = {"frame": Path(image_name).stem, "width": width, "height": height}
            if predicted_on:
                prediction["dataset"] = dict(predicted_on)
            if kind == "confidence":
                prediction["confidenceEncoding"] = "uint8"
            metadata["prediction"] = prediction
        run.upload_visualization(
            epoch,
            path,
            kind,
            epoch_uuid=epoch_uuid,
            metadata=metadata,
        )
        queued += 1
    return queued


def prediction_dataset(path: str | Path) -> dict[str, str] | None:
    """The Visin dataset marker surrounding a supplied frame list, separate from the training dataset."""
    path = Path(path).resolve()
    for folder in (path.parent, *path.parent.parents):
        marker = folder / ".visin-dataset.json"
        if not marker.is_file():
            continue
        try:
            data = json.loads(marker.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("id"), str) or not data["id"]:
                raise ValueError("dataset marker needs an id")
            if not data.get("archiveRevision") and data.get("file") and isinstance(data.get("revision"), str):
                data["archiveRevision"] = data["revision"]
            return {
                "source": "visin",
                "id": data["id"],
                **{
                    key: data[key]
                    for key in ("revision", "archiveRevision")
                    if isinstance(data.get(key), str) and data[key]
                },
            }
        except (OSError, ValueError) as exc:
            logger.warning("Could not read prediction provenance from %s: %s", marker, exc)
            return None
    return None
