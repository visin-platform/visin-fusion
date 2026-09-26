"""Uploading the visualize_*.py scripts' rendered frames to Visin.

    run = attach_to_training(config['Log']['logdir'])
    for image_name in images:
        ...render...
        queue_visualizations(run, epoch, epoch_uuid, output_base, image_name)
    run.finish()      # waits for the uploads and prints what was sent

Uploads go out in the background while the next image renders. Each file is
copied at the call, so the next image may reuse the same output paths.
"""

from __future__ import annotations

import os
from typing import Any

from integrations.vision_service import parse_checkpoint_name

# The kinds each visualize_*.py script renders, one subdirectory each.
VISUALIZATION_KINDS = ("segment", "overlay", "compare", "correct_only")


def queue_visualizations(run: Any, epoch: int, epoch_uuid: str, output_base: str, image_name: str) -> int:
    """Queue every kind rendered for one image. Returns how many were found to send."""
    queued = 0
    for kind in VISUALIZATION_KINDS:
        path = os.path.join(output_base, kind, image_name)
        if not os.path.exists(path):
            print(f"Skipping {kind} - file not found: {path}")
            continue
        run.upload_visualization(
            epoch,
            path,
            kind,
            epoch_uuid=epoch_uuid,
            metadata={"image_name": image_name, "source": "visualization_script"},
        )
        queued += 1
    return queued


def get_epoch_uuid_from_model_path(model_path: str) -> str | None:
    """The epoch UUID in a checkpoint's file name (epoch_{num}_{uuid}.pth), if it has one."""
    return parse_checkpoint_name(model_path)[1]
