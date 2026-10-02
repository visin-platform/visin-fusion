#!/usr/bin/env python3
"""Render any model's predictions on a list of frames: segment, overlay, compare and correct_only images.

The model is the best checkpoint in Log.logdir, built through visin_fusion/models/registry.py. Frames default to the
dataset's visualization split. With --upload, the images go to Visin on the checkpoint's epoch.

    python -m visin_fusion.engine.stages.visualize.common -c <config.json> [-p <frames.txt | image.png>] [--upload]
"""

import argparse
import logging
import os
import sys

import torch

from visin_fusion.config.config import load_config
from visin_fusion.config.splits import visualization_split
from visin_fusion.data.preprocessing import Preprocessor
from visin_fusion.engine.callbacks import RunEnd, Visualization, configured_callbacks
from visin_fusion.engine.epoch_ids import parse_checkpoint_name
from visin_fusion.engine.visualizer import Visualizer
from visin_fusion.logging_setup import configure_logging
from visin_fusion.models.registry import from_config
from visin_fusion.utils.helpers import get_annotation_path, get_device, get_lidar_path, get_model_path

logger = logging.getLogger(__name__)


def load_frames(path):
    """Frames to render: one image, or the lines of a split file."""
    if path.endswith((".png", ".jpg", ".jpeg")):
        return [path]
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_model(config, checkpoint, device):
    """The model with the checkpoint's weights (random init only: the weights come from the checkpoint)."""
    model = from_config(config, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(state)  # strict: a checkpoint for another model must not load
    return model.to(device).eval()


def render(config, frames, model_path, output_dir, upload=False, epoch_uuid=None):
    """Render every frame; returns the number that failed."""
    device = get_device(config)
    model = load_model(config, model_path, device)
    preprocessor, visualizer = Preprocessor.from_config(config), Visualizer(config, output_dir)
    dataroot = os.path.abspath(config["Dataset"]["dataset_root"])
    mode = config["CLI"]["mode"]

    epoch, checkpoint_uuid = parse_checkpoint_name(model_path)
    epoch_uuid = epoch_uuid or checkpoint_uuid
    uploading = bool(upload and epoch_uuid)
    events = configured_callbacks(config, visin=uploading)
    if upload and not epoch_uuid:
        logger.warning("No epoch UUID in the checkpoint name %s; not uploading", model_path)

    failed = queued = 0
    for index, frame in enumerate(frames, 1):
        cam_path = frame if os.path.isabs(frame) else os.path.join(dataroot, frame)
        anno_path = get_annotation_path(cam_path, config)
        logger.info("Rendering %s/%s: %s", index, len(frames), os.path.basename(cam_path))
        try:
            rgb = preprocessor.load_rgb(cam_path).to(device).unsqueeze(0)
            lidar = (
                rgb
                if mode == "rgb"
                else preprocessor.load_lidar(get_lidar_path(cam_path, config)).to(device).unsqueeze(0)
            )
            with torch.no_grad():
                prediction = model(rgb, lidar)
            visualizer.visualize_prediction(prediction, cam_path, anno_path, index)
        except Exception as e:  # keep rendering the other frames; the run still fails at the end
            logger.error("Failed to render %s: %s: %s", cam_path, type(e).__name__, e)
            failed += 1
            continue
        events.emit(
            Visualization(
                config=config,
                epoch=epoch or 0,
                epoch_uuid=epoch_uuid,
                output_dir=output_dir,
                image_name=os.path.basename(cam_path),
            )
        )
        queued += uploading and bool(events.callbacks)

    events.emit(RunEnd(config=config))
    if uploading:
        logger.info("Queued for upload: %s/%s", queued, len(frames))
    logger.info("Rendered %s/%s frames to %s", len(frames) - failed, len(frames), output_dir)
    return failed


def main(argv=None):
    """Entry point of the visualize stage (``python -m visin_fusion.engine.stages.visualize.common``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True, help="config file")
    parser.add_argument("-p", "--path", help="an image, or a file listing frames (default: the visualization split)")
    parser.add_argument("--upload", action="store_true", help="upload the images to Visin on the checkpoint's epoch")
    parser.add_argument("--epoch-uuid", help="epoch to upload to (default: the checkpoint's)")
    parser.add_argument("--output-dir", "--output_dir", help="where to write (default: <Log.logdir>/visualizations)")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    model_path = get_model_path(config, best=True)
    if not model_path:
        sys.exit(f"No checkpoint in {config['Log']['logdir']}/checkpoints; train first")
    logger.info("Using model: %s", model_path)
    frames_file = args.path or visualization_split(config)
    frames = load_frames(frames_file)
    logger.info("%s frames from %s", len(frames), frames_file)

    output_dir = args.output_dir or os.path.join(config["Log"]["logdir"].rstrip("/"), "visualizations")
    failed = render(config, frames, model_path, output_dir, args.upload, args.epoch_uuid)
    if failed:
        sys.exit(f"{failed} of {len(frames)} frames failed to render")


if __name__ == "__main__":
    main()
