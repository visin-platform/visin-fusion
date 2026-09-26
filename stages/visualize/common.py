#!/usr/bin/env python3
"""Render any model's predictions on a list of frames: segment, overlay, compare and correct_only images.

The model is the best checkpoint in Log.logdir, built through models/registry.py. Frames default to the
dataset's visualization split. With --upload, the images go to Visin on the checkpoint's epoch.

    python -m stages.visualize.common -c <config.json> [-p <frames.txt | image.png>] [--upload]
"""
import argparse
import os
import sys

import torch

from core.data_loader import DataLoader as InferenceDataLoader
from core.visualizer import Visualizer
from integrations.vision_service import attach_to_training, parse_checkpoint_name
from integrations.visualization_uploader import queue_visualizations
from models.registry import build_model, segment
from utils.config import load_config
from utils.helpers import get_annotation_path, get_device, get_lidar_path, get_model_path
from utils.splits import visualization_split


def load_frames(path):
    """Frames to render: one image, or the lines of a split file."""
    if path.endswith(('.png', '.jpg', '.jpeg')):
        return [path]
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_model(config, checkpoint, device):
    """The model with the checkpoint's weights (random init only: the weights come from the checkpoint)."""
    model = build_model(config, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict']
    model.load_state_dict(state)  # strict: a checkpoint for another model must not load
    return model.to(device).eval()


def render(config, frames, model_path, output_dir, upload=False, epoch_uuid=None):
    """Render every frame; returns the number that failed."""
    device = get_device(config)
    model = load_model(config, model_path, device)
    data_loader, visualizer = InferenceDataLoader(config), Visualizer(config, output_dir)
    dataroot = os.path.abspath(config['Dataset']['dataset_root'])
    mode = config['CLI']['mode']

    epoch, checkpoint_uuid = parse_checkpoint_name(model_path)
    epoch_uuid = epoch_uuid or checkpoint_uuid
    run = attach_to_training(config['Log']['logdir'], epoch_uuid=epoch_uuid) if upload and epoch_uuid else None
    if upload and not epoch_uuid:
        print(f"Warning: no epoch UUID in the checkpoint name {model_path}; not uploading")

    failed = queued = 0
    for index, frame in enumerate(frames, 1):
        cam_path = frame if os.path.isabs(frame) else os.path.join(dataroot, frame)
        anno_path = get_annotation_path(cam_path, config)
        print(f"Rendering {index}/{len(frames)}: {os.path.basename(cam_path)}")
        try:
            rgb = data_loader.load_rgb(cam_path).to(device).unsqueeze(0)
            lidar = rgb if mode == 'rgb' else data_loader.load_lidar(get_lidar_path(cam_path, config)).to(device).unsqueeze(0)
            with torch.no_grad():
                prediction = segment(model, config, rgb, lidar)
            visualizer.visualize_prediction(prediction, cam_path, anno_path, index)
        except Exception as e:  # keep rendering the other frames; the run still fails at the end
            print(f"Failed to render {cam_path}: {type(e).__name__}: {e}")
            failed += 1
            continue
        if run is not None and queue_visualizations(run, epoch or 0, epoch_uuid, output_dir, os.path.basename(cam_path)):
            queued += 1

    if run is not None:
        run.finish()  # waits for the queued uploads; visin logs any that failed
        print(f"Queued for upload: {queued}/{len(frames)}")
    print(f"Rendered {len(frames) - failed}/{len(frames)} frames to {output_dir}")
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-c', '--config', required=True, help='config file')
    parser.add_argument('-p', '--path', help='an image, or a file listing frames (default: the visualization split)')
    parser.add_argument('--upload', action='store_true', help="upload the images to Visin on the checkpoint's epoch")
    parser.add_argument('--epoch-uuid', help="epoch to upload to (default: the checkpoint's)")
    parser.add_argument('--output-dir', '--output_dir', help='where to write (default: <Log.logdir>/visualizations)')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_path = get_model_path(config, best=True)
    if not model_path:
        sys.exit(f"No checkpoint in {config['Log']['logdir']}/checkpoints; train first")
    print(f"Using model: {model_path}")
    frames_file = args.path or visualization_split(config)
    frames = load_frames(frames_file)
    print(f"{len(frames)} frames from {frames_file}")

    output_dir = args.output_dir or os.path.join(config['Log']['logdir'].rstrip('/'), 'visualizations')
    failed = render(config, frames, model_path, output_dir, args.upload, args.epoch_uuid)
    if failed:
        sys.exit(f"{failed} of {len(frames)} frames failed to render")


if __name__ == '__main__':
    main()
