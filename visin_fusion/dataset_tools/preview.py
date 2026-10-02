"""Render a dataset's annotations in color, since their pixel values (0, 1, 2, ...) look black.

    visin-fusion dataset preview --root /data/my_dataset                 # 8 training frames into ./preview
    visin-fusion dataset preview --root /data/my_dataset --frames 20 --split val --output previews/

Each image shows the camera frame, its annotation in the training classes' colors, and the two blended,
with a legend of every class and its share of the frame's pixels. Dataset values that no training class
maps are drawn gray: training treats them as background, so a gray area is a labelling choice to check.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visin_fusion.config.dataset_manifest import MANIFEST, apply_manifest
from visin_fusion.logging_setup import configure_logging
from visin_fusion.utils.helpers import get_annotation_path

logger = logging.getLogger(__name__)

UNMAPPED_COLOR = (128, 128, 128)
PANEL_WIDTH = 480


def annotation_colors(train_classes: Sequence[Mapping], size: int = 256) -> np.ndarray:
    """A lookup table from dataset class value to RGB, from the training classes' ``dataset_mapping``."""
    table = np.tile(np.array(UNMAPPED_COLOR, dtype=np.uint8), (size, 1))
    for cls in train_classes:
        for value in cls["dataset_mapping"]:
            table[value] = cls["color"]
    return table


def render_preview(
    camera: np.ndarray, annotation: np.ndarray, train_classes: Sequence[Mapping], alpha: float = 0.5
) -> np.ndarray:
    """Camera | colored annotation | overlay, above a legend, as an RGB array.

    ``camera`` is ``[H, W, 3]`` uint8 and ``annotation`` ``[H, W]`` dataset class values.
    """
    largest = max([int(annotation.max()), *(v for c in train_classes for v in c["dataset_mapping"])])
    table = annotation_colors(train_classes, max(256, largest + 1))
    color = table[annotation]
    blended = np.where((annotation > 0)[..., None], alpha * color + (1 - alpha) * camera, camera).astype(np.uint8)
    height = round(camera.shape[0] * PANEL_WIDTH / camera.shape[1])
    panels = [cv2.resize(p, (PANEL_WIDTH, height), interpolation=cv2.INTER_NEAREST) for p in (camera, color, blended)]
    return np.vstack([np.hstack(panels), _legend(annotation, train_classes, PANEL_WIDTH * 3)])


def _legend(annotation: np.ndarray, train_classes: Sequence[Mapping], width: int) -> np.ndarray:
    entries = [(c["name"], tuple(c["color"]), np.isin(annotation, c["dataset_mapping"]).mean()) for c in train_classes]
    mapped = np.isin(annotation, [v for c in train_classes for v in c["dataset_mapping"]])
    if not mapped.all():
        entries.append(("unmapped", UNMAPPED_COLOR, 1 - mapped.mean()))
    strip = np.full((36, width, 3), 24, dtype=np.uint8)
    x = 10
    for name, rgb, share in entries:
        strip[10:26, x : x + 16] = rgb
        label = f"{name} {share:.1%}"
        cv2.putText(strip, label, (x + 22, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        x += 40 + 9 * len(label)
    return strip


def _frames(config: Mapping, split: str, count: int) -> list[str]:
    dataset = config["Dataset"]
    path = {"train": dataset.get("train_split"), "val": dataset.get("val_split")}.get(split, split)
    with open(path) as f:
        frames = [line.strip() for line in f if line.strip()]
    step = max(1, len(frames) // count)
    return frames[::step][:count]


def main(argv: Sequence[str] | None = None) -> None:
    """Command line entry point: ``visin-fusion dataset preview``."""
    parser = argparse.ArgumentParser(prog="visin-fusion dataset preview", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="dataset root with a dataset.json")
    parser.add_argument("--split", default="train", help="train, val, or a split file (default: train)")
    parser.add_argument("--frames", type=int, default=8, help="how many frames, spread over the split (default: 8)")
    parser.add_argument("--annotation-path", help="annotation folder (default: the manifest's first)")
    parser.add_argument("--output", default="preview", help="folder for the images (default: ./preview)")
    args = parser.parse_args(argv)
    configure_logging()

    overrides = {"dataset_root": args.root}
    if args.annotation_path:
        overrides["annotation_path"] = args.annotation_path
    config = apply_manifest({"Dataset": overrides})
    dataset = config["Dataset"]
    if "train_classes" not in dataset:
        raise SystemExit(f"{args.root} has no {MANIFEST} with train_classes (visin-fusion dataset manifest writes one)")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for frame in _frames(config, args.split, args.frames):
        camera_path = frame if os.path.isabs(frame) else os.path.join(args.root, frame)
        annotation_path = get_annotation_path(camera_path, config)
        if not os.path.exists(annotation_path):
            logger.warning("No annotation %s", annotation_path)
            continue
        camera = np.asarray(Image.open(camera_path).convert("RGB"))
        annotation = np.asarray(Image.open(annotation_path))
        if annotation.ndim == 3:
            annotation = annotation[..., 0]
        if annotation.shape != camera.shape[:2]:
            annotation = cv2.resize(annotation, camera.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        preview = render_preview(camera, annotation, dataset["train_classes"])
        Image.fromarray(preview).save(output / f"{Path(camera_path).stem}_preview.png")
        written += 1
    logger.info("Wrote %d previews to %s", written, output)
