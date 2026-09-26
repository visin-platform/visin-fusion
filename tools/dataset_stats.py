#!/usr/bin/env python3
"""Statistics of a dataset's training split: LiDAR normalization and class frequencies.

    python tools/dataset_stats.py --root /data/my_dataset            # print them
    python tools/dataset_stats.py --root /data/my_dataset --write    # also store them in dataset.json

- LiDAR normalization (``lidar_mean`` / ``lidar_std``) is computed over the pixels that have a LiDAR
  point, on 0..1 pixel values, the way the existing configs' values were computed.
- Class frequencies are counted after merging dataset classes into the training classes
  (``train_classes`` and their ``dataset_mapping``), and suggest loss weights by median frequency
  balancing (weight = median frequency / class frequency). The existing configs' weights were
  tuned by hand; treat these as a starting point.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset_manifest import MANIFEST, apply_manifest  # noqa: E402
from utils.helpers import get_annotation_path, get_lidar_path  # noqa: E402


def lidar_statistics(paths):
    """Mean and std of the 0..1 values of pixels with a point, per channel."""
    count, total, squares = 0, np.zeros(3), np.zeros(3)
    for path in paths:
        pixels = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64).reshape(-1, 3) / 255
        pixels = pixels[pixels.sum(axis=1) > 0]
        count += len(pixels)
        total += pixels.sum(axis=0)
        squares += (pixels ** 2).sum(axis=0)
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(squares / max(count, 1) - mean ** 2, 0))
    return mean.round(5).tolist(), std.round(5).tolist()


def class_frequencies(paths, train_classes):
    """Pixel share of each training class, with dataset classes merged as train_classes says."""
    lookup = np.full(256, -1, dtype=np.int64)
    for train_class in train_classes:
        for dataset_index in train_class['dataset_mapping']:
            lookup[dataset_index] = train_class['index']
    counts = np.zeros(len(train_classes), dtype=np.int64)
    for path in paths:
        labels = lookup[np.asarray(Image.open(path)).astype(np.int64).ravel()]
        counts += np.bincount(labels[labels >= 0], minlength=len(train_classes))
    return counts / max(counts.sum(), 1)


def suggested_weights(frequencies):
    present = frequencies[frequencies > 0]
    median = np.median(present) if len(present) else 0
    return [round(float(median / f), 3) if f > 0 else 0.0 for f in frequencies]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', required=True, help='dataset root with a dataset.json')
    parser.add_argument('--write', action='store_true', help='store normalization and weights in dataset.json')
    args = parser.parse_args()

    config = apply_manifest({'Dataset': {'dataset_root': args.root}})
    dataset = config['Dataset']
    if 'train_split' not in dataset:
        sys.exit(f"{args.root} has no {MANIFEST} with a train split (tools/make_manifest.py writes one)")
    with open(dataset['train_split']) as f:
        cams = [os.path.join(args.root, line.strip()) for line in f if line.strip()]
    print(f"{len(cams)} training frames")

    lidar = [p for p in (get_lidar_path(c, config) for c in cams) if os.path.exists(p)]
    lidar_mean, lidar_std = lidar_statistics(lidar)
    print(f"lidar_mean {lidar_mean}\nlidar_std  {lidar_std}   ({len(lidar)} LiDAR images)")

    train_classes = dataset.get('train_classes')
    if train_classes:
        annotations = [p for p in (get_annotation_path(c, config) for c in cams) if os.path.exists(p)]
        frequencies = class_frequencies(annotations, train_classes)
        weights = suggested_weights(frequencies)
        print(f"\n{'class':<14}{'pixels':>9}{'weight now':>12}{'suggested':>11}")
        for cls, freq, weight in zip(train_classes, frequencies, weights):
            print(f"{cls['name']:<14}{freq:>9.2%}{cls.get('weight', 1.0):>12}{weight:>11}")

    if args.write:
        path = os.path.join(args.root, MANIFEST)
        with open(path) as f:
            manifest = json.load(f)
        manifest['normalization'] = {'lidar_mean': lidar_mean, 'lidar_std': lidar_std}
        if train_classes and manifest.get('train_classes'):
            for cls, weight in zip(manifest['train_classes'], weights):
                cls['weight'] = weight
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
            f.write('\n')
        print(f"\nWrote normalization{' and class weights' if train_classes else ''} to {path}")


if __name__ == '__main__':
    main()
