#!/usr/bin/env python3
"""Write dataset.json for a dataset in this repo's layout (see utils/dataset_manifest.py).

Split files and annotation folders are found in the dataset. The name, classes, suggested training
classes and LiDAR normalization come from a config for the dataset: one naming ``Dataset.name``,
``dataset_classes`` and ``train_classes`` (and optionally ``transforms.lidar_mean`` / ``lidar_std``,
e.g. from tools/dataset_stats.py). Every frame the splits list is checked for a camera image, a LiDAR
projection and each annotation.

    python tools/make_manifest.py --root /data/my_dataset --config my_dataset_config.json \\
        --description "My dataset, 1,000 labelled frames"
"""
import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset_manifest import FORMAT, MANIFEST  # noqa: E402
from utils.helpers import replace_camera_folder  # noqa: E402
from utils.splits import VISUALIZATION_SPLITS, WEATHER_TEST_SPLITS  # noqa: E402

TRAIN_SPLITS = ('train.txt', 'train_all.txt')
VAL_SPLITS = ('validation.txt', 'early_stop_valid.txt')
LAYOUT = {'camera': 'camera', 'lidar': 'lidar_png'}


def find_split_dir(root):
    """The directory holding the split files: the root itself, or one level down (e.g. splits_clft/)."""
    candidates = ['.'] + sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for candidate in candidates:
        if any(os.path.exists(os.path.join(root, candidate, name)) for name in TRAIN_SPLITS):
            return candidate
    raise SystemExit(f"no train split ({' or '.join(TRAIN_SPLITS)}) in {root} or its subfolders")


def first_existing(root, split_dir, names):
    """The first of ``names`` in the split directory that lists any frames (empty files are skipped)."""
    for name in names:
        path = os.path.join(root, split_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return os.path.normpath(os.path.join(split_dir, name))
    return None


def read_split(root, path):
    with open(os.path.join(root, path)) as f:
        return [line.strip() for line in f if line.strip()]


def find_annotations(root, frame, preferred):
    """Annotation folders beside a frame's camera folder that have that frame as a label map.

    Folders of colour renderings (e.g. annotation_visualized) are skipped: labels are
    single-channel images of class ids.
    """
    camera_dir = os.path.dirname(os.path.join(root, frame))
    parent, name = os.path.dirname(camera_dir), os.path.basename(frame)

    def is_label_map(folder):
        path = os.path.join(parent, folder, name)
        return os.path.exists(path) and Image.open(path).mode in ('L', 'P', 'I', 'I;16')

    found = sorted(d for d in os.listdir(parent) if d.startswith('annotation') and is_label_map(d))
    if preferred in found:  # the config's annotation first: it becomes the default
        found.remove(preferred)
        found.insert(0, preferred)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', required=True, help='dataset root (holds camera/, lidar_png/, ...)')
    parser.add_argument('--config', required=True, help='an existing config for this dataset')
    parser.add_argument('--description', default='', help='one line about the dataset')
    parser.add_argument('--output', help=f'where to write the manifest (default: <root>/{MANIFEST})')
    args = parser.parse_args()

    with open(args.config) as f:
        dataset = json.load(f)['Dataset']
    root = os.path.abspath(args.root)
    split_dir = find_split_dir(root)

    splits = {
        'train': first_existing(root, split_dir, TRAIN_SPLITS),
        'val': first_existing(root, split_dir, VAL_SPLITS),
        'test': {name: path for name, file in WEATHER_TEST_SPLITS.items()
                 if (path := first_existing(root, split_dir, [file]))},
        'visualization': first_existing(root, split_dir, VISUALIZATION_SPLITS),
    }
    if not splits['val']:
        raise SystemExit(f"no validation split ({' or '.join(VAL_SPLITS)}) in {os.path.join(root, split_dir)}")
    splits = {key: value for key, value in splits.items() if value}

    train = read_split(root, splits['train'])
    annotations = find_annotations(root, train[0], dataset.get('annotation_path') or 'annotation')
    if not annotations:
        raise SystemExit(f"no annotation folder beside the camera folder of {train[0]}")

    # Every listed frame has its camera image, LiDAR projection and annotations
    frames = sorted({frame for path in [splits['train'], splits['val'], *splits.get('test', {}).values(),
                                        *([splits['visualization']] if 'visualization' in splits else [])]
                     for frame in read_split(root, path)})
    missing = []
    for frame in frames:
        cam = os.path.join(root, frame)
        for path in [cam, replace_camera_folder(cam, LAYOUT['lidar'])] + \
                    [replace_camera_folder(cam, folder) for folder in annotations]:
            if not os.path.exists(path):
                missing.append(os.path.relpath(path, root))
    if missing:
        print(f"Warning: {len(missing)} files listed by the splits are missing, e.g. {missing[:3]}")

    transforms = dataset.get('transforms', {})
    manifest = {
        'format': FORMAT,
        'name': dataset['name'],
        'description': args.description,
        'layout': LAYOUT,
        'annotations': annotations,
        'splits': splits,
        'classes': dataset['dataset_classes'],
        'train_classes': dataset['train_classes'],
        'normalization': {key: transforms[key] for key in ('lidar_mean', 'lidar_std') if transforms.get(key)},
    }
    output = args.output or os.path.join(root, MANIFEST)
    with open(output, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print(f"Wrote {output}: {len(frames)} frames, splits {list(splits)}, annotations {annotations}")


if __name__ == '__main__':
    main()
