#!/usr/bin/env python3
"""Build the small sample dataset used by the end-to-end tests and the quick start.

Takes a few frames from each split of a full dataset in this repo's layout (camera/,
lidar_png/, annotation folders and split .txt files), downscales them, and writes the
same layout with its own split files, so any config can point at it unchanged.

    python tools/make_sample_dataset.py --source /data/zod_dataset --splits /data/zod_dataset \
        --output tests/data/zod_sample

Camera images are resized bilinearly; annotations and LiDAR projections with nearest
neighbour, so class ids and encoded LiDAR values are kept as they are.
"""
import argparse
import os

from PIL import Image

# Frames taken from each split file. Test splits are small on purpose: they only
# have to exercise every weather condition.
FRAMES_PER_SPLIT = {
    'train.txt': 8,
    'validation.txt': 4,
    'test_day_fair.txt': 2,
    'test_day_rain.txt': 2,
    'test_night_fair.txt': 2,
    'test_night_rain.txt': 2,
    'test_snow.txt': 2,
    'visualizations.txt': 8,
}
ANNOTATION_FOLDERS = ['annotation_camera_only', 'annotation_lidar_only', 'annotation_fusion']


def read_split(path, count):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()][:count]


def resize(src, dst, scale, resample):
    image = Image.open(src)
    size = (max(1, round(image.width / scale)), max(1, round(image.height / scale)))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    image.resize(size, resample).save(dst, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--source', required=True, help='dataset root holding camera/, lidar_png/, annotations')
    parser.add_argument('--splits', required=True, help='directory with the split .txt files')
    parser.add_argument('--output', required=True, help='where the sample dataset is written')
    parser.add_argument('--scale', type=float, default=4.0, help='downscale factor (default: 4)')
    args = parser.parse_args()

    frames = []
    for split, count in FRAMES_PER_SPLIT.items():
        selected = read_split(os.path.join(args.splits, split), count)
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, split), 'w') as f:
            f.write('\n'.join(selected) + '\n')
        frames += selected
    # all.txt lists every frame in the sample, like the full datasets
    frames = list(dict.fromkeys(frames))
    with open(os.path.join(args.output, 'all.txt'), 'w') as f:
        f.write('\n'.join(frames) + '\n')

    for cam_rel in frames:
        name = os.path.basename(cam_rel)
        resize(os.path.join(args.source, cam_rel), os.path.join(args.output, cam_rel),
               args.scale, Image.BILINEAR)
        lidar_rel = cam_rel.replace('camera/', 'lidar_png/')
        resize(os.path.join(args.source, lidar_rel), os.path.join(args.output, lidar_rel),
               args.scale, Image.NEAREST)
        for folder in ANNOTATION_FOLDERS:
            src = os.path.join(args.source, folder, name)
            if os.path.exists(src):
                resize(src, os.path.join(args.output, folder, name), args.scale, Image.NEAREST)

    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == '__main__':
    main()
