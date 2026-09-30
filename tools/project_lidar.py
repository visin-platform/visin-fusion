#!/usr/bin/env python3
"""Project LiDAR point clouds onto camera images as lidar_png, for a dataset of your own.

For every frame in the dataset's split files, the point cloud next to the camera image (its camera
folder replaced by --points, e.g. camera/000001.png -> lidar_points/000001.npy or .bin) is moved into
the camera frame, projected with the camera intrinsics, and written as lidar_png/000001.png.

    python tools/project_lidar.py --root /data/my_dataset --calibration calibration.json

calibration.json (one camera for the whole dataset):

    {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],   intrinsics, for an image of width x height
     "T_cam_lidar": [[...4x4...]],                  LiDAR -> camera transform (metres)
     "width": 1920, "height": 1280}                 image size K is for; rescaled to the camera images

Encoding (fixed for the whole dataset, so a value means the same distance in every frame): each
pixel's R, G, B are the X (right), Y (down), Z (forward) camera coordinates of the nearest point that
lands on it, mapped linearly from --x-range, --y-range, --z-range (metres) onto 1..255 and clipped.
0 means no point. Compute the model's normalization for it with tools/dataset_stats.py.

The existing datasets were encoded differently (per-image scaling), see docs/datasets.md.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from visin_fusion.utils.helpers import replace_camera_folder


def load_points(path):
    """Points as an (N, 3) float array: .npy with x, y, z first, or KITTI-style float32 .bin (x, y, z, i)."""
    if path.endswith(".npy"):
        points = np.load(path)
    elif path.endswith(".bin"):
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    else:
        raise ValueError(f"unsupported point cloud file {path} (use .npy or .bin)")
    return np.asarray(points[:, :3], dtype=np.float64)


def project(points, K, T_cam_lidar, width, height):
    """Pixel rows, columns and camera-frame XYZ of the points in front of the camera and inside the image."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    cam = (np.asarray(T_cam_lidar) @ homogeneous.T).T[:, :3]
    cam = cam[cam[:, 2] > 0.1]  # in front of the camera
    uvw = (np.asarray(K) @ cam.T).T
    cols = np.floor(uvw[:, 0] / uvw[:, 2]).astype(int)
    rows = np.floor(uvw[:, 1] / uvw[:, 2]).astype(int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows[inside], cols[inside], cam[inside]


def encode(rows, cols, cam, width, height, ranges):
    """The lidar_png image: nearest point per pixel, each axis mapped from its range onto 1..255."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    order = np.argsort(-cam[:, 2])  # far first, so nearer points overwrite them
    rows, cols, cam = rows[order], cols[order], cam[order]
    for axis, (low, high) in enumerate(ranges):
        scaled = 1 + np.clip((cam[:, axis] - low) / (high - low), 0, 1) * 254
        image[rows, cols, axis] = np.round(scaled).astype(np.uint8)
    return image


def scaled_intrinsics(K, from_size, to_size):
    K = np.array(K, dtype=np.float64)
    K[0] *= to_size[0] / from_size[0]
    K[1] *= to_size[1] / from_size[1]
    return K


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--calibration", required=True, help="calibration.json (see above)")
    parser.add_argument(
        "--points", default="lidar_points", help="point cloud folder beside camera/ (default: lidar_points)"
    )
    parser.add_argument("--output", default="lidar_png", help="output folder beside camera/ (default: lidar_png)")
    parser.add_argument("--frames", default="all.txt", help="split file listing the frames (default: all.txt)")
    parser.add_argument("--x-range", type=float, nargs=2, default=[-40, 40], metavar=("LOW", "HIGH"))
    parser.add_argument("--y-range", type=float, nargs=2, default=[-5, 5], metavar=("LOW", "HIGH"))
    parser.add_argument("--z-range", type=float, nargs=2, default=[0, 80], metavar=("LOW", "HIGH"))
    args = parser.parse_args()

    with open(args.calibration) as f:
        calibration = json.load(f)
    with open(os.path.join(args.root, args.frames)) as f:
        frames = [line.strip() for line in f if line.strip()]
    ranges = [args.x_range, args.y_range, args.z_range]

    written = missing = 0
    for frame in frames:
        cam_path = os.path.join(args.root, frame)
        stem = os.path.splitext(replace_camera_folder(cam_path, args.points))[0]
        points_path = next((stem + ext for ext in (".npy", ".bin") if os.path.exists(stem + ext)), None)
        if points_path is None:
            missing += 1
            continue
        width, height = Image.open(cam_path).size
        K = scaled_intrinsics(calibration["K"], (calibration["width"], calibration["height"]), (width, height))
        rows, cols, cam = project(load_points(points_path), K, calibration["T_cam_lidar"], width, height)
        out = replace_camera_folder(cam_path, args.output)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(encode(rows, cols, cam, width, height, ranges)).save(out)
        written += 1
    print(f"Wrote {written} LiDAR projections; {missing} frames had no point cloud")


if __name__ == "__main__":
    main()
