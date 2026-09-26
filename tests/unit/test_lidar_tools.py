"""tools/project_lidar.py and tools/dataset_stats.py on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from dataset_stats import class_frequencies, lidar_statistics, suggested_weights  # noqa: E402
from project_lidar import encode, project, scaled_intrinsics  # noqa: E402

K = [[100, 0, 50], [0, 100, 40], [0, 0, 1]]
IDENTITY = np.eye(4)
RANGES = [(-40, 40), (-5, 5), (0, 80)]


def test_point_straight_ahead_lands_on_the_principal_point():
    rows, cols, cam = project(np.array([[0.0, 0.0, 10.0]]), K, IDENTITY, 100, 80)
    assert (rows[0], cols[0]) == (40, 50)
    image = encode(rows, cols, cam, 100, 80, RANGES)
    assert image[40, 50].tolist() == [128, 128, 33]  # x and y at mid-range, z = 1 + 10/80 * 254
    assert image.sum() == 128 + 128 + 33  # every other pixel is 0: no point


def test_points_behind_the_camera_or_outside_the_image_are_dropped():
    points = np.array([[0.0, 0.0, -5.0], [100.0, 0.0, 10.0]])
    rows, _, _ = project(points, K, IDENTITY, 100, 80)
    assert len(rows) == 0


def test_nearest_point_wins_a_pixel():
    points = np.array([[0.0, 0.0, 60.0], [0.0, 0.0, 10.0], [0.0, 0.0, 30.0]])
    rows, cols, cam = project(points, K, IDENTITY, 100, 80)
    assert encode(rows, cols, cam, 100, 80, RANGES)[40, 50, 2] == 33


def test_extrinsics_move_points_into_the_camera_frame():
    T = np.eye(4)
    T[:3, 3] = [0, 0, 5]  # the LiDAR sits 5 m behind the camera
    rows, cols, cam = project(np.array([[0.0, 0.0, 5.0]]), K, T, 100, 80)
    assert cam[0, 2] == 10.0


def test_intrinsics_follow_the_image_size():
    assert scaled_intrinsics(K, (100, 80), (50, 40))[0].tolist() == [50, 0, 25]


def test_lidar_statistics_ignore_pixels_without_a_point(tmp_path):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 51]
    image[1, 1] = [0, 255, 51]
    Image.fromarray(image).save(tmp_path / 'a.png')
    mean, std = lidar_statistics([tmp_path / 'a.png'])
    assert mean == pytest.approx([0.5, 0.5, 0.2])
    assert std == pytest.approx([0.5, 0.5, 0.0])


def test_class_frequencies_merge_dataset_classes(tmp_path):
    labels = np.array([[0, 0, 1, 2], [3, 3, 3, 9]], dtype=np.uint8)  # 9 is not mapped: ignored
    Image.fromarray(labels).save(tmp_path / 'a.png')
    train_classes = [{'index': 0, 'dataset_mapping': [0, 1]}, {'index': 1, 'dataset_mapping': [2, 3]}]
    assert class_frequencies([tmp_path / 'a.png'], train_classes).tolist() == pytest.approx([3 / 7, 4 / 7])


def test_suggested_weights_balance_by_median_frequency():
    assert suggested_weights(np.array([0.9, 0.05, 0.05, 0.0])) == [0.056, 1.0, 1.0, 0.0]
