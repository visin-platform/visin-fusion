"""visin-fusion dataset project-lidar and stats on synthetic data."""

import json

import numpy as np
import pytest
from PIL import Image

from visin_fusion.dataset_tools.dataset_stats import class_frequencies, lidar_statistics, suggested_weights
from visin_fusion.dataset_tools.dataset_stats import main as stats_main
from visin_fusion.dataset_tools.project_lidar import encode, project, scaled_intrinsics

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
    _rows, _cols, cam = project(np.array([[0.0, 0.0, 5.0]]), K, T, 100, 80)
    assert cam[0, 2] == 10.0


def test_intrinsics_follow_the_image_size():
    assert scaled_intrinsics(K, (100, 80), (50, 40))[0].tolist() == [50, 0, 25]


def test_lidar_statistics_ignore_pixels_without_a_point(tmp_path):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 51]
    image[1, 1] = [0, 255, 51]
    Image.fromarray(image).save(tmp_path / "a.png")
    mean, std = lidar_statistics([tmp_path / "a.png"])
    assert mean == pytest.approx([0.5, 0.5, 0.2])
    assert std == pytest.approx([0.5, 0.5, 0.0])


def test_class_frequencies_merge_dataset_classes(tmp_path):
    labels = np.array([[0, 0, 1, 2], [3, 3, 3, 9]], dtype=np.uint8)  # 9 is not mapped: ignored
    Image.fromarray(labels).save(tmp_path / "a.png")
    train_classes = [{"index": 0, "dataset_mapping": [0, 1]}, {"index": 1, "dataset_mapping": [2, 3]}]
    assert class_frequencies([tmp_path / "a.png"], train_classes).tolist() == pytest.approx([3 / 7, 4 / 7])


def test_suggested_weights_balance_by_median_frequency():
    assert suggested_weights(np.array([0.9, 0.05, 0.05, 0.0])) == [0.056, 1.0, 1.0, 0.0]


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("with_classes", [False, True])
def test_stats_command_reports_and_optionally_writes_statistics(tmp_path, caplog, write, with_classes):
    for folder in ("camera", "lidar_png", "labels"):
        (tmp_path / folder).mkdir()
    Image.fromarray(np.array([[[255, 0, 51], [0, 255, 51]]], dtype=np.uint8)).save(tmp_path / "lidar_png" / "a.png")
    Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(tmp_path / "labels" / "a.png")
    # Missing images and blank split lines should be ignored.
    (tmp_path / "train.txt").write_text("camera/a.png\n\ncamera/missing.png\n")
    manifest = {
        "format": 1,
        "name": "synthetic",
        "layout": {"camera": "camera", "lidar": "lidar_png"},
        "annotations": ["labels"],
        "splits": {"train": "train.txt"},
    }
    if with_classes:
        manifest["train_classes"] = [
            {"index": 0, "name": "background", "dataset_mapping": [0], "weight": 0.1},
            {"index": 1, "name": "object", "dataset_mapping": [1]},
        ]
    path = tmp_path / "dataset.json"
    original = json.dumps(manifest)
    path.write_text(original)

    stats_main(["--root", str(tmp_path), *(["--write"] if write else [])])

    assert "2 training frames" in caplog.text
    assert "1 LiDAR images" in caplog.text
    if write:
        updated = json.loads(path.read_text())
        assert updated["normalization"] == {"lidar_mean": [0.5, 0.5, 0.2], "lidar_std": [0.5, 0.5, 0.0]}
        if with_classes:
            assert [cls["weight"] for cls in updated["train_classes"]] == [1.0, 1.0]
        assert updated["splits"] == manifest["splits"]
    else:
        assert path.read_text() == original


def test_stats_command_requires_a_training_split(tmp_path):
    with pytest.raises(SystemExit, match=r"has no dataset\.json with a train split"):
        stats_main(["--root", str(tmp_path)])
