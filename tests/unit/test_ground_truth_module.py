"""Ground-truth rendering and its standalone CLI use real image files."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from visin_fusion.config.config import prepare_config
from visin_fusion.engine.stages.visualize import ground_truth as gt

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "zod_sample"


def config_for(tmp_path):
    config = prepare_config({"extends": "clftv2", "Dataset": {"dataset_root": str(SAMPLE)}})
    config["Log"]["logdir"] = str(tmp_path)
    return config


def test_rendered_overlay_preserves_background_and_colors_foreground(tmp_path):
    rgb_path = tmp_path / "camera.png"
    anno_path = tmp_path / "annotation.png"
    image = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
    anno = np.array([[0, 2], [2, 0]], dtype=np.uint8)
    cv2.imwrite(str(rgb_path), image)
    cv2.imwrite(str(anno_path), anno)
    visualizer = gt.GroundTruthVisualizer(config_for(tmp_path), str(tmp_path / "output"))
    visualizer.visualize_ground_truth(str(rgb_path), str(anno_path), 1)
    rendered = cv2.imread(str(tmp_path / "output" / "ground_truth" / "camera.png"))
    assert rendered.shape == image.shape
    assert np.array_equal(rendered[0, 0], image[0, 0])
    assert not np.array_equal(rendered[0, 1], image[0, 1])


def test_missing_image_and_annotation_skip_output(tmp_path, caplog):
    visualizer = gt.GroundTruthVisualizer(config_for(tmp_path), str(tmp_path / "out"))
    visualizer.visualize_ground_truth(str(tmp_path / "missing.png"), str(tmp_path / "none.png"), 1)
    rgb_path = tmp_path / "camera.png"
    cv2.imwrite(str(rgb_path), np.zeros((2, 2, 3), dtype=np.uint8))
    visualizer.visualize_ground_truth(str(rgb_path), str(tmp_path / "none.png"), 2)
    assert "Could not load RGB image" in caplog.text
    assert list((tmp_path / "out" / "ground_truth").iterdir()) == []


def test_fallback_colors_and_overlay_mask(tmp_path):
    visualizer = gt.GroundTruthVisualizer(config_for(tmp_path), str(tmp_path / "out"))
    classes = [{"name": name} for name in ("background", "sign", "vehicle", "human", "unknown")]
    colors = visualizer._draw_segmentation_map_from_labels(
        np.array([[0, 1, 2, 3, 4]], dtype=np.uint8), {"Dataset": {"train_classes": classes}}
    )
    assert colors[0].tolist() == [[0, 0, 0], [255, 0, 0], [128, 0, 128], [0, 255, 255], [255, 255, 255]]
    image = np.full((1, 2, 3), 100, dtype=np.uint8)
    overlay = visualizer._create_overlay(image, np.array([[[0, 0, 0], [200, 0, 0]]], dtype=np.uint8), alpha=0.5)
    assert overlay[0, 0].tolist() == [100, 100, 100]
    assert overlay[0, 1].tolist() == [150, 50, 50]


def test_path_resolution_and_name_validation(tmp_path, monkeypatch):
    paths_file = tmp_path / "frames.txt"
    paths_file.write_text("camera/one.png\ncamera/two.png\n")
    assert gt.load_image_paths(str(paths_file), str(tmp_path), "sample") == ["camera/one.png", "camera/two.png"]
    assert gt.load_image_paths("frame_000004", str(tmp_path), "sample") == ["camera/frame_000004.png"]
    assert gt.load_image_paths("one.png", str(tmp_path), "sample") == ["one.png"]
    seen = []

    class Recorder:
        def visualize_ground_truth(self, rgb, anno, idx):
            seen.append((rgb, anno, idx))

    monkeypatch.setattr(gt, "get_annotation_path", lambda path, config: path.replace("camera", "annotation"))
    gt.process_images(
        None, Recorder(), ["camera/a.png", str(tmp_path / "camera" / "b.png")], str(tmp_path), "sample", {}
    )
    assert [item[2] for item in seen] == [1, 2]
    monkeypatch.setattr(gt, "get_annotation_path", lambda path, config: path.replace("a.png", "other.png"))
    with pytest.raises(AssertionError, match=r"names don.t match"):
        gt.process_images(None, Recorder(), ["camera/a.png"], str(tmp_path), "sample", {})


def test_cli_renders_a_sample_frame(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    monkeypatch.setattr(gt, "load_config", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["ground_truth", "-c", "unused.json", "-p", "frame_000004"])
    gt.main()
    output = tmp_path / "visualizations" / "ground_truth" / "frame_000004.png"
    assert output.is_file() and cv2.imread(str(output)) is not None
