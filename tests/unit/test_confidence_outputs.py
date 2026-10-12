"""Confidence PNGs and visualized predictions share Predictor's decoder at original image dimensions."""

import numpy as np
import torch
from PIL import Image

from visin_fusion.engine.visualizer import Visualizer
from visin_fusion.inference import Predictor, save_prediction


class FixedScores(torch.nn.Module):
    def forward(self, rgb, lidar):
        return torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]], [[-2.0, 2.0], [2.0, -2.0]]]])


def test_predictor_and_visualizer_write_matching_input_size_masks_and_confidence(tmp_path):
    classes = [
        {"name": "background", "index": 0, "color": [0, 0, 0]},
        {"name": "vehicle", "index": 1, "color": [255, 0, 0]},
    ]
    transforms = {"resize": 2, "image_mean": [0.0] * 3, "image_std": [1.0] * 3}
    info = {"mode": "rgb", "train_classes": classes, "transforms": transforms}
    camera = tmp_path / "camera.png"
    Image.new("RGB", (13, 11)).save(camera)
    predictor = Predictor(FixedScores(), info)
    probabilities = predictor.probabilities(camera)
    mask, confidence = predictor.predict(camera), predictor.confidence(camera)
    assert probabilities.shape == (2, 11, 13)
    assert torch.allclose(probabilities.sum(dim=0), torch.ones(11, 13))
    distinct = (probabilities[0] - probabilities[1]).abs().numpy() > np.finfo(np.float32).eps
    assert np.array_equal(probabilities.argmax(dim=0).numpy()[distinct], mask[distinct])
    assert confidence.dtype == np.uint8 and confidence.shape == (11, 13)
    assert confidence[5, 6] == 128 and confidence[0, 0] == 250
    save_prediction(predictor, camera, mask, tmp_path / "predict", "frame", confidence)
    saved = Image.open(tmp_path / "predict" / "frame_confidence.png")
    assert saved.mode == "L" and np.array_equal(np.array(saved), confidence)
    visualizer = Visualizer({"Dataset": {"train_classes": classes}}, str(tmp_path / "visualize"))
    visualizer.visualize_prediction(FixedScores()(None, None), str(camera), str(tmp_path / "absent.png"), 1)
    assert np.array_equal(np.array(Image.open(tmp_path / "visualize" / "mask" / "camera.png")), mask)
    assert np.array_equal(np.array(Image.open(tmp_path / "visualize" / "confidence" / "camera.png")), confidence)
