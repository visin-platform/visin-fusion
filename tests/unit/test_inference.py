"""Preprocessor and Predictor: a checkpoint written by the training stage predicts without its config."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from visin_fusion import cli
from visin_fusion.config.config import prepare_config
from visin_fusion.data.dataset_png import DatasetPNG
from visin_fusion.data.preprocessing import Preprocessor
from visin_fusion.inference import Predictor, load_checkpoint, save_prediction
from visin_fusion.models.registry import from_config
from visin_fusion.sample import SAMPLE_DIR
from visin_fusion.utils.helpers import save_model_dict

SAMPLE = SAMPLE_DIR
FRAME = "camera/frame_000004.png"


@pytest.fixture(scope="module")
def config():
    config = prepare_config({"extends": "clftv2", "Dataset": {"dataset_root": str(SAMPLE)}})
    config["CLFTv2"]["pretrained"] = False
    return config


@pytest.fixture(scope="module")
def checkpoint(config, tmp_path_factory):
    config = {**config, "Log": {"logdir": str(tmp_path_factory.mktemp("run"))}}
    model = from_config(config, pretrained=False)
    save_model_dict(config, 0, model, torch.optim.SGD(model.parameters(), lr=0.1), epoch_uuid="abc")
    return next(Path(config["Log"]["logdir"], "checkpoints").glob("*.pth"))


def test_preprocessing_matches_what_the_dataset_gives_the_model(config):
    sample = DatasetPNG(config, "val", str(SAMPLE / "validation.txt"))[0]
    preprocessor = Preprocessor.from_config(config)
    frame = next(f for f in (SAMPLE / "validation.txt").read_text().split() if f)
    camera = SAMPLE / frame
    lidar = SAMPLE / frame.replace("camera", "lidar_png", 1)
    assert torch.allclose(preprocessor.load_rgb(camera), sample["rgb"])
    assert torch.allclose(preprocessor.load_lidar(lidar), sample["lidar"])


def test_a_checkpoint_loads_with_weights_only_and_carries_its_model_info(checkpoint):
    model, info = load_checkpoint(checkpoint)
    assert info["backbone"] == "clftv2"
    assert info["train_classes"][0]["name"] == "background"
    assert not model.training


def test_predict_returns_a_mask_the_size_of_the_input(checkpoint):
    predictor = Predictor.from_checkpoint(checkpoint, device="cpu")
    camera, lidar = SAMPLE / FRAME, SAMPLE / FRAME.replace("camera", "lidar_png", 1)
    mask = predictor.predict(camera, lidar)
    assert mask.shape == Image.open(camera).size[::-1]
    assert mask.dtype == np.uint8
    assert mask.max() < len(predictor.class_names)
    assert predictor.colorize(mask).shape == (*mask.shape, 3)
    assert predictor.overlay(camera, mask).shape == (*mask.shape, 3)


def test_a_missing_input_is_named(checkpoint):
    predictor = Predictor.from_checkpoint(checkpoint, device="cpu")
    with pytest.raises(ValueError, match="needs rgb and lidar; lidar is None"):
        predictor.predict(SAMPLE / FRAME)


def test_an_old_checkpoint_needs_its_config(config, tmp_path):
    model = from_config(config, pretrained=False)
    path = tmp_path / "old.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, path)
    with pytest.raises(ValueError, match="predates model_info"):
        load_checkpoint(path)
    assert load_checkpoint(path, config=config)[1]["backbone"] == "clftv2"


def test_the_predict_command_writes_masks_and_overlays(checkpoint, tmp_path, capsys):
    cli.main(
        ["predict", "--checkpoint", str(checkpoint), "--input", str(SAMPLE / FRAME), "--output", str(tmp_path),
         "--device", "cpu"]
    )  # fmt: skip
    assert (tmp_path / "frame_000004_mask.png").is_file()
    assert (tmp_path / "frame_000004_overlay.png").is_file()
    assert "classes: background" in capsys.readouterr().out


def test_the_saved_mask_keeps_class_indices_and_shows_in_class_colors(checkpoint, tmp_path):
    predictor = Predictor.from_checkpoint(checkpoint, device="cpu")
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    save_prediction(predictor, None, mask, tmp_path, "x")
    saved = Image.open(tmp_path / "x_mask.png")
    assert saved.mode == "P"
    assert np.array_equal(np.array(saved), mask)
    assert np.array_equal(np.array(saved.convert("RGB"))[1, 1], predictor.palette[3])


def nested_frames(root):
    """Two frames with the same file name in different folders, in the camera/lidar_png layout."""
    for folder in ("day", "night"):
        for kind in ("camera", "lidar_png"):
            (root / kind / folder).mkdir(parents=True)
            source = SAMPLE / FRAME.replace("camera", kind, 1)
            (root / kind / folder / "x.png").write_bytes(source.read_bytes())
    return root / "camera"


def test_predict_keeps_same_named_images_apart_and_skips_its_own_output(checkpoint, tmp_path):
    camera = nested_frames(tmp_path)
    output = camera / "predictions"
    args = ["predict", "--checkpoint", str(checkpoint), "--input", str(camera), "--output", str(output)]
    cli.main([*args, "--device", "cpu"])
    cli.main([*args, "--device", "cpu"])
    written = sorted(p.relative_to(output).as_posix() for p in output.rglob("*.png"))
    assert written == ["day/x_mask.png", "day/x_overlay.png", "night/x_mask.png", "night/x_overlay.png"]


def test_predict_names_a_missing_lidar_projection(checkpoint, tmp_path):
    camera = nested_frames(tmp_path)
    (tmp_path / "lidar_png" / "night" / "x.png").unlink()
    with pytest.raises(SystemExit, match=r"No LiDAR projection for .*night.*--lidar-dir"):
        cli.main(["predict", "--checkpoint", str(checkpoint), "--input", str(camera), "--output", str(tmp_path / "o")])
