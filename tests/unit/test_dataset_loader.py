"""visin_fusion/data/dataset_png.py on the sample dataset: shapes, inputs per mode, augmentation."""

import copy
from pathlib import Path

import pytest
import torch

from visin_fusion.config.config import prepare_config
from visin_fusion.data.dataset_png import DatasetPNG

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "zod_sample"


def config_for(mode="cross_fusion", **transforms):
    config = prepare_config({"extends": "clftv2", "CLI": {"mode": mode}, "Dataset": {"dataset_root": str(SAMPLE)}})
    config["Dataset"]["transforms"].update(transforms)
    return config


def item(config, split="val", index=0):
    return DatasetPNG(config, split, config["Dataset"]["val_split"])[index]


def test_shapes_follow_the_configured_size():
    sample = item(config_for())
    size = 256  # the clftv2 preset's resize
    assert sample["rgb"].shape == (3, size, size)
    assert sample["lidar"].shape == (3, size, size)
    assert sample["anno"].shape == (size, size)
    assert sample["anno"].dtype == torch.long


def test_annotations_keep_dataset_class_indices():
    # Relabelling to training classes happens later (utils.helpers.relabel_annotation)
    values = set(item(config_for())["anno"].unique().tolist())
    known = {c["index"] for c in config_for()["Dataset"]["dataset_classes"]}
    assert values <= known


def test_rgb_mode_gets_an_empty_lidar_input():
    lidar = item(config_for("rgb"))["lidar"]
    assert torch.allclose(lidar, lidar.flatten(1)[:, :1].unsqueeze(-1))  # constant per channel
    assert not torch.allclose(item(config_for("cross_fusion"))["lidar"], lidar)


def test_validation_is_not_augmented():
    config = config_for(p_flip=1.0, p_crop=1.0, p_rot=1.0)
    first, second = item(config, "val"), item(config, "val")
    assert torch.equal(first["rgb"], second["rgb"]) and torch.equal(first["anno"], second["anno"])


@pytest.mark.parametrize("key", ["rgb", "lidar", "anno"])
def test_flip_moves_image_labels_and_lidar_together(key):
    plain = item(config_for(p_flip=0.0, p_crop=0.0, p_rot=0.0), "train")
    flipped = item(config_for(p_flip=1.0, p_crop=0.0, p_rot=0.0), "train")
    assert torch.equal(flipped[key], torch.flip(plain[key], dims=[-1]))


def test_missing_annotation_warns_and_gives_an_empty_mask(tmp_path, caplog):
    import shutil

    dataset = tmp_path / "sample"
    shutil.copytree(SAMPLE, dataset)
    config = copy.deepcopy(config_for())
    config["Dataset"]["dataset_root"] = str(dataset)
    frame = (dataset / "validation.txt").read_text().split()[0]
    (dataset / "annotation_camera_only" / Path(frame).name).unlink()
    sample = DatasetPNG(config, "val", str(dataset / "validation.txt"))[0]
    assert sample["anno"].sum() == 0
    assert "No annotation" in caplog.text


def test_lidar_normalization_is_optional():
    config = config_for()
    config["Dataset"]["transforms"].pop("lidar_mean")
    config["Dataset"]["transforms"].pop("lidar_std")
    sample = item(config)
    assert sample["lidar"].shape == sample["rgb"].shape
    assert torch.isfinite(sample["lidar"]).all()


def test_random_crop_supports_128_pixel_input():
    config = config_for(p_crop=1.0)
    config["Dataset"]["transforms"]["resize"] = 128
    sample = item(config, "train")
    assert sample["rgb"].shape == (3, 128, 128)
    assert sample["anno"].shape == (128, 128)
