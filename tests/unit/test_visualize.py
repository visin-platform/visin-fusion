"""stages/visualize/common.py: render the best checkpoint's predictions, for any model."""

import json
import uuid
from pathlib import Path

import pytest
import torch

from visin_fusion.config.config import load_config
from visin_fusion.engine.stages.visualize import common
from visin_fusion.models.registry import build_model

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "zod_sample"
KINDS = ["segment", "overlay", "compare", "correct_only"]


@pytest.fixture
def trained(tmp_path, monkeypatch):
    """A config and a 'trained' (randomly initialised) DeepLab rgb checkpoint, as training leaves them."""
    monkeypatch.setenv("VISIN_MODE", "disabled")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "extends": "deeplabv3plus",
                "CLI": {"mode": "rgb"},
                "General": {"device": "cpu"},
                "Dataset": {"dataset_root": str(SAMPLE)},
                "Log": {"logdir": str(tmp_path / "logs")},
            }
        )
    )
    config = load_config(config_path)
    epoch_uuid = str(uuid.uuid4())
    (tmp_path / "logs" / "epochs").mkdir(parents=True)
    (tmp_path / "logs" / "checkpoints").mkdir()
    (tmp_path / "logs" / "epochs" / f"epoch_2_{epoch_uuid}.json").write_text(
        json.dumps(
            {"training_uuid": "run-1", "epoch_uuid": epoch_uuid, "epoch": 2, "results": {"val": {"mean_iou": 0.1}}}
        )
    )
    model = build_model(config, pretrained=False)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 2},
        tmp_path / "logs" / "checkpoints" / f"epoch_2_{epoch_uuid}.pth",
    )
    return config_path, tmp_path / "logs"


def test_renders_every_kind_for_every_frame(trained):
    config_path, logdir = trained
    common.main(["-c", str(config_path)])
    frames = (SAMPLE / "visualizations.txt").read_text().split()
    for kind in KINDS:
        assert len(list((logdir / "visualizations" / kind).iterdir())) == len(frames)


def test_one_image_and_another_output_dir(trained, tmp_path):
    config_path, _ = trained
    frame = SAMPLE / (SAMPLE / "visualizations.txt").read_text().split()[0]
    common.main(["-c", str(config_path), "-p", str(frame), "--output-dir", str(tmp_path / "out")])
    assert len(list((tmp_path / "out" / "overlay").iterdir())) == 1


def test_a_frame_that_cannot_be_rendered_fails_the_run(trained, tmp_path):
    config_path, _ = trained
    frames = tmp_path / "frames.txt"
    first = (SAMPLE / "visualizations.txt").read_text().split()[0]
    frames.write_text(f"{first}\ncamera/no_such_frame.png\n")
    with pytest.raises(SystemExit, match="1 of 2 frames failed"):
        common.main(["-c", str(config_path), "-p", str(frames)])


def test_checkpoint_of_another_model_does_not_load(trained, tmp_path):
    config_path, logdir = trained
    other = json.loads(config_path.read_text())
    other["CLI"]["mode"] = "fusion"  # a late-fusion DeepLab has two branches: different weights
    config_path.write_text(json.dumps(other))
    with pytest.raises(RuntimeError, match=r"Missing key|Unexpected key"):
        common.main(["-c", str(config_path)])
