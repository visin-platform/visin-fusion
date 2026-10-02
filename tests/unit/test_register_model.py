"""register_model: a model added by a plugin module is accepted by the config, built by the registry and
trained by a stage process, with no change to any stage."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from visin_fusion.config.config import prepare_config
from visin_fusion.config.config_schema import BACKBONES, ConfigError, export_schema, validate_config
from visin_fusion.models import FusionModel, register_model
from visin_fusion.models.registry import from_config, training_setup
from visin_fusion.sample import SAMPLE_DIR

PLUGIN = "tests.unit.custom_model_plugin"
REPO = Path(__file__).resolve().parents[2]


def tiny_config(**overrides):
    config = {
        "extends": "clftv2",
        "plugins": [PLUGIN],
        "CLI": {"backbone": "tinynet", "mode": "fusion"},
        "TinyNet": {"width": 6},
        "Dataset": {"dataset_root": str(SAMPLE_DIR)},
        "Log": {"logdir": "tiny-logs"},
        "General": {"epochs": 1, "batch_size": 2, "early_stop_patience": 1, "max_checkpoints": 1, "device": "cpu"},
    }
    config.update(overrides)
    return config


def test_a_registered_model_is_accepted_built_and_trainable():
    config = prepare_config(tiny_config())
    assert "tinynet" in BACKBONES
    assert export_schema()["$defs"]["CLI"]["properties"]["backbone"]["enum"][-1] == "tinynet"
    model = from_config(config)
    assert type(model).__name__ == "TinyNet"
    assert model.implementation.head.in_channels == 6
    assert model.mode == "cross_fusion"
    rgb = torch.randn(2, 3, 16, 16)
    setup = training_setup(config, model, torch.device("cpu"))
    loss = setup.loss(model.raw_forward(rgb, rgb), model(rgb, rgb), torch.zeros(2, 16, 16, dtype=torch.long))
    loss.backward()
    setup.optimizer.step()


def test_an_unregistered_model_names_the_way_to_register_it():
    config = prepare_config(tiny_config())
    config["CLI"]["backbone"] = "ghostnet"
    with pytest.raises(ConfigError, match=r"register_model.*plugins"):
        validate_config(config)


def test_other_unknown_top_level_keys_are_still_errors():
    config = prepare_config(tiny_config())
    config["TinyNett"] = {}
    with pytest.raises(ConfigError, match="TinyNett"):
        validate_config(config)


def test_a_plugin_that_cannot_be_imported_is_named():
    with pytest.raises(ValueError, match="plugins: cannot import 'no_such_plugin_module'"):
        prepare_config(tiny_config(plugins=["no_such_plugin_module"]))


def test_registration_is_checked():
    class Plain:
        pass

    with pytest.raises(TypeError, match="subclass"):
        register_model("plain", Plain, section="Plain")
    prepare_config(tiny_config())
    cls = from_config(prepare_config(tiny_config())).__class__
    register_model("tinynet", cls, section="TinyNet")  # the same registration again is fine
    with pytest.raises(ValueError, match="already"):
        register_model("tinynet", cls, section="Other")
    with pytest.raises(ValueError, match="already used"):
        register_model("tinynet2", FusionModel, section="CLFTv2")


def test_a_stage_process_trains_a_registered_model(tmp_path):
    config = tiny_config(Log={"logdir": str(tmp_path / "logs")})
    config["General"]["num_workers"] = 0
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(prepare_config(config)))
    result = subprocess.run(
        [sys.executable, "-m", "visin_fusion.engine.stages.train.common", "-c", str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "VISIN_MODE": "disabled"},
        check=False,
    )
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    checkpoints = list((tmp_path / "logs" / "checkpoints").glob("*.pth"))
    assert checkpoints
    assert torch.load(checkpoints[0], weights_only=True)["model_info"]["backbone"] == "tinynet"
    from visin_fusion.inference import Predictor

    frame = SAMPLE_DIR / "camera" / "frame_000004.png"
    predictor = Predictor.from_checkpoint(checkpoints[0], device="cpu")
    mask = predictor.predict(frame, SAMPLE_DIR / "lidar_png" / "frame_000004.png")
    assert mask.max() < len(predictor.class_names)
