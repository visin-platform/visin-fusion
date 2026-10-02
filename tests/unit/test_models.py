"""visin_fusion/models/registry.py: every model builds from its preset and gives a segmentation map of the right
shape, in every mode.

Random weights (``pretrained: false``), so nothing is downloaded; one CPU forward pass each.
"""

import pytest
import torch

from visin_fusion.config.config import prepare_config
from visin_fusion.config.config_schema import BACKBONES
from visin_fusion.models.registry import from_config
from visin_fusion.sample import SAMPLE_DIR
from visin_fusion.utils.helpers import calculate_num_classes

SAMPLE = SAMPLE_DIR
CASES = [
    (preset, mode)
    for preset in ("clft", "clftv2", "maskformer", "mask2former", "deeplabv3plus")
    for mode in ("rgb", "lidar", "fusion")
]


def config_for(preset, mode):
    overrides = {"extends": preset, "Dataset": {"dataset_root": str(SAMPLE)}}
    config = prepare_config(overrides)
    if mode != "fusion":
        config["CLI"]["mode"] = mode
    config[BACKBONES[config["CLI"]["backbone"]][0]]["pretrained"] = False
    return config


def build(config):
    return from_config(config)


@pytest.mark.parametrize(("preset", "mode"), CASES, ids=[f"{p}-{m}" for p, m in CASES])
def test_forward_pass(preset, mode):
    torch.manual_seed(0)
    config = config_for(preset, mode)
    size = config["Dataset"]["transforms"]["resize"]
    model = build(config).eval()
    rgb, lidar = torch.randn(1, 3, size, size), torch.randn(1, 3, size, size)
    with torch.no_grad():
        output = model(rgb, lidar)
    assert output.shape == (1, calculate_num_classes(config), size, size)
    assert torch.isfinite(output).all()


def test_checkpoint_missing_layers_does_not_load(tmp_path):
    # With strict=False (as before) the missing layer would stay randomly initialised, unnoticed
    config = config_for("clftv2", "fusion")
    state = build(config).state_dict()
    del state[next(iter(state))]
    torch.save({"model_state_dict": state, "epoch": 0}, tmp_path / "clftv2.pth")
    with pytest.raises(RuntimeError, match="Missing key"):
        build(config).load_state_dict(torch.load(tmp_path / "clftv2.pth", weights_only=True)["model_state_dict"])


def test_pretrained_override_leaves_the_config_alone():
    config = config_for("deeplabv3plus", "rgb")
    config["DeepLabV3Plus"]["pretrained"] = True
    from_config(config, pretrained=False)
    assert config["DeepLabV3Plus"]["pretrained"] is True
