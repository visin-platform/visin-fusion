"""A layer used by more than one model is defined once, in visin_fusion/models/layers.py."""

import ast
from pathlib import Path

import torch

from visin_fusion.models import layers

MODELS = Path(layers.__file__).parent
SHARED = {"ResidualConvUnit", "Read_ignore", "Read_add", "Read_projection"}


def test_shared_layers_are_not_redefined_in_model_files():
    redefined = {}
    for path in MODELS.rglob("*.py"):
        if path.name == "layers.py":
            continue
        names = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
        if names & SHARED:
            redefined[path.name] = sorted(names & SHARED)
    assert not redefined


def test_a_residual_conv_unit_keeps_shape_and_skips_the_rectified_input():
    """Its ReLU is in place, so the skip adds relu(x); trained weights depend on that."""
    unit = layers.ResidualConvUnit(4)
    torch.nn.init.zeros_(unit.conv2.weight)
    torch.nn.init.zeros_(unit.conv2.bias)
    x = torch.randn(1, 4, 8, 8)
    assert torch.equal(unit(x.clone()), x.relu())
