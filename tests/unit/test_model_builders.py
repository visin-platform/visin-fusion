"""Legacy builder adapters preserve config arguments and load saved weights."""

import pytest
import torch

from visin_fusion.engine import advanced_model_builder, model_builder


@pytest.mark.parametrize(
    ("module", "section", "constructor"),
    [
        (model_builder, "CLFT", "CLFT"),
        (advanced_model_builder, "CLFTv2", "CLFTv2Network"),
    ],
)
def test_builder_constructs_model_and_restores_checkpoint(monkeypatch, tmp_path, module, section, constructor):
    seen = {}

    def make_model(**kwargs):
        seen.update(kwargs)
        return torch.nn.Linear(2, 3)

    monkeypatch.setattr(module, constructor, make_model)
    config = {
        "Dataset": {"train_classes": [{}, {}, {}], "transforms": {"resize": 16}},
        "CLI": {"backbone": section},
        section: {
            "pretrained": False,
            "resample_dim": 4,
            "read": "ignore",
            "reassembles": [4],
            "type": "segmentation",
            "model_timm": "unused",
            "patch_size": 4,
            "emb_dim": 4,
            "hooks": [0],
            "emb_dims": [4],
            "fusion_strategy": "simple_average",
        },
    }
    builder = module.ModelBuilder(config, "cpu") if section == "CLFT" else module.AdvancedModelBuilder(config, "cpu")
    model = builder.build_model()
    assert model.out_features == 3
    assert seen["nclasses"] == 3
    assert seen["pretrained"] is False
    assert seen["model_timm"] == "unused"
    if section == "CLFT":
        assert seen["RGB_tensor_size"] == (3, 16, 16)
        assert seen["XYZ_tensor_size"] == (3, 16, 16)
    else:
        assert seen["fusion_strategy"] == "simple_average"

    with torch.no_grad():
        model.weight.fill_(2)
    path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, path)
    target = torch.nn.Linear(2, 3)
    loaded, epoch = builder.load_checkpoint(target, path)
    assert loaded is target
    assert epoch == 3
    assert torch.equal(target.weight, model.weight)
