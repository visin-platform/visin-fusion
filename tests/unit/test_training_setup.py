"""visin_fusion/models/registry.py:training_setup: each model's optimizer, schedule and loss."""

import pytest
import torch

from visin_fusion.config.config import prepare_config
from visin_fusion.models.registry import from_config, training_setup
from visin_fusion.sample import SAMPLE_DIR

SAMPLE = SAMPLE_DIR


def setup_for(preset, **overrides):
    config = prepare_config({"extends": preset, "Dataset": {"dataset_root": str(SAMPLE)}, **overrides})
    model = from_config(config, pretrained=False)
    return config, model, training_setup(config, model, torch.device("cpu"))


def learning_rates(setup, epochs):
    rates = []
    for _ in range(epochs):
        rates.append(setup.optimizer.param_groups[-1]["lr"])
        setup.optimizer.step()
        setup.scheduler.step()
    return rates


def test_clft_multiplies_the_rate_by_lr_momentum_each_epoch():
    _, _, setup = setup_for("clft")
    assert learning_rates(setup, 3) == pytest.approx([8e-5, 8e-5 * 0.99, 8e-5 * 0.99**2])


def test_clftv2_warms_up_then_decays():
    _, _, setup = setup_for("clftv2", General={"epochs": 20}, CLFTv2={"warmup_epochs": 5})
    rates = learning_rates(setup, 20)
    assert rates[0] == pytest.approx(8e-5 * 0.01)
    assert max(rates) == pytest.approx(8e-5) and rates[-1] < rates[5]
    assert setup.clip_grad_norm == 1.0


def test_query_models_train_the_backbone_ten_times_slower():
    for preset, clip in [("maskformer", 1.0), ("mask2former", 0.01)]:
        _, model, setup = setup_for(preset)
        backbone, rest = setup.optimizer.param_groups
        assert backbone["lr"] == pytest.approx(rest["lr"] * 0.1)
        assert sum(p.numel() for p in backbone["params"]) == sum(p.numel() for p in model.backbone.parameters())
        assert setup.clip_grad_norm == clip


def test_deeplab_cosine_spans_the_epochs():
    _, _, setup = setup_for("deeplabv3plus", General={"epochs": 4})
    assert setup.scheduler.T_max == 4
    assert setup.mixed_precision is False


def test_deeplab_plateau_is_a_choice():
    _, _, setup = setup_for("deeplabv3plus", DeepLabV3Plus={"lr_scheduler": {"type": "plateau"}})
    assert isinstance(setup.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_unknown_deeplab_schedule_is_named():
    with pytest.raises(ValueError, match="sgdr"):
        setup_for("deeplabv3plus", DeepLabV3Plus={"lr_scheduler": {"type": "sgdr"}})


def test_cross_entropy_uses_the_class_weights():
    _config, _, setup = setup_for("deeplabv3plus", CLI={"mode": "rgb"})
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    confident_background = torch.zeros(1, 4, 4, 4)
    confident_background[:, 0] = 10
    assert setup.loss(None, confident_background, labels).item() < 1e-3


def test_schedule_survives_a_checkpoint(tmp_path):
    # A resumed run continues its schedule instead of starting the warmup again
    _, _, setup = setup_for("clftv2", General={"epochs": 20}, CLFTv2={"warmup_epochs": 5})
    learning_rates(setup, 7)
    torch.save(
        {"scheduler_state_dict": setup.scheduler.state_dict(), "optimizer_state_dict": setup.optimizer.state_dict()},
        tmp_path / "c.pth",
    )
    _, _, resumed = setup_for("clftv2", General={"epochs": 20}, CLFTv2={"warmup_epochs": 5})
    state = torch.load(tmp_path / "c.pth", weights_only=False)
    resumed.optimizer.load_state_dict(state["optimizer_state_dict"])
    resumed.scheduler.load_state_dict(state["scheduler_state_dict"])
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(setup.optimizer.param_groups[0]["lr"])
