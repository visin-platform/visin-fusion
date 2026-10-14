"""TrainingEngine: accumulation steps once per group of batches; non-finite values are skipped, then stop the run."""

import pytest
import torch
from torch import nn

from tests.unit.custom_model_plugin import TinyNet
from visin_fusion.config.config import prepare_config
from visin_fusion.engine.training_engine import TrainingDiverged, TrainingEngine
from visin_fusion.models.training import TrainingSetup
from visin_fusion.sample import SAMPLE_DIR


class Metrics:
    """Just enough of MetricsCalculator for one epoch."""

    def create_accumulators(self, device):
        return {}

    def update_accumulators(self, accumulators, segmap, labels, num_classes):
        pass

    def compute_epoch_metrics(self, accumulators, total_loss, batches):
        return {"epoch_loss": total_loss / batches}


def engine_for(tmp_path, accumulate):
    config = prepare_config({"extends": "clftv2", "Dataset": {"dataset_root": str(SAMPLE_DIR)}})
    config["General"]["accumulate_batches"] = accumulate
    config["Log"]["logdir"] = str(tmp_path)
    model = TinyNet(num_classes=4, mode="rgb")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = []
    real_step = optimizer.step
    optimizer.step = lambda *a, **k: (steps.append(1), real_step(*a, **k))[1]
    cross_entropy = nn.CrossEntropyLoss()
    setup = TrainingSetup(optimizer, None, lambda outputs, segmap, labels: cross_entropy(segmap, labels), None, False)
    engine = TrainingEngine(model, setup, Metrics(), config, None, str(tmp_path), torch.device("cpu"))
    return engine, steps


def batches(count):
    return [
        {
            "rgb": torch.randn(2, 3, 8, 8),
            "lidar": torch.randn(2, 3, 8, 8),
            "anno": torch.zeros(2, 8, 8, dtype=torch.long),
        }
        for _ in range(count)
    ]


@pytest.mark.parametrize(("accumulate", "steps"), [(1, 7), (3, 3), (7, 1), (10, 1)])
def test_the_optimizer_steps_once_per_group_of_batches(tmp_path, accumulate, steps):
    engine, counted = engine_for(tmp_path, accumulate)
    engine.train_epoch(batches(7), num_classes=4)
    assert len(counted) == steps


def test_accumulated_gradients_are_the_average_of_the_batches(tmp_path):
    data = batches(4)
    torch.manual_seed(0)
    engine, _ = engine_for(tmp_path / "a", 4)
    reference, _ = engine_for(tmp_path / "b", 1)
    reference.model.load_state_dict(engine.model.state_dict())
    engine.train_epoch(data, num_classes=4)
    reference.optimizer.zero_grad()
    for batch in data:
        rgb, lidar, labels = reference._batch(batch)
        (reference._loss(rgb, lidar, labels)[1] / 4).backward()
    reference.optimizer.step()
    for ours, expected in zip(engine.model.parameters(), reference.model.parameters(), strict=True):
        assert torch.allclose(ours, expected, atol=1e-6)


def poisoned(engine, bad):
    """Make the loss NaN on the batches whose index is in ``bad``."""
    real = engine._loss
    seen = iter(range(1000))

    def loss(rgb, lidar, labels):
        segmap, value = real(rgb, lidar, labels)
        return segmap, value * float("nan") if next(seen) in bad else value

    engine._loss = loss


def test_a_batch_with_a_non_finite_loss_is_skipped(tmp_path):
    engine, counted = engine_for(tmp_path, 1)
    poisoned(engine, {2})
    before = [p.clone() for p in engine.model.parameters()]
    metrics = engine.train_epoch(batches(7), num_classes=4)
    assert len(counted) == 6
    assert metrics["epoch_loss"] == metrics["epoch_loss"]  # not NaN
    assert any(not torch.equal(a, b) for a, b in zip(before, engine.model.parameters(), strict=True))
    assert all(torch.isfinite(p).all() for p in engine.model.parameters())


def test_a_skipped_last_batch_of_a_group_still_steps_the_group(tmp_path):
    engine, counted = engine_for(tmp_path, 3)
    poisoned(engine, {2})  # the last of the first group of three
    engine.train_epoch(batches(6), num_classes=4)
    assert len(counted) == 2


def test_too_many_bad_batches_stop_the_epoch(tmp_path):
    engine, _ = engine_for(tmp_path, 1)
    engine.max_nonfinite_batches = 2
    poisoned(engine, {0, 1, 2})
    with pytest.raises(TrainingDiverged, match="3 batches"):
        engine.train_epoch(batches(7), num_classes=4)


def test_non_finite_gradients_do_not_reach_the_weights_without_a_scaler(tmp_path):
    engine, counted = engine_for(tmp_path, 1)
    engine.model.zero_grad()
    for parameter in engine.model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    before = [p.clone() for p in engine.model.parameters()]
    assert engine._optimizer_step() is False
    assert not counted
    assert all(torch.equal(a, b) for a, b in zip(before, engine.model.parameters(), strict=True))


@pytest.mark.parametrize(
    ("precision", "mixed", "scaled"), [(None, False, False), ("fp32", False, False), ("bf16", False, False)]
)
def test_precision_is_off_on_the_cpu(tmp_path, precision, mixed, scaled):
    engine, _ = engine_for(tmp_path, 1)
    engine.config["General"]["precision"] = precision
    rebuilt = TrainingEngine(
        engine.model, engine.setup, Metrics(), engine.config, None, str(tmp_path), torch.device("cpu")
    )
    assert rebuilt.mixed_precision is mixed
    assert rebuilt.scaler.is_enabled() is scaled


def test_a_non_finite_validation_loss_ends_training_without_saving(tmp_path):
    engine, _ = engine_for(tmp_path, 1)
    engine.config["General"]["epochs"] = 5
    engine.config["Log"]["logdir"] = str(tmp_path)
    losses = iter([0.5, float("nan"), 0.4])
    engine.validate_epoch = lambda loader, classes: {"epoch_loss": next(losses), "mean_iou": 0.5}
    engine.metrics_calc.print_metrics = lambda *args, **kwargs: None
    engine.metrics_calc.prepare_results_dict = lambda *args: {}
    engine._log_tensorboard = lambda *args: None
    saved = []
    engine._save = lambda epoch, uuid: saved.append(epoch)
    engine.train_full(batches(2), batches(2), num_classes=4)
    assert engine.diverged is not None
    assert saved == [0]  # epoch 0 only: neither the NaN epoch nor a final save
