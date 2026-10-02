"""TrainingEngine.train_epoch: General.accumulate_batches steps the optimizer once per group of batches."""

import pytest
import torch
from torch import nn

from tests.unit.custom_model_plugin import TinyNet
from visin_fusion.config.config import prepare_config
from visin_fusion.engine.training_engine import TrainingEngine
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
