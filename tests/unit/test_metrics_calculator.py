"""core/metrics_calculator.py against a small hand-computed example.

Classes: 0 background, 1, 2.   labels [0, 1, 1, 2]   predicted [1, 1, 2, 2]
  class 1: overlap 1, predicted 2, labelled 2, union 3 -> IoU 1/3, precision 1/2, recall 1/2
  class 2: overlap 1, predicted 2, labelled 1, union 2 -> IoU 1/2, precision 1/2, recall 1
  pixel accuracy 2/4 (pixels 1 and 3 are right)
"""
import pytest
import torch

from core.metrics_calculator import MetricsCalculator
from utils.metrics import find_overlap_exclude_bg_ignore

CONFIG = {'Dataset': {'train_classes': [
    {'name': 'background', 'index': 0}, {'name': 'car', 'index': 1}, {'name': 'person', 'index': 2}]}}
LABELS = torch.tensor([[[0, 1, 1, 2]]])
PREDICTED = torch.tensor([[[1, 1, 2, 2]]])


def logits(prediction):
    return torch.nn.functional.one_hot(prediction, 3).permute(0, 3, 1, 2).float()


@pytest.fixture
def calculator():
    return MetricsCalculator(CONFIG, 2, find_overlap_exclude_bg_ignore)


def test_evaluated_classes_exclude_background(calculator):
    assert calculator.eval_classes == ['car', 'person']
    assert calculator.eval_indices == [1, 2]


def test_update_and_compute(calculator):
    calculator.update(logits(PREDICTED), LABELS)
    metrics = calculator.compute()
    assert metrics['iou'].tolist() == pytest.approx([1 / 3, 1 / 2], abs=1e-5)
    assert metrics['precision'].tolist() == pytest.approx([1 / 2, 1 / 2], abs=1e-5)
    assert metrics['recall'].tolist() == pytest.approx([1 / 2, 1], abs=1e-5)
    assert metrics['f1'].tolist() == pytest.approx([1 / 2, 2 / 3], abs=1e-5)
    assert metrics['mean_iou'] == pytest.approx(5 / 12, abs=1e-5)


def test_batches_accumulate(calculator):
    for _ in range(3):
        calculator.update(logits(PREDICTED), LABELS)
    assert calculator.compute()['iou'].tolist() == pytest.approx([1 / 3, 1 / 2], abs=1e-5)
    assert calculator.num_samples == 3


def test_epoch_accumulators(calculator):
    accumulators = calculator.create_accumulators('cpu')
    calculator.update_accumulators(accumulators, logits(PREDICTED), LABELS, 3)
    metrics = calculator.compute_epoch_metrics(accumulators, total_loss=3.0, num_batches=2)
    assert metrics['epoch_IoU'].tolist() == pytest.approx([1 / 3, 1 / 2], abs=1e-5)
    assert metrics['pixel_accuracy'] == pytest.approx(0.5, abs=1e-5)
    assert metrics['mean_accuracy'] == pytest.approx(0.75, abs=1e-5)  # mean recall
    assert metrics['epoch_loss'] == 1.5
    assert accumulators['class_pixels'].tolist() == [2, 1]
    # rows are labels, columns predictions
    assert accumulators['confusion_matrix'].tolist() == [[0, 1, 0], [0, 1, 1], [0, 0, 1]]


def test_no_batches_is_zero_loss(calculator):
    metrics = calculator.compute_epoch_metrics(calculator.create_accumulators('cpu'), 0.0, 0)
    assert metrics['epoch_loss'] == 0.0
