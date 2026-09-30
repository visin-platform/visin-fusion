"""Shared metrics: average precision (visin_fusion/utils/metrics.py) and class counts (utils/helpers.py)."""

import pytest
import torch

from visin_fusion.utils.helpers import calculate_num_classes, calculate_num_eval_classes, set_seed
from visin_fusion.utils.metrics import compute_ap_for_class, store_predictions_for_ap, voc_ap


def store(logits, anno, classes=("a",), indices=(1,)):
    predictions = {name: [] for name in classes}
    targets = {name: [] for name in classes}
    store_predictions_for_ap(logits, anno, predictions, targets, list(classes), list(indices))
    return predictions, targets


def test_voc_ap_of_perfect_ranking_is_one():
    recall = torch.tensor([0.5, 1.0])
    precision = torch.tensor([1.0, 1.0])
    assert voc_ap(recall, precision) == pytest.approx(1.0)


def test_voc_ap_by_hand():
    # Ranked targets [1, 0, 1]: precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1.
    # Precision made monotone: 1 up to recall 1/2, then 2/3 up to 1.
    recall = torch.tensor([0.5, 0.5, 1.0])
    precision = torch.tensor([1.0, 0.5, 2 / 3])
    assert voc_ap(recall, precision) == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)


def test_voc_ap_of_nothing_is_zero():
    assert voc_ap(torch.tensor([]), torch.tensor([])) == 0.0


def test_ap_of_a_perfect_prediction():
    logits = torch.full((1, 3, 4, 4), -5.0)
    anno = torch.zeros(1, 4, 4, dtype=torch.long)
    anno[0, :2] = 1
    logits[0, 1, :2] = 5
    logits[0, 0, 2:] = 5
    predictions, targets = store(logits, anno)
    assert compute_ap_for_class("a", predictions, targets) == pytest.approx(1.0, abs=1e-5)


def test_ap_is_deterministic_and_uses_every_pixel():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 3, 64, 64, generator=generator)
    anno = torch.randint(0, 3, (4, 64, 64), generator=generator)
    predictions, targets = store(logits, anno)
    relevant = ((logits.argmax(1) == 1) | (anno == 1)).sum().item()
    assert sum(len(p) for p in predictions["a"]) == relevant  # no sampling
    first = compute_ap_for_class("a", predictions, targets)
    assert compute_ap_for_class("a", predictions, targets) == first


def test_stored_pixels_are_on_the_cpu():
    predictions, _ = store(torch.randn(1, 3, 4, 4), torch.ones(1, 4, 4, dtype=torch.long))
    assert predictions["a"][0].device.type == "cpu"


def test_ap_of_a_class_never_seen_is_zero():
    assert compute_ap_for_class("a", {"a": []}, {"a": []}) == 0.0


def classes(*indices):
    return {"Dataset": {"train_classes": [{"name": f"c{i}", "index": i} for i in indices]}}


def test_class_counts():
    config = classes(0, 1, 2, 3)
    assert calculate_num_classes(config) == 4
    assert calculate_num_eval_classes(config) == 3


def test_class_indices_with_a_gap_are_rejected():
    with pytest.raises(ValueError, match="no gaps or duplicates"):
        calculate_num_classes(classes(0, 1, 3))


def test_duplicate_class_indices_are_rejected():
    with pytest.raises(ValueError, match="no gaps or duplicates"):
        calculate_num_classes(classes(0, 1, 1))


def test_no_classes_is_rejected():
    with pytest.raises(ValueError, match="No training classes"):
        calculate_num_classes(classes())


def test_set_seed_makes_torch_repeatable():
    set_seed(7)
    first = torch.rand(3)
    set_seed(7)
    assert torch.equal(torch.rand(3), first)
