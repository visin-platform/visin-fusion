"""IoU areas, class relabelling, and how checkpoints and epoch logs name their run."""
import json

import numpy as np
import pytest
import torch

from integrations.vision_service import parse_checkpoint_name, training_uuid_for_epoch
from utils.helpers import relabel_annotation
from utils.metrics import find_overlap_exclude_bg_ignore


def logits_for(prediction, n_classes):
    """Logits whose argmax is ``prediction`` ([B, H, W] of class indices)."""
    return torch.nn.functional.one_hot(prediction, n_classes).permute(0, 3, 1, 2).float()


def test_iou_areas_by_hand():
    # classes: 0 background, 1, 2
    anno = torch.tensor([[[0, 1, 1, 2]]])
    pred = torch.tensor([[[1, 1, 2, 2]]])
    overlap, predicted, label, union = find_overlap_exclude_bg_ignore(3, logits_for(pred, 3), anno)
    assert overlap.tolist() == [1, 1]      # class 1: pixel 1; class 2: pixel 3
    assert predicted.tolist() == [2, 2]
    assert label.tolist() == [2, 1]
    assert union.tolist() == [3, 2]        # IoU 1/3 and 1/2


def test_background_is_not_evaluated():
    anno = torch.zeros(1, 2, 2, dtype=torch.long)
    overlap, predicted, label, union = find_overlap_exclude_bg_ignore(3, logits_for(anno, 3), anno)
    assert overlap.tolist() == [0, 0] and label.tolist() == [0, 0]
    assert (union > 0).all()  # clamped, so IoU is 0 rather than a division by zero


TRAIN_CLASSES = {'Dataset': {'train_classes': [
    {'name': 'background', 'index': 0, 'dataset_mapping': [0, 1]},
    {'name': 'vehicle', 'index': 1, 'dataset_mapping': [2]},
    {'name': 'human', 'index': 2, 'dataset_mapping': [4, 5]},
]}}


def test_relabel_merges_dataset_classes():
    anno = np.array([[0, 1, 2, 4, 5]])
    assert relabel_annotation(anno, TRAIN_CLASSES).tolist() == [[[0, 0, 1, 2, 2]]]


def test_relabel_turns_unmapped_classes_into_background():
    # 3 is between mapped classes; 255 is above all of them (it used to raise IndexError)
    anno = np.array([[3, 255, 2]])
    assert relabel_annotation(anno, TRAIN_CLASSES).tolist() == [[[0, 0, 1]]]


@pytest.mark.parametrize('name, expected', [
    ('logs/x/checkpoints/epoch_12_7a437f6c-a2c3-5389-b500-62e165b628c4.pth',
     (12, '7a437f6c-a2c3-5389-b500-62e165b628c4')),
    ('checkpoint_3.pth', (None, None)),
    ('model.pth', (None, None)),
])
def test_checkpoint_names(name, expected):
    assert parse_checkpoint_name(name) == expected


def test_training_is_found_from_an_epoch_log(tmp_path):
    epochs = tmp_path / 'epochs'
    epochs.mkdir()
    (epochs / 'epoch_0_abc.json').write_text(json.dumps({'training_uuid': 'run-1', 'epoch_uuid': 'abc'}))
    assert training_uuid_for_epoch(str(tmp_path), 'abc') == 'run-1'
    assert training_uuid_for_epoch(str(tmp_path), 'other') is None


def test_epoch_uuids_are_deterministic():
    import visin
    first = visin.epoch_uuid_for('c331414c-2513-49b7-b2f3-b41763307f0c', 0)
    assert first == visin.epoch_uuid_for('c331414c-2513-49b7-b2f3-b41763307f0c', 0)
    assert first != visin.epoch_uuid_for('c331414c-2513-49b7-b2f3-b41763307f0c', 1)
