"""Dataset manifests (utils/dataset_manifest.py) and frame paths (utils/helpers.py)."""
import json
from pathlib import Path

import pytest

from utils import splits
from utils.dataset_manifest import apply_manifest
from utils.helpers import get_annotation_path, get_lidar_path, replace_camera_folder

SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'zod_sample'


def write_manifest(root, **fields):
    manifest = {
        'format': 1,
        'name': 'toy',
        'layout': {'camera': 'camera', 'lidar': 'lidar_png'},
        'annotations': ['annotation', 'annotation_fine'],
        'splits': {'train': 'splits/train.txt', 'val': 'splits/val.txt',
                   'test': {'test': 'test.txt'}, 'visualization': 'vis.txt'},
        'classes': [{'name': 'background', 'index': 0}, {'name': 'car', 'index': 1}],
        'normalization': {'lidar_mean': [0.1, 0.2, 0.3], 'lidar_std': [1, 1, 1]},
        **fields,
    }
    (root / 'dataset.json').write_text(json.dumps(manifest))


def test_manifest_fills_what_the_config_leaves_out(tmp_path):
    write_manifest(tmp_path)
    config = apply_manifest({'Dataset': {'dataset_root': str(tmp_path), 'transforms': {'resize': 256}}})
    dataset = config['Dataset']
    assert dataset['name'] == 'toy'
    assert dataset['train_split'] == str(tmp_path / 'splits/train.txt')
    assert dataset['val_split'] == str(tmp_path / 'splits/val.txt')
    assert dataset['split_dir'] == str(tmp_path)
    assert dataset['test_splits'] == {'test': 'test.txt'}
    assert dataset['annotation_path'] == 'annotation'  # the first is the default
    assert dataset['dataset_classes'][1]['name'] == 'car'
    assert dataset['transforms'] == {'resize': 256, 'lidar_mean': [0.1, 0.2, 0.3], 'lidar_std': [1, 1, 1]}


def test_config_wins_over_the_manifest(tmp_path):
    write_manifest(tmp_path)
    config = apply_manifest({'Dataset': {
        'dataset_root': str(tmp_path), 'annotation_path': 'annotation_fine',
        'val_split': '/elsewhere/val.txt', 'transforms': {'lidar_mean': [9, 9, 9]},
    }})
    dataset = config['Dataset']
    assert dataset['annotation_path'] == 'annotation_fine'
    assert dataset['val_split'] == '/elsewhere/val.txt'
    assert dataset['transforms']['lidar_mean'] == [9, 9, 9]


def test_unknown_annotation_is_rejected(tmp_path):
    write_manifest(tmp_path)
    with pytest.raises(ValueError, match='annotation_typo'):
        apply_manifest({'Dataset': {'dataset_root': str(tmp_path), 'annotation_path': 'annotation_typo'}})


def test_unsupported_format_is_rejected(tmp_path):
    write_manifest(tmp_path, format=99)
    with pytest.raises(ValueError, match='format'):
        apply_manifest({'Dataset': {'dataset_root': str(tmp_path)}})


def test_without_a_manifest_the_config_is_unchanged(tmp_path):
    config = {'Dataset': {'dataset_root': str(tmp_path), 'val_split': 'v.txt'}}
    assert apply_manifest(json.loads(json.dumps(config))) == config


def test_sample_dataset_resolves_every_split():
    config = apply_manifest({'Dataset': {'dataset_root': str(SAMPLE), 'annotation_path': 'annotation_camera_only'}})
    assert list(splits.test_splits(config)) == ['day_fair', 'day_rain', 'night_fair', 'night_rain', 'snow']
    assert splits.visualization_split(config) == str(SAMPLE / 'visualizations.txt')
    frame = str(SAMPLE / 'camera' / 'frame_000004.png')
    assert Path(get_annotation_path(frame, config)).exists()
    assert Path(get_lidar_path(frame, config)).exists()


def test_only_the_camera_folder_is_replaced():
    # The old code replaced every "camera" in the path, including the root and file name
    path = '/data/camera_rig/labeled/day/camera/seg_with_camera_labels_01.png'
    assert replace_camera_folder(path, 'lidar_png') == '/data/camera_rig/labeled/day/lidar_png/seg_with_camera_labels_01.png'


def test_nested_annotation_folder():
    config = {'Dataset': {'annotation_path': 'vlm/human_verified/annotation'}}
    assert get_annotation_path('/mnt/zod/camera/f.png', config) == '/mnt/zod/vlm/human_verified/annotation/f.png'


def test_default_folders_without_layout():
    config = {'Dataset': {}}
    assert get_annotation_path('/d/camera/f.png', config) == '/d/annotation/f.png'
    assert get_lidar_path('/d/camera/f.png', config) == '/d/lidar_png/f.png'


def test_frame_path_without_a_camera_folder_fails():
    with pytest.raises(ValueError, match="no 'camera' folder"):
        replace_camera_folder('/d/images/f.png', 'lidar_png')
