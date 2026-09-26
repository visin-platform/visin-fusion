"""stages/benchmark/common.py: one benchmark for every model."""
import json
from pathlib import Path

import pytest
import torch

from stages.benchmark import common

SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'zod_sample'


def write_config(tmp_path, preset='deeplabv3plus', **extra):
    config = {'extends': preset, 'CLI': {'mode': 'rgb'}, 'Dataset': {'dataset_root': str(SAMPLE)},
              'Log': {'logdir': str(tmp_path / 'logs')}, **extra}
    path = tmp_path / f'{preset}.json'
    path.write_text(json.dumps(config))
    return path


def test_benchmark_config_measures_the_model(tmp_path):
    result = common.benchmark_config(str(write_config(tmp_path)), torch.device('cpu'), num_runs=2, warmup_runs=1)
    assert result['backbone'] == 'deeplabv3plus' and result['modality'] == 'rgb'
    assert result['total_parameters_m'] > 1
    assert result['num_runs'] == 2 and result['mean_time_ms'] > 0 and result['fps'] > 0
    assert 'ram_memory_mean_mb' in result and 'gpu_memory_peak_mb' not in result  # CPU run


def test_profiling_leaves_the_model_in_eval_mode():
    # thop restores the training flag of the module it profiles; timing a model in training
    # mode would be wrong (and fails for BatchNorm with batch size 1)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    segmenter = torch.nn.Module()
    segmenter.forward = lambda rgb, lidar: model(rgb)
    segmenter.add_module('model', model)
    segmenter.eval()
    common.count_flops(segmenter, torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))
    assert not model.training


def test_results_are_saved_and_linked_to_the_newest_epoch(tmp_path, monkeypatch):
    monkeypatch.setenv('VISIN_MODE', 'disabled')
    config = write_config(tmp_path)
    epochs = tmp_path / 'logs' / 'epochs'
    epochs.mkdir(parents=True)
    (epochs / 'epoch_3_abc.json').write_text(json.dumps({'epoch': 3, 'epoch_uuid': 'abc', 'training_uuid': 'run-1'}))
    common.main(['-c', str(config), '--single', '--device', 'cpu', '--num-runs', '2', '--warmup-runs', '1'])
    saved = json.loads(next((tmp_path / 'logs' / 'benchmark').glob('*.json')).read_text())
    assert (saved['epoch'], saved['epoch_uuid'], saved['training_uuid']) == (3, 'abc', 'run-1')
    assert len(saved['results']) == 1


def test_a_failing_config_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv('VISIN_MODE', 'disabled')
    good = write_config(tmp_path)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'extends': 'deeplabv3plus', 'Dataset': {'dataset_root': str(SAMPLE)},
                               'Log': {'logdir': str(tmp_path / 'logs')}, 'General': {'epcohs': 1}}))
    with pytest.raises(SystemExit, match='failed for 1 of 2'):
        common.main(['-c', str(good), str(bad), '--single', '--device', 'cpu', '--num-runs', '1', '--warmup-runs', '0'])
    assert list((tmp_path / 'logs' / 'benchmark').glob('*.json'))  # the good one was still saved
