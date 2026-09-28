"""Checkpoint continuation and class-head transfer with real torch state dictionaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visin_fusion.engine.stages.train import common as train


def config_for(path, *, resume=True, reset=False, transfer=False, epochs=4):
    return {
        'General': {
            'resume_training': resume, 'model_path': str(path), 'reset_lr': reset,
            'transfer_learning': transfer, 'epochs': epochs,
            'source_classes': [{'name': 'background', 'index': 0},
                               {'name': 'vehicle', 'index': 1}],
        },
        'Dataset': {'train_classes': [
            {'name': 'background', 'index': 0}, {'name': 'human', 'index': 1},
            {'name': 'vehicle', 'index': 2},
        ]},
        'Log': {'logdir': str(path.parent)},
    }


def setup_for(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return SimpleNamespace(optimizer=optimizer, scheduler=scheduler)


def test_transfer_weights_remaps_matching_class_rows_and_keeps_new_classes(capsys):
    model = torch.nn.Linear(2, 3)
    before = model.state_dict()['weight'][1].clone()
    checkpoint = {'model_state_dict': {
        'weight': torch.tensor([[1., 2.], [3., 4.]]),
        'bias': torch.tensor([5., 6.]),
        'extra': torch.tensor([9.]),
    }}
    config = config_for(Path('/tmp/model.pth'))
    train.transfer_weights(checkpoint, config, model)
    assert torch.equal(model.weight[0], torch.tensor([1., 2.]))
    assert torch.equal(model.weight[1], before)
    assert torch.equal(model.weight[2], torch.tensor([3., 4.]))
    assert torch.equal(model.bias[[0, 2]], torch.tensor([5., 6.]))
    assert 'extra' in capsys.readouterr().out


def test_transfer_weights_skips_unresolvable_shapes(tmp_path):
    model = torch.nn.Linear(2, 3)
    original = model.weight.detach().clone()
    checkpoint = {'model_state_dict': {'weight': torch.ones(4, 2),
                                       'bias': torch.ones(3)}}
    config = config_for(tmp_path / 'source.pth')
    config['General']['source_classes'] = []
    train.transfer_weights(checkpoint, config, model)
    assert torch.equal(model.weight, original)
    assert torch.equal(model.bias, torch.ones(3))


def test_resume_starts_fresh_without_request_or_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 3)
    setup = setup_for(model)
    path = tmp_path / 'missing.pth'
    assert train.resume(config_for(path, resume=False), model, setup, 'cpu') == 0
    config = config_for(path)
    config['General']['model_path'] = ''
    assert train.resume(config, model, setup, 'cpu') == 0


def test_resume_restores_model_optimizer_and_scheduler(tmp_path):
    source = torch.nn.Linear(2, 3)
    source_setup = setup_for(source)
    source_setup.optimizer.step()
    source_setup.scheduler.step()
    path = tmp_path / 'checkpoint_1.pth'
    torch.save({'epoch': 1, 'model_state_dict': source.state_dict(),
                'optimizer_state_dict': source_setup.optimizer.state_dict(),
                'scheduler_state_dict': source_setup.scheduler.state_dict()}, path)
    target = torch.nn.Linear(2, 3)
    target_setup = setup_for(target)
    assert train.resume(config_for(path), target, target_setup, 'cpu') == 2
    assert torch.equal(target.weight, source.weight)
    assert target_setup.optimizer.param_groups[0]['lr'] == source_setup.optimizer.param_groups[0]['lr']
    assert target_setup.scheduler.last_epoch == source_setup.scheduler.last_epoch

    with pytest.raises(SystemExit, match='already trained'):
        train.resume(config_for(path, epochs=2), target, target_setup, 'cpu')


def test_resume_reset_and_transfer_start_at_epoch_zero(tmp_path):
    path = tmp_path / 'checkpoint_1.pth'
    source = torch.nn.Linear(2, 2)
    torch.save({'epoch': 1, 'model_state_dict': source.state_dict()}, path)

    reset_target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        reset_target.weight.zero_()
    reset_setup = setup_for(reset_target)
    assert train.resume(config_for(path, reset=True), reset_target, reset_setup, 'cpu') == 0
    assert torch.equal(reset_target.weight, source.weight)
    assert reset_setup.optimizer.param_groups[0]['lr'] == 0.1

    transfer_target = torch.nn.Linear(2, 3)
    transfer_setup = setup_for(transfer_target)
    assert train.resume(config_for(path, transfer=True), transfer_target, transfer_setup, 'cpu') == 0
    assert torch.equal(transfer_target.weight[2], source.weight[1])


def test_resume_without_scheduler_state_warns(tmp_path, capsys):
    model = torch.nn.Linear(2, 3)
    setup = setup_for(model)
    path = tmp_path / 'checkpoint_0.pth'
    torch.save({'epoch': 0, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': setup.optimizer.state_dict()}, path)
    assert train.resume(config_for(path), model, setup, 'cpu') == 1
    assert 'no learning-rate schedule state' in capsys.readouterr().out
