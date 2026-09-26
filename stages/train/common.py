#!/usr/bin/env python3
"""Train any model: the model, optimizer, schedule and loss come from models/registry.py, the loop
from core/training_engine.py.

Each epoch is logged to <Log.logdir>/epochs/ and reported to Visin; checkpoints go to
<Log.logdir>/checkpoints/ (the best General.max_checkpoints by validation mIoU are kept).

    python -m stages.train.common -c <config.json> [--seed N]
"""
import argparse
import multiprocessing
import sys

import torch
from torch.utils.data import DataLoader

from core.dataset_png import DatasetPNG
from core.metrics_calculator import MetricsCalculator
from core.training_engine import TrainingEngine
from integrations.vision_service import start_training_run
from models.registry import build_model, training_setup
from utils.config import load_config
from utils.helpers import calculate_num_classes, calculate_num_eval_classes, get_device, get_model_path, set_seed
from utils.metrics import find_overlap_exclude_bg_ignore


def transfer_weights(checkpoint, config, model):
    """Load another run's weights for different classes: matching layers as they are, the class head
    rows remapped by class name (General.source_classes names the checkpoint's classes)."""
    src_state, dst_state = checkpoint['model_state_dict'], model.state_dict()
    source_classes = config['General'].get('source_classes') or []
    target_classes = sorted(config['Dataset']['train_classes'], key=lambda c: c['index'])
    src_index = {c['name']: c['index'] for c in source_classes}

    loaded, skipped, remapped = {}, [], []
    for key, src in src_state.items():
        dst = dst_state.get(key)
        if dst is None:
            skipped.append(key)
        elif src.shape == dst.shape:
            loaded[key] = src
        elif source_classes and src.shape[0] == len(source_classes) and dst.shape[0] == len(target_classes):
            head = dst.clone()
            for cls in target_classes:
                if cls['name'] in src_index:
                    head[cls['index']] = src[src_index[cls['name']]]
            loaded[key] = head
            remapped.append(key)
        else:
            skipped.append(key)
    if remapped:
        print(f"  Head layers remapped for classes {[c['name'] for c in target_classes if c['name'] in src_index]}; "
              f"new (random init): {[c['name'] for c in target_classes if c['name'] not in src_index]}")
    if skipped:
        print(f"  Skipped (unresolvable mismatch): {skipped}")
    dst_state.update(loaded)
    model.load_state_dict(dst_state)


def resume(config, model, setup, device):
    """The epoch to start from: 0, or the one after the checkpoint being resumed."""
    general = config['General']
    if not general['resume_training']:
        print('Training from the beginning')
        return 0
    model_path = get_model_path(config)
    if not model_path:
        print('No checkpoint found, training from the beginning')
        return 0
    print(f'Resuming from {model_path}')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if general.get('transfer_learning', False):
        print('Transfer learning: loading weights with the class head remapped')
        transfer_weights(checkpoint, config, model)
        return 0
    if general['reset_lr']:
        print('Reset the epoch to 0')
        return 0

    finished = checkpoint['epoch']  # 0-based: finished + 1 epochs are done
    if general['epochs'] <= finished + 1:
        sys.exit(f"All {finished + 1} epochs are already trained; set General.epochs higher to continue")
    model.load_state_dict(checkpoint['model_state_dict'])
    setup.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if setup.scheduler is not None and 'scheduler_state_dict' in checkpoint:
        setup.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    elif setup.scheduler is not None:
        print('Warning: the checkpoint has no learning-rate schedule state; the schedule starts over')
    print(f"Resuming from epoch {finished + 1}")
    return finished + 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-c', '--config', required=True, help='config file')
    parser.add_argument('--seed', type=int, help='override General.seed and add _seed<N> to Log.logdir '
                                                 '(for repeated runs of one config)')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.seed is not None:
        config['General']['seed'] = args.seed
        config['Log']['logdir'] = config['Log']['logdir'].rstrip('/') + f'_seed{args.seed}'
        print(f"Seed {args.seed}: logs in {config['Log']['logdir']}")
    set_seed(config['General']['seed'])
    multiprocessing.set_start_method('spawn', force=True)

    # The Visin run: a new one, or the one a resumed training reports into
    run, training_uuid = start_training_run(config, model=config['CLI']['backbone'])
    device = get_device(config)
    print(f"Device: {device}")
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)
    print(f"Classes: {num_classes} ({num_eval_classes} evaluated)")

    model = build_model(config).to(device)
    setup = training_setup(config, model, device)
    metrics = MetricsCalculator(config, num_eval_classes, find_overlap_exclude_bg_ignore)
    start_epoch = resume(config, model, setup, device)

    batch_size = config['General']['batch_size']
    train_loader = DataLoader(DatasetPNG(config, 'train', config['Dataset']['train_split']), batch_size=batch_size,
                              shuffle=True, drop_last=True, pin_memory=True, num_workers=8, persistent_workers=True)
    valid_loader = DataLoader(DatasetPNG(config, 'val', config['Dataset']['val_split']), batch_size=batch_size,
                              shuffle=False, drop_last=False, pin_memory=True, num_workers=8,
                              persistent_workers=True)  # every validation frame, in order

    engine = TrainingEngine(model, setup, metrics, config, training_uuid, config['Log']['logdir'], device, run=run)
    # Leaving the block finishes the Visin run: completed, or failed if training raised
    with run:
        engine.train_full(train_loader, valid_loader, num_classes, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
