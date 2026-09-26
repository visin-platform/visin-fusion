#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Swin Transformer Fusion model.
"""
import os
import json
import glob
import argparse
import multiprocessing
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from core.advanced_model_builder import AdvancedModelBuilder
from core.metrics_calculator import MetricsCalculator
from core.training_engine import TrainingEngine
from utils.metrics import find_overlap_exclude_bg_ignore
from integrations.vision_service import start_training_run
from utils.helpers import get_model_path


class SwinTrainingEngine(TrainingEngine):
    """TrainingEngine subclass for SwinFusion.

    Adds per-batch gradient clipping and an optional LR scheduler that steps
    once per epoch (warmup + cosine annealing).  All other behaviour — AMP,
    logging, checkpointing — is inherited from the base class.
    """

    def __init__(self, *args, scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler

    def train_epoch(self, dataloader, modality, num_classes):
        """One training epoch with gradient clipping."""
        from torch.amp import autocast
        from utils.helpers import relabel_annotation
        from tqdm import tqdm

        self.model.train()
        accumulators = self.metrics_calc.create_accumulators(self.device)
        train_loss = 0.0

        progress_bar = tqdm(dataloader)
        for batch in progress_bar:
            rgb   = batch['rgb'].to(self.device,  non_blocking=True)
            lidar = batch['lidar'].to(self.device, non_blocking=True)
            anno  = batch['anno'].to(self.device,  non_blocking=True)

            self.optimizer.zero_grad()
            rgb_input, lidar_input = self._prepare_inputs(rgb, lidar, modality)

            with autocast('cuda'):
                model_outputs = self.model(rgb_input, lidar_input, modality)
                output_seg = (
                    model_outputs[1] if model_outputs[0] is None else model_outputs[0]
                ).squeeze(1)
                anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)
                loss = self.criterion(output_seg, anno)

            self.metrics_calc.update_accumulators(accumulators, output_seg, anno, num_classes)
            train_loss += loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            progress_bar.set_description(f'Train loss: {loss:.4f}')

        metrics = self.metrics_calc.compute_epoch_metrics(
            accumulators, train_loss, len(dataloader)
        )
        if self.scheduler is not None:
            self.scheduler.step()
        return metrics


def calculate_num_classes(config):
    """
    Calculate number of training classes.
    
    Returns the count of classes defined in train_classes.
    """
    return len(config['Dataset']['train_classes'])


def calculate_num_eval_classes(config, num_classes):
    """
    Calculate number of evaluation classes (excludes background).
    
    Excludes only class 0 (background) from evaluation.
    All train_classes with index > 0 are evaluated.
    """
    # Count classes with index > 0
    eval_count = sum(1 for cls in config['Dataset']['train_classes'] if cls['index'] > 0)
    return eval_count


def setup_dataset():
    """Setup dataset based on configuration."""
    from tools.dataset_png import DatasetPNG as Dataset
    return Dataset


def setup_criterion(config):
    """Setup loss criterion with class weights."""
    train_classes = config['Dataset']['train_classes']
    
    # Extract weights in order of class index
    sorted_classes = sorted(train_classes, key=lambda x: x['index'])
    class_weights = [cls['weight'] for cls in sorted_classes]
    
    weight_loss = torch.Tensor(class_weights)
    print(f"Using class weights: {class_weights}")
    print(f"For classes: {[cls['name'] for cls in sorted_classes]}")
    
    return nn.CrossEntropyLoss(weight=weight_loss)


def setup_overlap_function(config):
    """Setup dataset-specific overlap calculation function."""
    dataset_name = config['Dataset']['name']
    if dataset_name in ['zod', 'waymo', 'iseauto']:
        print(f"Using unified IoU calculation (excludes background only)")
        return find_overlap_exclude_bg_ignore


def load_checkpoint_if_resume(config, model, optimizer, device):
    """Load checkpoint if resuming training."""
    if not config['General']['resume_training']:
        print('Training from the beginning')
        return 0
    
    model_path = get_model_path(config)
    if not model_path:
        print('No checkpoint found, training from beginning')
        return 0
    
    print(f'Resuming training from {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    is_transfer = config['General'].get('transfer_learning', False)

    if is_transfer:
        print('Transfer learning mode: loading backbone weights with class-aware head remapping')
        src_state = checkpoint['model_state_dict']
        dst_state = model.state_dict()

        source_classes = config['General'].get('source_classes', [])
        target_classes = sorted(config['Dataset']['train_classes'], key=lambda x: x['index'])
        src_name_to_idx = {c['name']: c['index'] for c in source_classes}
        src_num_classes = len(source_classes) if source_classes else None

        filtered = {}
        skipped = []
        remapped = []

        for k, src_v in src_state.items():
            dst_v = dst_state.get(k)
            if dst_v is None:
                skipped.append(k)
                continue
            if src_v.shape == dst_v.shape:
                filtered[k] = src_v
            elif (
                source_classes
                and src_num_classes is not None
                and src_v.shape[0] == src_num_classes
                and dst_v.shape[0] == len(target_classes)
            ):
                # Head layer: remap each row from source class index to target class index by name
                new_tensor = dst_v.clone()
                for tgt_cls in target_classes:
                    src_idx = src_name_to_idx.get(tgt_cls['name'])
                    if src_idx is not None:
                        new_tensor[tgt_cls['index']] = src_v[src_idx]
                filtered[k] = new_tensor
                remapped.append(k)
            else:
                skipped.append(k)

        if remapped:
            mapped_names = ', '.join(
                f"{c['name']}(src {src_name_to_idx[c['name']]}→dst {c['index']})"
                for c in target_classes if c['name'] in src_name_to_idx
            )
            new_names = [c['name'] for c in target_classes if c['name'] not in src_name_to_idx]
            print(f"  Head layers remapped: {mapped_names}")
            if new_names:
                print(f"  New classes (random init): {new_names}")
        if skipped:
            print(f"  Skipped (unresolvable mismatch): {skipped}")

        dst_state.update(filtered)
        model.load_state_dict(dst_state)
        model.to(device)
        return 0

    if config['General']['reset_lr']:
        print('Reset the epoch to 0')
        return 0
    
    finished_epochs = checkpoint['epoch']
    print(f"Finished epochs in previous training: {finished_epochs}")
    
    if config['General']['epochs'] <= finished_epochs:
        print(f'Error: Current epochs ({config["General"]["epochs"]}) <= finished epochs ({finished_epochs})')
        print(f"Please set epochs > {finished_epochs}")
        exit(1)
    
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print('Loading trained optimizer...')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = finished_epochs + 1
    print(f"Resuming from epoch {start_epoch} (next after {finished_epochs})")
    return start_epoch


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Swin Transformer Fusion Training')
    parser.add_argument('-c', '--config', type=str, required=False,
                       default='config.json', help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override General.seed and append _seed<N> to the logdir '
                            '(for multi-seed replication runs)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        config['General']['seed'] = args.seed
        config['Log']['logdir'] = config['Log']['logdir'].rstrip('/') + f'_seed{args.seed}'
        print(f"Seed override: {args.seed}  ->  logdir {config['Log']['logdir']}")

    # Set random seed (numpy AND torch — torch was previously unseeded, so each
    # pre-existing run corresponds to one arbitrary torch draw)
    seed = config['General']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # The Visin run: a new one, or the one a resumed training reports into
    run, training_uuid = start_training_run(config, model=config['CLI']['backbone'])
    
    # Setup device
    device = torch.device(config['General']['device'] 
                         if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Calculate class counts
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)
    print(f"Total classes: {num_classes}, Evaluation classes: {num_eval_classes}")
    
    # Build model
    model_builder = AdvancedModelBuilder(config, device)
    model = model_builder.build_model()
    model.to(device)

    # Setup optimizer — AdamW with weight decay for better regularisation
    weight_decay = config['SwinFusion'].get('weight_decay', 0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['SwinFusion']['clft_lr'],
        weight_decay=weight_decay,
    )
    
    # Setup criterion
    criterion = setup_criterion(config)
    criterion.to(device)

    # LR scheduler: linear warmup then cosine annealing.
    # Disable the legacy momentum-based decay by removing lr_momentum so that
    # adjust_learning_rate() in train_full() becomes a no-op.
    config['SwinFusion'].pop('lr_momentum', None)
    warmup_epochs  = config['SwinFusion'].get('warmup_epochs', 10)
    total_epochs   = config['General']['epochs']
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs]
    )

    # Setup overlap function
    find_overlap_func = setup_overlap_function(config)
    
    # Setup metrics calculator
    metrics_calc = MetricsCalculator(config, num_eval_classes, find_overlap_func)
    
    # Load checkpoint if resuming
    start_epoch = load_checkpoint_if_resume(config, model, optimizer, device)
    
    # Setup datasets
    Dataset = setup_dataset()
    train_data = Dataset(config, 'train', config['Dataset']['train_split'])
    valid_data = Dataset(config, 'val', config['Dataset']['val_split'])
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=config['General']['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True
    )
    
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=config['General']['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True
    )
    
    # Setup training engine
    training_engine = SwinTrainingEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics_calculator=metrics_calc,
        config=config,
        training_uuid=training_uuid,
        log_dir=config['Log']['logdir'],
        device=device,
        run=run,
        scheduler=scheduler,
    )
    
    # Train. Leaving the block finishes the run: completed, or failed if it raised.
    modality = config['CLI']['mode']
    with run:
        training_engine.train_full(
            train_dataloader, 
            valid_dataloader, 
            modality, 
            num_classes,
            start_epoch=start_epoch
        )


if __name__ == '__main__':
    main()