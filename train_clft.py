#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored training script using modular components.
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

from core.model_builder import ModelBuilder
from core.metrics_calculator import MetricsCalculator
from core.training_engine import TrainingEngine
from utils.metrics import find_overlap_exclude_bg_ignore
from integrations.vision_service import start_training_run
from utils.helpers import get_model_path


def calculate_num_classes(config):
    """
    Calculate number of training classes.
    
    Returns max_index + 1, where max_index is the highest class index in train_classes.
    This ensures the model outputs predictions for all possible class indices.
    """
    train_classes = config['Dataset']['train_classes']
    if not train_classes:
        raise ValueError("No training classes defined in config")
    max_index = max(cls['index'] for cls in train_classes)
    return max_index + 1


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
    
    # resume at next epoch index (previous checkpoint used 0-based numbering)
    start_epoch = finished_epochs + 1
    print(f"Resuming from epoch {start_epoch} (next after {finished_epochs})")
    return start_epoch


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CLFT Training (Refactored)')
    parser.add_argument('-c', '--config', type=str, required=False, 
                       default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set random seed
    np.random.seed(config['General']['seed'])
    
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
    model_builder = ModelBuilder(config, device)
    model = model_builder.build_model()
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config['CLFT']['clft_lr'])
    
    # Setup criterion
    criterion = setup_criterion(config)
    criterion.to(device)
    
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
    training_engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics_calculator=metrics_calc,
        config=config,
        training_uuid=training_uuid,
        log_dir=config['Log']['logdir'],
        device=device,
        run=run
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
