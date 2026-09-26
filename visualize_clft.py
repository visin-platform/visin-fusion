#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored visualization script using modular components.
Works with both ZOD and Waymo datasets.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

from core.model_builder import ModelBuilder
from core.data_loader import DataLoader as InferenceDataLoader
from core.visualizer import Visualizer
from utils.helpers import get_model_path, get_annotation_path
from integrations.visualization_uploader import (
    queue_visualizations,
    get_epoch_uuid_from_model_path
)
from integrations.vision_service import attach_to_training, parse_checkpoint_name


def calculate_num_classes(config):
    """
    Calculate number of unique classes after relabeling.
    
    Returns the count of unique training_index values in the config,
    which represents the actual number of output classes for the model
    after any class merging/relabeling.
    """
    unique_indices = set(cls['training_index'] for cls in config['Dataset']['classes'])
    return len(unique_indices)


def load_image_paths(path_arg, dataroot):
    """Load image paths from file or single path."""
    if path_arg.endswith(('.png', '.jpg', '.jpeg')):
        # Single image
        return [path_arg]
    else:
        # Text file with multiple paths
        with open(path_arg, 'r') as f:
            paths = f.read().splitlines()
        return paths


def get_lidar_path(cam_path, dataset_name):
    """Get LiDAR path based on dataset."""
    if dataset_name == 'zod':
        return cam_path.replace('/camera', '/lidar_png')
    else:  # waymo
        return cam_path.replace('/camera/', '/lidar_png/')


def prepare_model_inputs(rgb, lidar, modality):
    """Prepare model inputs based on modality."""
    if modality == 'rgb':
        return rgb, rgb
    elif modality == 'lidar':
        return lidar, lidar
    else:  # cross_fusion
        return rgb, lidar


def process_images(model, data_loader, visualizer, image_paths, dataroot, 
                   modality, dataset_name, device, config, epoch_uuid=None, upload=False):
    """Process and visualize all images."""
    model.eval()
    
    uploaded_count = 0
    failed_count = 0
    run = attach_to_training(config['Log']['logdir'], epoch_uuid=epoch_uuid) if upload and epoch_uuid else None
    epoch_num = parse_checkpoint_name(config['General'].get('model_path') or '')[0] or 0
    
    for idx, path in enumerate(image_paths, 1):
        # Construct full paths
        if os.path.isabs(path):
            cam_path = path
        else:
            cam_path = os.path.join(dataroot, path)
        
        anno_path = get_annotation_path(cam_path, dataset_name, config)
        lidar_path = get_lidar_path(cam_path, dataset_name)
        
        # Verify paths match
        rgb_name = os.path.basename(cam_path).split('.')[0]
        anno_name = os.path.basename(anno_path).split('.')[0]
        lidar_name = os.path.basename(lidar_path).split('.')[0]
        
        assert rgb_name == anno_name, f"RGB and annotation names don't match: {rgb_name} vs {anno_name}"
        assert rgb_name == lidar_name, f"RGB and LiDAR names don't match: {rgb_name} vs {lidar_name}"
        
        # Load data
        print(f'Processing image {idx}/{len(image_paths)}: {rgb_name}')
        rgb = data_loader.load_rgb(cam_path).to(device, non_blocking=True).unsqueeze(0)
        
        # Only load LiDAR data if not in RGB-only mode
        if modality == 'rgb':
            lidar = rgb  # Use RGB as dummy LiDAR input to avoid unnecessary loading
        else:
            lidar = data_loader.load_lidar(lidar_path).to(device, non_blocking=True).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            rgb_input, lidar_input = prepare_model_inputs(rgb, lidar, modality)
            model_outputs = model(rgb_input, lidar_input, modality)
            
            # Process model outputs same as testing engine
            if isinstance(model_outputs, tuple):
                if len(model_outputs) == 2:
                    # (depth, segmentation) format
                    if model_outputs[0] is None:
                        output_seg = model_outputs[1]
                    else:
                        output_seg = model_outputs[0]
                elif len(model_outputs) == 3:
                    # (segmentation, None, None) format
                    output_seg = model_outputs[0]
                else:
                    raise ValueError(f"Unexpected model output format: {len(model_outputs)} outputs")
            else:
                # Single output
                output_seg = model_outputs
            output_seg = output_seg.squeeze(1)
        
        # Visualize
        visualizer.visualize_prediction(output_seg, cam_path, anno_path, idx)
        
        # Upload to Vision API if enabled
        if upload and epoch_uuid:
            print(f'Uploading visualizations for {rgb_name}...')
            image_filename = os.path.basename(cam_path)
            # Queued: sent in the background while the next image renders
            if queue_visualizations(run, epoch_num, epoch_uuid, visualizer.output_base, image_filename):
                uploaded_count += 1
            else:
                failed_count += 1
    
    print(f'\nCompleted visualization of {len(image_paths)} images')
    if run is not None:
        # Waits for the queued uploads; visin logs anything that failed
        run.finish()
    if upload:
        print(f'Upload summary: {uploaded_count} images queued for upload, {failed_count} failed')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize Model Predictions (Refactored)')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--upload', action='store_true',
                       help='Upload visualizations to Vision API')
    parser.add_argument('--epoch-uuid', type=str, default=None,
                       help='Epoch UUID for upload (auto-detected from model if not provided)')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Set visualization path based on dataset
    dataset_name = config['Dataset']['name']
    if dataset_name == 'waymo':
        path = 'waymo_dataset/splits_clft/visualizations.txt'
    elif dataset_name == 'iseauto':
        path = 'xod_dataset/visualization.txt'  # Use dedicated visualization set
    else:
        path = 'zod_dataset/visualizations.txt'
    
    # Setup device
    device = torch.device(config['General']['device']
                         if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get model path
    model_path = get_model_path(config, best=True)
    if not model_path:
        print("No model checkpoint found!")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    config['General']['model_path'] = model_path
    
    # Extract epoch UUID if uploading
    epoch_uuid = args.epoch_uuid
    if args.upload and not epoch_uuid:
        epoch_uuid = get_epoch_uuid_from_model_path(model_path)
        if epoch_uuid:
            print(f"Auto-detected epoch UUID: {epoch_uuid}")
        else:
            print("Warning: Could not auto-detect epoch UUID from model path")
            print("Upload will be skipped unless --epoch-uuid is provided")
            args.upload = False
    
    if args.upload and epoch_uuid:
        print(f"Visualizations will be uploaded to Vision API for epoch: {epoch_uuid}")
    
    # Extract config name for output directory
    dataset_name = config['Dataset']['name']
    log_dir = config['Log']['logdir']
    output_base = os.path.join(log_dir, 'visualizations')
    
    # Build and load model
    model_builder = ModelBuilder(config, device)
    model = model_builder.build_model()
    model, _ = model_builder.load_checkpoint(model, model_path)
    model.eval()
    
    # Setup data loader
    data_loader = InferenceDataLoader(config)
    
    # Setup visualizer
    visualizer = Visualizer(config, output_base)
    
    # Load image paths
    dataroot = os.path.abspath(config['Dataset']['dataset_root'])
    image_paths = load_image_paths(path, dataroot)
    print(f"Found {len(image_paths)} images to process")
    
    # Get modality
    modality = config['CLI']['mode']
    print(f"Using modality: {modality}")
    
    # Process images
    process_images(
        model, data_loader, visualizer, image_paths, 
        dataroot, modality, dataset_name, device, config,
        epoch_uuid=epoch_uuid, upload=args.upload
    )


if __name__ == '__main__':
    main()
