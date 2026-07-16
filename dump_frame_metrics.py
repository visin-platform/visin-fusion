#!/usr/bin/env python3
"""
Per-frame IoU counts for a trained checkpoint against an arbitrary reference
annotation — the basis for common-reference re-evaluation and bootstrap CIs.

The standard test path (test_clft.py) evaluates each variant against the
annotation_path it was trained on, so cross-variant comparisons mix different
references (and, for the clean-partition baseline, different test frames).
This script overrides the reference annotation dir and split dir, evaluates
all frames (no drop_last), and writes per-frame overlap/pred/label/union
counts per class so aggregate metrics and bootstrap confidence intervals can
be computed offline for any frame subset.

Usage:
    python dump_frame_metrics.py -c config/vlm/clftv2-base/llava/config_swin_discovery_fusion.json \
        --splits /run/media/tom/ml/zod_temp/splits_good \
        --reference annotation_camera_only \
        --out logs/vlm/frame_metrics/common_ref/llava_swin_discovery_fusion.json
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.advanced_model_builder import AdvancedModelBuilder
from core.model_builder import ModelBuilder
from tools.dataset_png import DatasetPNG
from utils.metrics import find_overlap_exclude_bg_ignore
from utils.helpers import relabel_annotation, get_checkpoint_path_with_fallback

WEATHER_FILES = ['test_day_fair.txt', 'test_night_fair.txt', 'test_day_rain.txt',
                 'test_night_rain.txt', 'test_snow.txt']


def check_annotations_exist(cam_paths, dataroot, reference):
    missing = [c for c in cam_paths
               if not os.path.exists(os.path.join(dataroot, c.replace('camera', reference)))]
    if missing:
        raise SystemExit(f"{len(missing)} reference annotations missing under '{reference}' "
                         f"(first: {missing[0]}) — refusing to evaluate against dummy labels.")


def evaluate_split(model, config, split_path, device, num_classes):
    """Return per-frame count dicts for one weather split (all frames, in order)."""
    dataset = DatasetPNG(config, 'test', split_path)
    check_annotations_exist(dataset.list_examples_cam, config['Dataset']['dataset_root'],
                            config['Dataset']['annotation_path'])
    loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False,
                        pin_memory=True, num_workers=4)
    modality = config['CLI']['mode']
    frames = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb'].to(device, non_blocking=True)
            lidar = batch['lidar'].to(device, non_blocking=True)
            anno = batch['anno']

            if modality == 'rgb':
                rgb_in, lidar_in = rgb, rgb
            elif modality == 'lidar':
                rgb_in, lidar_in = lidar, lidar
            else:
                rgb_in, lidar_in = rgb, lidar

            outputs = model(rgb_in, lidar_in, modality)
            if isinstance(outputs, tuple):
                output_seg = outputs[1] if (len(outputs) == 2 and outputs[0] is None) else outputs[0]
            else:
                output_seg = outputs
            output_seg = output_seg.squeeze(1)

            anno = relabel_annotation(anno.cpu(), config).squeeze(0).to(device)

            for i in range(output_seg.shape[0]):
                overlap, pred, label, union = find_overlap_exclude_bg_ignore(
                    num_classes, output_seg[i:i + 1], anno[i:i + 1])
                frames.append({
                    'frame': os.path.basename(dataset.list_examples_cam[idx]),
                    'overlap': overlap.cpu().tolist(),
                    'pred': pred.cpu().tolist(),
                    'label': label.cpu().tolist(),
                    'union': union.cpu().tolist(),
                })
                idx += 1
    return frames


def aggregate_miou(frames):
    """mIoU from summed per-frame counts (matches accumulator-based evaluation)."""
    overlap = np.sum([f['overlap'] for f in frames], axis=0)
    union = np.sum([f['union'] for f in frames], axis=0)
    iou = overlap / (union + 1e-6)
    return iou, float(np.mean(iou))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Training config of the checkpoint')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path (default: best from logdir)')
    parser.add_argument('--logdir-suffix', default='',
                        help="Appended to Log.logdir before checkpoint lookup, e.g. '_seed1'")
    parser.add_argument('--splits', required=True, help='Directory containing test_*.txt split files')
    parser.add_argument('--reference', required=True,
                        help="Reference annotation dir relative to dataset root "
                             "(replaces 'camera' in split paths), e.g. annotation_camera_only")
    parser.add_argument('--out', required=True, help='Output JSON path')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Override the evaluation reference — this is the whole point of the script.
    config['Dataset']['annotation_path'] = args.reference
    if args.logdir_suffix:
        config['Log']['logdir'] = config['Log']['logdir'].rstrip('/') + args.logdir_suffix

    device = torch.device(config['General']['device'] if torch.cuda.is_available() else 'cpu')
    num_classes = len(config['Dataset']['train_classes'])

    checkpoint_path = args.checkpoint or get_checkpoint_path_with_fallback(config)
    if not checkpoint_path:
        raise SystemExit('No checkpoint found.')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Reference : {args.reference}   Splits: {args.splits}')

    builder = (AdvancedModelBuilder(config, device) if 'SwinFusion' in config
               else ModelBuilder(config, device))
    model = builder.build_model()
    model, _ = builder.load_checkpoint(model, checkpoint_path)
    model.eval()

    conditions = {}
    weather_mious = []
    for wf in WEATHER_FILES:
        split_path = os.path.join(args.splits, wf)
        if not os.path.exists(split_path):
            print(f'  {wf}: not found, skipping')
            continue
        frames = evaluate_split(model, config, split_path, device, num_classes)
        weather = wf.replace('test_', '').replace('.txt', '')
        conditions[weather] = frames
        iou, miou = aggregate_miou(frames)
        weather_mious.append(miou)
        print(f'  {weather:12s} n={len(frames):4d}  IoU(veh,sign,human)='
              f'{iou[0]:.4f},{iou[1]:.4f},{iou[2]:.4f}  mIoU={miou:.4f}')

    overall = float(np.mean(weather_mious)) if weather_mious else float('nan')
    print(f'  overall mIoU (mean over conditions): {overall:.4f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'config': args.config,
            'checkpoint': checkpoint_path,
            'reference': args.reference,
            'splits': args.splits,
            'eval_classes': ['vehicle', 'sign', 'human'],
            'overall_miou': overall,
            'conditions': conditions,
        }, f)
    print(f'Written: {args.out}')


if __name__ == '__main__':
    main()
