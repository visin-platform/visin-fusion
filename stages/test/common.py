#!/usr/bin/env python3
"""Test any model's best checkpoint on every test set of its dataset.

Every frame of every test set is evaluated (utils/splits.py names the sets), with the same metrics
for every model (core/testing_engine.py). Results go to <Log.logdir>/test_results/ and to Visin,
on the checkpoint's epoch.

    python -m stages.test.common -c <config.json> [--checkpoint <epoch_N_uuid.pth>]
"""
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.dataset_png import DatasetPNG
from core.metrics_calculator import MetricsCalculator
from core.testing_engine import TestingEngine
from models.registry import build_model
from utils.config import load_config
from utils.helpers import (calculate_num_classes, calculate_num_eval_classes, get_checkpoint_path_with_fallback,
                           get_device, set_seed)
from utils.metrics import find_overlap_exclude_bg_ignore
from utils.splits import test_splits
from utils.test_aggregator import test_checkpoint_and_save

# Averaged over the test sets into results['overall']
OVERALL_KEYS = ('mIoU_foreground', 'mean_precision', 'mean_recall', 'mean_f1', 'mean_ap',
                'mean_accuracy', 'fw_iou', 'pixel_accuracy')


def load_model(config, checkpoint, device):
    """The model with the checkpoint's weights (random init only: the weights come from the checkpoint)."""
    model = build_model(config, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict']
    model.load_state_dict(state)  # strict: a checkpoint for another model must not load
    return model.to(device).eval()


def test_checkpoint(checkpoint, config, device):
    """{test set: results} for every test set, plus their average as 'overall'."""
    model = load_model(config, checkpoint, device)
    num_classes = calculate_num_classes(config)
    metrics = MetricsCalculator(config, calculate_num_eval_classes(config), find_overlap_exclude_bg_ignore)
    tester = TestingEngine(model, metrics, config, device)

    results = {}
    for name, split in test_splits(config).items():
        print(f"\nTesting on {name}: {split}")
        dataset = DatasetPNG(config, 'test', split)
        if len(dataset) == 0:
            print(f"{split} lists no frames; skipping")
            continue
        loader = DataLoader(dataset, batch_size=config['General']['batch_size'], shuffle=False,
                            num_workers=4, pin_memory=True)  # every frame: no drop_last
        results[name], _ = tester.test(loader, config['CLI']['mode'], num_classes)
    if not results:
        raise RuntimeError("no test set had any frames")

    results['overall'] = {key: float(np.mean([r['overall'][key] for r in results.values()]))
                          for key in OVERALL_KEYS}
    print(f"\nOverall over {len(results) - 1} test sets: mIoU {results['overall']['mIoU_foreground']:.4f}, "
          f"mean AP {results['overall']['mean_ap']:.4f}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-c', '--config', required=True, help='config file')
    parser.add_argument('--checkpoint', help='checkpoint to test (default: the best in Log.logdir)')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    set_seed(config['General']['seed'])
    checkpoint = args.checkpoint or get_checkpoint_path_with_fallback(config)
    if not checkpoint:
        sys.exit(f"No checkpoint in {config['Log']['logdir']}/checkpoints; train first")
    test_checkpoint_and_save(checkpoint, test_checkpoint, config, get_device(config))


if __name__ == '__main__':
    main()
