#!/usr/bin/env python3
"""Test any model's best checkpoint on every test set of its dataset.

Every frame of every test set is evaluated (visin_fusion/config/splits.py), with the same metrics
for every model (visin_fusion/engine/testing_engine.py). Results go to <Log.logdir>/test_results/ and to Visin,
on the checkpoint's epoch.

    python -m visin_fusion.engine.stages.test.common -c <config.json> [--checkpoint <epoch_N_uuid.pth>]
"""

import argparse
import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from visin_fusion.config.config import load_config
from visin_fusion.config.splits import test_splits
from visin_fusion.data.dataset_png import DatasetPNG
from visin_fusion.engine.metrics_calculator import MetricsCalculator
from visin_fusion.engine.test_aggregator import test_checkpoint_and_save
from visin_fusion.engine.testing_engine import TestingEngine
from visin_fusion.logging_setup import configure_logging
from visin_fusion.models.registry import from_config
from visin_fusion.utils.helpers import (
    calculate_num_classes,
    calculate_num_eval_classes,
    get_checkpoint_path_with_fallback,
    get_device,
    num_workers,
    set_seed,
)
from visin_fusion.utils.metrics import find_overlap_exclude_bg_ignore

logger = logging.getLogger(__name__)

# Averaged over the test sets into results['overall']
OVERALL_KEYS = (
    "mIoU_foreground",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "mean_ap",
    "mean_accuracy",
    "fw_iou",
    "pixel_accuracy",
)


def load_model(config, checkpoint, device):
    """The model with the checkpoint's weights (random init only: the weights come from the checkpoint)."""
    model = from_config(config, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"]
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
        logger.info("\nTesting on %s: %s", name, split)
        dataset = DatasetPNG(config, "test", split)
        if len(dataset) == 0:
            logger.warning("%s lists no frames; skipping", split)
            continue
        loader = DataLoader(
            dataset,
            batch_size=config["General"]["batch_size"],
            shuffle=False,
            num_workers=num_workers(config, len(dataset)),
            pin_memory=True,
        )  # every frame: no drop_last
        results[name], _ = tester.test(loader, config["CLI"]["mode"], num_classes)
    if not results:
        raise RuntimeError("no test set had any frames")

    results["overall"] = {key: float(np.mean([r["overall"][key] for r in results.values()])) for key in OVERALL_KEYS}
    logger.info(
        "\nOverall over %s test sets: mIoU %.4f, mean AP %.4f",
        len(results) - 1,
        results["overall"]["mIoU_foreground"],
        results["overall"]["mean_ap"],
    )
    return results


def main(argv=None):
    """Entry point of the test stage (``python -m visin_fusion.engine.stages.test.common``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True, help="config file")
    parser.add_argument("--checkpoint", help="checkpoint to test (default: the best in Log.logdir)")
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    set_seed(config["General"]["seed"])
    checkpoint = args.checkpoint or get_checkpoint_path_with_fallback(config)
    if not checkpoint:
        sys.exit(f"No checkpoint in {config['Log']['logdir']}/checkpoints; train first")
    test_checkpoint_and_save(checkpoint, test_checkpoint, config, get_device(config))


if __name__ == "__main__":
    main()
