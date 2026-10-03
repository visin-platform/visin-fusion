#!/usr/bin/env python3
"""Train any model: the model, optimizer, schedule and loss come from visin_fusion/models/registry.py, the loop
from visin_fusion/engine/training_engine.py.

Each epoch is logged to <Log.logdir>/epochs/ and reported to Visin; checkpoints go to
<Log.logdir>/checkpoints/ (the best General.max_checkpoints by validation mIoU are kept).

    python -m visin_fusion.engine.stages.train.common -c <config.json> [--seed N]
"""

import argparse
import logging
import multiprocessing
import sys

import torch
from torch.utils.data import DataLoader

from visin_fusion.config.config import load_config
from visin_fusion.data.dataset_png import DatasetPNG
from visin_fusion.engine.callbacks import RunEnd, RunStart, configured_callbacks
from visin_fusion.engine.epoch_ids import training_uuid_for_run
from visin_fusion.engine.metrics_calculator import MetricsCalculator
from visin_fusion.engine.training_engine import TrainingEngine
from visin_fusion.hub import push_best_checkpoint
from visin_fusion.logging_setup import configure_logging
from visin_fusion.models.registry import from_config, training_setup
from visin_fusion.utils.helpers import (
    calculate_num_classes,
    calculate_num_eval_classes,
    get_device,
    get_model_path,
    num_workers,
    set_seed,
)
from visin_fusion.utils.metrics import find_overlap_exclude_bg_ignore

logger = logging.getLogger(__name__)


def transfer_weights(checkpoint, config, model):
    """Load another run's weights for different classes: matching layers as they are, the class head
    rows remapped by class name (General.source_classes names the checkpoint's classes)."""
    src_state, dst_state = checkpoint["model_state_dict"], model.state_dict()
    source_classes = config["General"].get("source_classes") or []
    target_classes = sorted(config["Dataset"]["train_classes"], key=lambda c: c["index"])
    src_index = {c["name"]: c["index"] for c in source_classes}

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
                if cls["name"] in src_index:
                    head[cls["index"]] = src[src_index[cls["name"]]]
            loaded[key] = head
            remapped.append(key)
        else:
            skipped.append(key)
    if remapped:
        logger.info(
            "  Head layers remapped for classes %s; new (random init): %s",
            [c["name"] for c in target_classes if c["name"] in src_index],
            [c["name"] for c in target_classes if c["name"] not in src_index],
        )
    if skipped:
        logger.warning("  Skipped (unresolvable mismatch): %s", skipped)
    dst_state.update(loaded)
    model.load_state_dict(dst_state)


def resume(config, model, setup, device):
    """The epoch to start from: 0, or the one after the checkpoint being resumed."""
    general = config["General"]
    if not general["resume_training"]:
        logger.info("Training from the beginning")
        return 0
    model_path = get_model_path(config)
    if not model_path:
        logger.info("No checkpoint found, training from the beginning")
        return 0
    logger.info("Resuming from %s", model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if general.get("transfer_learning", False):
        logger.info("Transfer learning: loading weights with the class head remapped")
        transfer_weights(checkpoint, config, model)
        return 0
    if general["reset_lr"]:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Restarting the schedule at epoch 0 with checkpoint weights")
        return 0

    finished = checkpoint["epoch"]  # 0-based: finished + 1 epochs are done
    if general["epochs"] <= finished + 1:
        sys.exit(f"All {finished + 1} epochs are already trained; set General.epochs higher to continue")
    model.load_state_dict(checkpoint["model_state_dict"])
    setup.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if setup.scheduler is not None and "scheduler_state_dict" in checkpoint:
        setup.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    elif setup.scheduler is not None:
        logger.warning("The checkpoint has no learning-rate schedule state; the schedule starts over")
    logger.info("Resuming from epoch %s", finished + 1)
    return finished + 1


def main(argv=None):
    """Entry point of the train stage (``python -m visin_fusion.engine.stages.train.common``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True, help="config file")
    parser.add_argument(
        "--seed",
        type=int,
        help="override General.seed and add _seed<N> to Log.logdir (for repeated runs of one config)",
    )
    args = parser.parse_args(argv)
    configure_logging()

    config = load_config(args.config)
    if args.seed is not None:
        config["General"]["seed"] = args.seed
        config["Log"]["logdir"] = config["Log"]["logdir"].rstrip("/") + f"_seed{args.seed}"
        logger.info("Seed %s: logs in %s", args.seed, config["Log"]["logdir"])
    set_seed(config["General"]["seed"])
    multiprocessing.set_start_method("spawn", force=True)

    device = get_device(config)
    logger.info("Device: %s", device)
    num_classes = calculate_num_classes(config)
    num_eval_classes = calculate_num_eval_classes(config, num_classes)
    logger.info("Classes: %s (%s evaluated)", num_classes, num_eval_classes)

    model = from_config(config).to(device)
    setup = training_setup(config, model, device)
    metrics = MetricsCalculator(config, num_eval_classes, find_overlap_exclude_bg_ignore)
    start_epoch = resume(config, model, setup, device)

    batch_size = config["General"]["batch_size"]
    train_set = DatasetPNG(config, "train", config["Dataset"]["train_split"])
    valid_set = DatasetPNG(config, "val", config["Dataset"]["val_split"])
    train_workers, valid_workers = num_workers(config, len(train_set)), num_workers(config, len(valid_set))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=train_workers,
        persistent_workers=train_workers > 0,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=valid_workers,
        persistent_workers=valid_workers > 0,
    )  # every validation frame, in order

    events = configured_callbacks(config)
    state = {"training_uuid": None}
    events.emit(RunStart(config=config, state=state))
    # A reporting callback may have chosen it; otherwise the same rules, from the local logs
    training_uuid = state["training_uuid"] or training_uuid_for_run(config)[0]
    engine = TrainingEngine(
        model, setup, metrics, config, training_uuid, config["Log"]["logdir"], device, events=events
    )
    try:
        engine.train_full(train_loader, valid_loader, num_classes, start_epoch=start_epoch)
    except BaseException as error:
        events.emit(RunEnd(config=config, error=error))
        raise
    else:
        push_best_checkpoint(config)
        events.emit(RunEnd(config=config))


if __name__ == "__main__":
    main()
