#!/usr/bin/env python3
"""
The training loop, the same for every model: what differs between models (optimizer, learning-rate
schedule, loss, gradient clipping, mixed precision) comes from visin_fusion/models/registry.py:training_setup.
"""

import logging
import os
import time

import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from visin_fusion.engine.callbacks import Checkpoint, EpochEnd, Events
from visin_fusion.engine.epoch_logger import log_epoch_results
from visin_fusion.utils.helpers import EarlyStopping, manage_checkpoints_by_miou, relabel_annotation, save_model_dict
from visin_fusion.utils.system_monitor import get_epoch_system_snapshot

logger = logging.getLogger(__name__)


class TrainingEngine:
    """Trains a model epoch by epoch: logs, checkpoints, early stopping and reports to Visin."""

    def __init__(self, model, setup, metrics_calculator, config, training_uuid, log_dir, device, run=None, events=None):
        self.model = model
        self.setup = setup  # models.registry.TrainingSetup
        self.optimizer = setup.optimizer
        self.scheduler = setup.scheduler
        self.metrics_calc = metrics_calculator
        self.config = config
        self.training_uuid = training_uuid
        self.log_dir = log_dir
        self.device = device
        self.events = events or Events()
        # TensorBoard logs beside the run's other logs, not in ./runs of whatever directory it started in
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
        self.early_stopping = EarlyStopping(config)
        self.mixed_precision = setup.mixed_precision and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.mixed_precision)

    def _batch(self, batch):
        """Inputs and training labels on the device (labels relabelled on the CPU, then sent once)."""
        rgb = batch["rgb"].to(self.device, non_blocking=True)
        lidar = batch["lidar"].to(self.device, non_blocking=True)
        labels = relabel_annotation(batch["anno"], self.config).squeeze(0).to(self.device, non_blocking=True)
        return rgb, lidar, labels

    def _loss(self, rgb, lidar, labels):
        with autocast("cuda", enabled=self.mixed_precision):
            outputs = self.model.raw_forward(rgb, lidar)
            segmap = self.model.segment_from_raw(outputs)
            return segmap, self.setup.loss(outputs, segmap, labels)

    def _optimizer_step(self):
        """Unscale, clip, step and update the scaler for the gradients accumulated so far."""
        if self.setup.clip_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.setup.clip_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_epoch(self, dataloader, num_classes):
        """One training epoch; returns its metrics.

        With ``General.accumulate_batches`` = N the optimizer steps once per N batches, on their average
        gradient (the last step of an epoch averages the batches that remain), so the effective batch is
        N times ``General.batch_size`` at the memory of one.
        """
        self.model.train()
        accumulators = self.metrics_calc.create_accumulators(self.device)
        total_loss = 0.0
        batches = len(dataloader)
        accumulate = max(1, self.config["General"].get("accumulate_batches", 1))
        progress_bar = tqdm(dataloader)
        for index, batch in enumerate(progress_bar):
            group_start = index - index % accumulate
            group_size = min(accumulate, batches - group_start)
            if index == group_start:
                self.optimizer.zero_grad(set_to_none=True)
            rgb, lidar, labels = self._batch(batch)
            segmap, loss = self._loss(rgb, lidar, labels)
            self.metrics_calc.update_accumulators(accumulators, segmap, labels, num_classes)
            total_loss += loss.item()

            self.scaler.scale(loss / group_size).backward()
            if index == group_start + group_size - 1:
                self._optimizer_step()
            progress_bar.set_description(f"Train loss: {loss:.4f}")
        return self.metrics_calc.compute_epoch_metrics(accumulators, total_loss, batches)

    def validate_epoch(self, dataloader, num_classes):
        """One validation epoch, with the training loss; returns its metrics."""
        self.model.eval()
        accumulators = self.metrics_calc.create_accumulators(self.device)
        total_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(dataloader)
            for batch in progress_bar:
                rgb, lidar, labels = self._batch(batch)
                segmap, loss = self._loss(rgb, lidar, labels)
                self.metrics_calc.update_accumulators(accumulators, segmap, labels, num_classes)
                total_loss += loss.item()
                progress_bar.set_description(f"Valid loss: {loss:.4f}")
        return self.metrics_calc.compute_epoch_metrics(accumulators, total_loss, len(dataloader))

    def _step_scheduler(self, val_metrics):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_metrics["mean_iou"])
        else:
            self.scheduler.step()

    def train_full(self, train_dataloader, valid_dataloader, num_classes, start_epoch=0):
        """Train from ``start_epoch`` to General.epochs, or until early stopping."""
        epochs = self.config["General"]["epochs"]
        last_epoch_uuid, last_epoch = None, start_epoch - 1
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            lr = self.optimizer.param_groups[0]["lr"]  # the rate this epoch trains with
            logger.info("Epoch: %s, LR: %.6f", epoch, lr)

            logger.info("Training...")
            train_metrics = self.train_epoch(train_dataloader, num_classes)
            self.metrics_calc.print_metrics(train_metrics, prefix="Training ")

            logger.info("Validating...")
            val_metrics = self.validate_epoch(valid_dataloader, num_classes)
            self.metrics_calc.print_metrics(val_metrics, prefix="Validation ")
            self._step_scheduler(val_metrics)

            epoch_time = time.time() - epoch_start_time
            self._log_tensorboard(train_metrics, val_metrics, epoch)
            epoch_uuid = self._log_and_upload_results(
                epoch, train_metrics, val_metrics, lr, epoch_time, get_epoch_system_snapshot()
            )
            last_epoch_uuid, last_epoch = epoch_uuid, epoch

            logger.info("Saving model checkpoint...")
            self._save(epoch, epoch_uuid)
            self.events.emit(Checkpoint(config=self.config, epoch=epoch, epoch_uuid=epoch_uuid))
            manage_checkpoints_by_miou(self.config, self.log_dir)  # keep the best max_checkpoints

            # Early stopping on the validation mIoU (higher is better; negated for min-tracking)
            self.early_stopping(-round(val_metrics["mean_iou"], 4), epoch, self.model, self.optimizer, epoch_uuid)
            if self.early_stopping.early_stop_trigger:
                break

        # The final checkpoint under the epoch it holds
        logger.info("Saving final model checkpoint...")
        self._save(max(last_epoch, 0), last_epoch_uuid)
        self.events.emit(Checkpoint(config=self.config, epoch=max(last_epoch, 0), epoch_uuid=last_epoch_uuid))
        self.writer.close()
        logger.info("Training Complete")

    def _save(self, epoch, epoch_uuid):
        save_model_dict(self.config, epoch, self.model, self.optimizer, epoch_uuid, scheduler=self.scheduler)

    def _log_tensorboard(self, train_metrics, val_metrics, epoch):
        self.writer.add_scalars(
            "Loss", {"train": train_metrics["epoch_loss"], "valid": val_metrics["epoch_loss"]}, epoch
        )
        for i, cls in enumerate(self.metrics_calc.eval_classes):
            self.writer.add_scalars(
                f"{cls}_IoU", {"train": train_metrics["epoch_IoU"][i], "valid": val_metrics["epoch_IoU"][i]}, epoch
            )
        self.writer.flush()

    def _log_and_upload_results(self, epoch, train_metrics, val_metrics, lr, epoch_time, system_info=None):
        """Log results locally and report them to Visin. Returns the epoch's UUID."""
        if not (self.training_uuid and self.log_dir):
            return None
        results = self.metrics_calc.prepare_results_dict(train_metrics, val_metrics)
        epoch_uuid = log_epoch_results(
            epoch,
            self.training_uuid,
            results,
            self.log_dir,
            learning_rate=lr,
            epoch_time=epoch_time,
            system_info=system_info,
        )
        self.events.emit(
            EpochEnd(
                config=self.config,
                epoch=epoch,
                epoch_uuid=epoch_uuid,
                results=results,
                learning_rate=lr,
                epoch_time=epoch_time,
            )
        )
        return epoch_uuid
