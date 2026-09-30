"""Stable local training and epoch IDs, compatible with visin-py.

Epoch logs and checkpoint names carry these IDs, so they are chosen the same way whether or not
Visin is configured.
"""

import glob
import json
import logging
import os
import re
import uuid

logger = logging.getLogger(__name__)

_EPOCH_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "epochs.visin")
_CHECKPOINT_NAME = re.compile(r"epoch_(\d+)_([0-9a-fA-F-]{36})\.pth$")


def epoch_uuid_for(training_uuid, epoch):
    """The UUID an epoch of a run always has (the same as visin.epoch_uuid_for)."""
    return str(uuid.uuid5(_EPOCH_NS, f"{training_uuid}:{epoch}"))


def parse_checkpoint_name(path):
    """The epoch number and epoch UUID in a checkpoint's file name, if it has them."""
    match = _CHECKPOINT_NAME.search(os.path.basename(path))
    return (int(match.group(1)), match.group(2)) if match else (None, None)


def training_uuid_for_epoch(log_dir, epoch_uuid):
    """The training an epoch belongs to, from its log file (logs/.../epochs/epoch_{n}_{uuid}.json)."""
    for path in glob.glob(os.path.join(log_dir, "epochs", f"epoch_*_{epoch_uuid}.json")):
        try:
            with open(path) as handle:
                training_uuid = json.load(handle).get("training_uuid")
        except (OSError, ValueError):
            continue
        if training_uuid:
            return training_uuid
    return None


def training_uuid_for_checkpoint(checkpoint_path):
    """The training a checkpoint belongs to, from its epoch's log beside the checkpoints directory."""
    _, epoch_uuid = parse_checkpoint_name(checkpoint_path)
    if not epoch_uuid:
        return None
    log_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    return training_uuid_for_epoch(log_dir, epoch_uuid)


def training_uuid_for_run(config):
    """The training UUID for this run, and whether it continues an earlier one found in the logs.

    - A fresh training gets a new UUID.
    - A resumed one (General.resume_training) takes the UUID of the checkpoint it resumes, or else
      the latest in the epoch logs of Log.logdir.
    - General.create_new_training or General.transfer_learning forces a new UUID, and so does
      General.reset_lr: its epochs start again from 0, and reusing the UUID would give them the
      epoch UUIDs (and checkpoint names) of the earlier run's epochs.
    """
    from visin_fusion.utils.helpers import get_model_path, get_training_uuid_from_logs

    general = config["General"]
    fresh = general.get("create_new_training") or general.get("transfer_learning") or general.get("reset_lr")
    training_uuid = None
    if general.get("resume_training") and not fresh:
        # The run of the checkpoint being resumed, which the logs directory may
        # share with an earlier run of the same config.
        checkpoint = get_model_path(config)
        training_uuid = training_uuid_for_checkpoint(checkpoint) if checkpoint else None
        training_uuid = training_uuid or get_training_uuid_from_logs(config["Log"]["logdir"])[0]
        if not training_uuid:
            logger.info("No earlier run in the epoch logs; starting a new one")
    if training_uuid is not None:
        return training_uuid, True
    return str(uuid.uuid4()), False
