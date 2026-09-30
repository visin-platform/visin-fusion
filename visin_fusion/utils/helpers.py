#!/usr/bin/env python3
import glob
import json
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def creat_dir(config):
    logdir = config["Log"]["logdir"]
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        logger.info("Making log directory %s...", logdir)
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


def calculate_num_classes(config):
    """Number of model output classes: one per entry in Dataset.train_classes.

    Relabeled annotations index the model output directly, so class indices must be
    0..n-1 with no gaps or duplicates.
    """
    indices = sorted(cls["index"] for cls in config["Dataset"]["train_classes"])
    if not indices:
        raise ValueError("No training classes defined in config (Dataset.train_classes)")
    if indices != list(range(len(indices))):
        raise ValueError(
            f"Dataset.train_classes indices must be 0..{len(indices) - 1} with no gaps or duplicates, got {indices}"
        )
    return len(indices)


def calculate_num_eval_classes(config, num_classes=None):
    """Number of evaluated classes: every train class except background (index 0)."""
    return sum(1 for cls in config["Dataset"]["train_classes"] if cls["index"] > 0)


def set_seed(seed):
    """Seed Python, NumPy and torch (CPU and CUDA) for a reproducible run.

    DataLoader workers derive their seeds from torch's, so this also fixes shuffling and
    the augmentations in visin_fusion/data/dataset_png.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config):
    """The config's device, or the CPU when CUDA is not available (e.g. a CPU-only container)."""
    return torch.device(config["General"]["device"] if torch.cuda.is_available() else "cpu")


def replace_camera_folder(cam_path, folder, camera="camera"):
    """A frame's camera image path with its camera folder replaced by ``folder``.

    Only the last folder named ``camera`` changes, never the file name or the dataset root,
    e.g. ``/data/zod/camera/frame_1.png`` -> ``/data/zod/lidar_png/frame_1.png``.
    """
    parts = cam_path.split("/")
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == camera:
            return "/".join([*parts[:i], folder, *parts[i + 1 :]])
    raise ValueError(f"no '{camera}' folder in frame path {cam_path}")


def _layout(config):
    return config["Dataset"].get("layout") or {}


def get_annotation_path(cam_path, config):
    """Annotation of a frame: its camera folder replaced by Dataset.annotation_path."""
    folder = config["Dataset"].get("annotation_path") or "annotation"
    return replace_camera_folder(cam_path, folder, _layout(config).get("camera", "camera"))


def get_lidar_path(cam_path, config):
    """LiDAR projection of a frame: its camera folder replaced by the dataset's LiDAR folder."""
    layout = _layout(config)
    return replace_camera_folder(cam_path, layout.get("lidar", "lidar_png"), layout.get("camera", "camera"))


def relabel_annotation(annotation, config):
    """
    Relabel annotation from dataset indices to training indices.

    Uses the new config format with dataset_classes and train_classes.
    Supports class merging through dataset_mapping.

    Args:
        annotation: numpy array or torch tensor with dataset class indices
        config: configuration dictionary with Dataset.train_classes

    Returns:
        torch tensor with training indices [1, H, W]
    """
    annotation = annotation.cpu().numpy() if torch.is_tensor(annotation) else np.asarray(annotation)

    train_classes = config["Dataset"]["train_classes"]

    # Find max dataset index to create mapping array
    max_dataset_index = max(max(mapping) for cls in train_classes for mapping in [cls["dataset_mapping"]])

    # Create mapping from dataset index to training index
    # Default to 0 (background) for unmapped indices, including label values above every mapped
    # class (e.g. 255 as "ignore"), which would otherwise index past the end of the table
    size = max(max_dataset_index, int(annotation.max()) if annotation.size else 0) + 1
    dataset_to_train_mapping = np.zeros(size, dtype=int)

    for train_cls in train_classes:
        train_index = train_cls["index"]
        for dataset_index in train_cls["dataset_mapping"]:
            dataset_to_train_mapping[dataset_index] = train_index

    # Apply mapping
    relabeled = dataset_to_train_mapping[annotation]

    return torch.from_numpy(relabeled).unsqueeze(0).long()  # [H,W]->[1,H,W]


def draw_test_segmentation_map(outputs, config=None):
    """
    Create segmentation visualization with colors based on config class definitions.

    Args:
        outputs: Model output tensor
        config: Configuration dictionary with Dataset.train_classes containing color field.
    """
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()

    # Create color mapping based on config or use default
    if config is not None and "train_classes" in config.get("Dataset", {}):
        train_classes = config["Dataset"]["train_classes"]

        # Create color list for training indices using colors from config
        color_list = []
        for cls in train_classes:
            # Use color from config if available, otherwise use default
            if "color" in cls:
                color_list.append(tuple(cls["color"]))
            else:
                # Fallback to hardcoded colors based on class name
                if cls["name"] == "background":
                    color_list.append((0, 0, 0))  # Black
                elif cls["name"] == "sign":
                    color_list.append((0, 0, 255))  # Blue
                elif cls["name"] == "vehicle":
                    color_list.append((128, 0, 128))  # Purple
                elif cls["name"] == "human":
                    color_list.append((255, 255, 0))  # Yellow
                else:
                    color_list.append((255, 255, 255))  # White fallback

    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(len(color_list)):
        idx = labels == label_num
        red_map[idx] = color_list[label_num][0]
        green_map[idx] = color_list[label_num][1]
        blue_map[idx] = color_list[label_num][2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image


def get_all_checkpoint_paths(config, ignore_model_path=False):
    """Get all checkpoint file paths, sorted by epoch number.

    Args:
        config: Configuration dictionary
        ignore_model_path: If True, ignore config['General']['model_path'] and return all checkpoints
    """
    import glob
    import os

    # If model path is specified and we're not ignoring it, return just that one
    model_path = config["General"].get("model_path", "")
    if model_path != "" and not ignore_model_path:
        return [model_path]

    # Otherwise, find all checkpoints
    checkpoint_dir = os.path.join(config["Log"]["logdir"], "checkpoints")
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if len(files) == 0:
        return []

    # Sort by checkpoint number (not by file creation time which can be unreliable)
    def get_checkpoint_num(filepath):
        try:
            filename = os.path.basename(filepath)
            # Handle both old format (checkpoint_0.pth) and new format (epoch_0_uuid.pth)
            if filename.startswith("checkpoint_"):
                num_str = filename.replace("checkpoint_", "").replace(".pth", "")
            elif filename.startswith("epoch_"):
                # Extract epoch number from epoch_0_uuid.pth format
                parts = filename.replace("epoch_", "").replace(".pth", "").split("_")
                num_str = parts[0] if parts else "0"
            else:
                num_str = "0"
            return int(num_str)
        except ValueError:
            logger.warning("Cannot read an epoch number from checkpoint %s; sorting it as epoch 0", filepath)
            return 0

    # Only the current run's checkpoints: when an earlier run of the config
    # shares this directory, its higher epoch numbers would otherwise win, and
    # resuming would continue the wrong run.
    epochs_dir = os.path.join(config["Log"]["logdir"], "epochs")
    run_uuid = current_training_uuid(epochs_dir)
    if run_uuid:

        def is_current(filepath):
            name = os.path.basename(filepath)
            if not name.startswith("epoch_"):
                return True
            log = os.path.join(epochs_dir, name[: -len(".pth")] + ".json")
            try:
                with open(log) as f:
                    return belongs_to_run(json.load(f), run_uuid)
            except (OSError, ValueError):
                return True  # no log to say otherwise

        files = [f for f in files if is_current(f)]

    # Sort files by epoch number
    sorted_files = sorted(files, key=get_checkpoint_num)
    return sorted_files


def current_training_uuid(epochs_dir):
    """The training UUID in the newest epoch log: the run a logs directory is for now.

    A config trained afresh into a directory an earlier run used shares it with
    that run's logs and checkpoints. Picking the best checkpoint, or pruning,
    across both would test a stale run's checkpoint or delete the new run's.
    """
    files = glob.glob(os.path.join(epochs_dir, "epoch_*.json"))
    if not files:
        return None
    try:
        with open(max(files, key=os.path.getmtime)) as f:
            return json.load(f).get("training_uuid")
    except (OSError, ValueError):
        return None


def belongs_to_run(epoch_data, training_uuid):
    """Whether an epoch log is the given run's; logs that name no run count as anyone's."""
    return not training_uuid or not epoch_data.get("training_uuid") or epoch_data["training_uuid"] == training_uuid


def get_best_checkpoint_path(config):
    """Find the current run's checkpoint with the best validation mIoU."""
    import re

    logdir = config["Log"]["logdir"]
    epochs_dir = os.path.join(logdir, "epochs")

    if not os.path.exists(epochs_dir):
        logger.warning("Epochs directory not found: %s", epochs_dir)
        return None

    best_epoch = None
    best_miou = -1.0
    run_uuid = current_training_uuid(epochs_dir)

    for file in os.listdir(epochs_dir):
        if file.endswith(".json"):
            filepath = os.path.join(epochs_dir, file)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    if not belongs_to_run(data, run_uuid):
                        continue  # an earlier run's, in the same logs directory
                    val_miou = data["results"]["val"].get("mean_iou", 0)
                    if val_miou > best_miou:
                        best_miou = val_miou
                        # Extract epoch and uuid from filename
                        match = re.search(r"epoch_(\d+)_([a-f0-9\-]+)\.json", file)
                        if match:
                            epoch_num = int(match.group(1))
                            epoch_uuid = match.group(2)
                            best_epoch = f"epoch_{epoch_num}_{epoch_uuid}.pth"
            except Exception as e:
                logger.error("Reading %s failed: %s", file, e)
                continue

    if best_epoch:
        checkpoint_path = os.path.join(logdir, "checkpoints", best_epoch)
        if os.path.exists(checkpoint_path):
            logger.info("Found best checkpoint: %s (val mIoU: %.4f)", checkpoint_path, best_miou)
            return checkpoint_path
        logger.warning("Best checkpoint file not found: %s", checkpoint_path)

    return None


def get_model_path(config, best=False):
    """Get the model checkpoint file path. If best=True, get the best checkpoint."""
    if best:
        return get_best_checkpoint_path(config)
    checkpoint_paths = get_all_checkpoint_paths(config)
    if not checkpoint_paths:
        return False
    # Return the latest checkpoint (last in sorted list)
    return checkpoint_paths[-1]


def get_checkpoint_path_with_fallback(config):
    """Get the best checkpoint path, or fall back to the latest checkpoint if best is not found."""
    # Try to get the best checkpoint first
    checkpoint_path = get_best_checkpoint_path(config)
    if checkpoint_path:
        return checkpoint_path

    # Fall back to the latest checkpoint
    checkpoint_paths = get_all_checkpoint_paths(config, ignore_model_path=True)
    if checkpoint_paths:
        return checkpoint_paths[-1]  # Latest checkpoint

    return None


def save_model_dict(config, epoch, model, optimizer, epoch_uuid=None, scheduler=None):
    """Save a checkpoint: weights, optimizer and (so a resumed run continues it) the LR schedule."""
    creat_dir(config)
    if epoch_uuid:
        filename = f"epoch_{epoch}_{epoch_uuid}.pth"
    else:
        filename = f"checkpoint_{epoch}.pth"
    state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, os.path.join(config["Log"]["logdir"], "checkpoints", filename))


def manage_checkpoints_by_miou(config, log_dir):
    """Keep only the top max_checkpoints checkpoints based on validation mIoU from JSON files.

    Only deletes .pth checkpoint files, preserving JSON files for analysis and testing.
    """
    max_checkpoints = config["General"].get("max_checkpoints", 10)
    import glob
    import json
    import os

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    epochs_dir = os.path.join(log_dir, "epochs")

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= max_checkpoints:
        return  # No need to delete anything

    # Get validation mIoU for each checkpoint from JSON files. Only the current
    # run's are ranked, and so only they can be deleted: an earlier run's are
    # left as they are rather than traded against the new run's.
    checkpoints_with_miou = []
    run_uuid = current_training_uuid(epochs_dir)

    for checkpoint_path in checkpoint_files:
        checkpoint_filename = os.path.basename(checkpoint_path)

        # Extract epoch and uuid from checkpoint filename (format: epoch_X_uuid.pth)
        if checkpoint_filename.startswith("epoch_"):
            parts = checkpoint_filename.replace("epoch_", "").replace(".pth", "").split("_")
            if len(parts) >= 2:
                epoch_num = int(parts[0])
                epoch_uuid = parts[1]

                # Find corresponding JSON file
                json_pattern = os.path.join(epochs_dir, f"epoch_{epoch_num}_{epoch_uuid}.json")
                json_files = glob.glob(json_pattern)

                if json_files:
                    try:
                        with open(json_files[0]) as f:
                            json_data = json.load(f)
                        if not belongs_to_run(json_data, run_uuid):
                            continue

                        # Get validation mIoU
                        val_miou = json_data.get("results", {}).get("val", {}).get("mean_iou", -1.0)

                        checkpoints_with_miou.append(
                            {"path": checkpoint_path, "epoch": epoch_num, "uuid": epoch_uuid, "miou": val_miou}
                        )
                    except (json.JSONDecodeError, KeyError, FileNotFoundError):
                        # If we can't read the JSON, treat as lowest priority
                        checkpoints_with_miou.append(
                            {"path": checkpoint_path, "epoch": epoch_num, "uuid": epoch_uuid, "miou": -1.0}
                        )
                else:
                    # No JSON file found, treat as lowest priority
                    checkpoints_with_miou.append(
                        {"path": checkpoint_path, "epoch": epoch_num, "uuid": epoch_uuid, "miou": -1.0}
                    )
            else:
                # Can't parse filename, treat as lowest priority
                checkpoints_with_miou.append({"path": checkpoint_path, "epoch": -1, "uuid": "unknown", "miou": -1.0})
        else:
            # Not an epoch checkpoint, treat as lowest priority
            checkpoints_with_miou.append({"path": checkpoint_path, "epoch": -1, "uuid": "unknown", "miou": -1.0})

    # Sort by validation mIoU descending (highest first)
    checkpoints_with_miou.sort(key=lambda x: x["miou"], reverse=True)

    # Keep only top max_checkpoints
    to_delete = checkpoints_with_miou[max_checkpoints:]

    for checkpoint_info in to_delete:
        try:
            # Delete the checkpoint file only (keep JSON files)
            os.remove(checkpoint_info["path"])
            logger.info(
                "Deleted checkpoint: %s (mIoU: %.4f)",
                os.path.basename(checkpoint_info["path"]),
                checkpoint_info["miou"],
            )
        except OSError as e:
            logger.error("Deleting checkpoint %s failed: %s", checkpoint_info["path"], e)


class EarlyStopping:
    def __init__(self, config):
        self.patience = config["General"]["early_stop_patience"]
        self.config = config
        self.min_param = None
        self.early_stop_trigger = False
        self.count = 0

    def __call__(self, valid_param, epoch, model, optimizer, epoch_uuid=None):
        if self.min_param is None:
            self.min_param = valid_param
        elif valid_param >= self.min_param:
            self.count += 1
            logger.info("Early Stopping Counter: %s of %s", self.count, self.patience)
            if self.count >= self.patience:
                self.early_stop_trigger = True
                logger.info("Saving model for last epoch...")
                save_model_dict(self.config, epoch, model, optimizer, epoch_uuid)
                logger.info("Saving Model Complete")
                logger.info("Early Stopping Triggered!")
        else:
            logger.info("Valid loss decreased from %.4f to %.4f", self.min_param, valid_param)
            self.min_param = valid_param
            # No need to save additional checkpoint - we save every epoch now
            self.count = 0


def sanitize_for_json(data):
    """Recursively sanitize data for JSON serialization (handle NaN/Inf)."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    if isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return data
    if isinstance(data, (np.float32, np.float64)):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return float(data)
    if isinstance(data, (np.int32, np.int64)):
        return int(data)
    return data


def get_training_uuid_from_logs(log_dir):
    """Extract training_uuid and vision_training_id from existing epoch log files."""
    epochs_dir = os.path.abspath(os.path.join(log_dir, "epochs"))
    if not os.path.exists(epochs_dir):
        return None, None

    # Find all epoch JSON files
    epoch_files = glob.glob(os.path.join(epochs_dir, "epoch_*.json"))
    if not epoch_files:
        return None, None

    # The most recently written epoch file: the current run's, even when an
    # earlier run's logs remain here (sorting names put epoch_9 after epoch_10)
    latest_epoch_file = max(epoch_files, key=os.path.getmtime)

    try:
        with open(latest_epoch_file) as f:
            epoch_data = json.load(f)
        training_uuid = epoch_data.get("training_uuid")
        vision_training_id = epoch_data.get("vision_training_id")
        if training_uuid:
            logger.info("Found existing training_uuid from logs: %s", training_uuid)
            if vision_training_id:
                logger.info("Found existing vision_training_id from logs: %s", vision_training_id)
            return training_uuid, vision_training_id
    except Exception as e:
        logger.warning("Could not read training data from %s: %s", latest_epoch_file, e)

    return None, None
