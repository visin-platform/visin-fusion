import json
import logging
import math
from datetime import datetime
from pathlib import Path

from visin_fusion.engine.epoch_ids import epoch_uuid_for

logger = logging.getLogger(__name__)


def clean_nan_values(obj):
    """
    Recursively replace NaN values with 0.0 in nested dictionaries/lists.

    Args:
        obj: The object to clean (dict, list, or primitive)

    Returns:
        The cleaned object with NaN values replaced
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return 0.0
    return obj


def log_epoch_results(
    epoch, training_uuid, results, log_dir, learning_rate=None, epoch_time=None, system_info=None, run=None
):
    """
    Log training and validation results for a specific epoch, and report them to Visin.

    Args:
        epoch (int): Current epoch number
        training_uuid (str): UUID for the entire training run
        results (dict): Dictionary containing training and validation results per class
                       Expected structure: {
                           "train": {"class_name": {"metric": value, ...}, ...},
                           "val": {"class_name": {"metric": value, ...}, ...}
                       }
        log_dir (str or Path): Directory to save the log files
        learning_rate (float, optional): Learning rate for this epoch
        epoch_time (float, optional): Time taken for this epoch in seconds
        system_info (dict, optional): System resource usage snapshot
        run (visin.Run, optional): The Visin run to report the epoch to

    Returns:
        str: The epoch's UUID. It is the same one Visin records, and it names the
        epoch's log file and checkpoint, which is how the test and visualization
        scripts find the epoch later.
    """
    epoch_uuid = epoch_uuid_for(training_uuid, epoch)

    data = {
        "training_uuid": training_uuid,
        "epoch_uuid": epoch_uuid,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    if learning_rate is not None:
        data["learning_rate"] = learning_rate
    if epoch_time is not None:
        data["epoch_time"] = epoch_time
    if system_info is not None:
        # Visin's System tab reads it from the epoch's results
        data["results"]["system_info"] = system_info

    # Clean NaN values from the data before writing
    data = clean_nan_values(data)

    # Create epochs subfolder
    log_dir = Path(log_dir)
    epochs_dir = log_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"epoch_{epoch}_{epoch_uuid}.json"
    filepath = epochs_dir / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Epoch %s results logged to %s", epoch, filepath)

    return epoch_uuid
