"""A 28-frame sample of ZOD (downscaled 4x) shipped with the package, for the quickstart and the tests.

``QUICKSTART_CONFIG`` is the config ``visin-fusion quickstart`` runs; ``configs/quickstart.json`` in the
repository is the same config with a path relative to the checkout (a test keeps them equal).
"""

from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent / "zod_sample"

QUICKSTART_CONFIG = {
    "extends": "clftv2",
    "Summary": "Quick start: CLFTv2 fusion, ZOD sample",
    "description": "Two short epochs of the CLFTv2 preset on the 28-frame sample dataset shipped with the package.",
    "tags": ["quickstart"],
    "Dataset": {"dataset_root": str(SAMPLE_DIR)},
    "Log": {"logdir": "logs/quickstart"},
    "General": {"epochs": 2, "batch_size": 2, "early_stop_patience": 2, "max_checkpoints": 1},
    "CLFTv2": {"warmup_epochs": 0},
}
