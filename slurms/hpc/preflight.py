"""Check, before a job spends hours, that the paths a config names exist; print every problem found."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from visin_fusion.config.config import load_config
from visin_fusion.config.config_schema import ConfigError


def problems(config: dict, stage: str) -> list[str]:
    """What is missing or wrong for ``stage`` (``train`` or ``test``); an empty list means go."""
    dataset, general = config["Dataset"], config["General"]
    root = Path(dataset["dataset_root"])
    found: list[str] = []

    def need(label: str, path: Path) -> None:
        if not path.exists():
            found.append(f"MISSING {label}: {path}")

    need("dataset_root", root)
    need("dataset.json", root / "dataset.json")
    need("annotation folder", root / dataset["annotation_path"])
    need("camera folder", root / dataset.get("layout", {}).get("camera", "camera"))
    if stage == "train":
        need("train split", Path(dataset["train_split"]))
        need("validation split", Path(dataset["val_split"]))
    else:
        for name, split in dataset["test_splits"].items():
            need(f"test split {name}", Path(dataset["split_dir"]) / split)
    if general.get("suite_file"):
        need("suite file", Path(general["suite_file"]))

    checkpoints = Path(config["Log"]["logdir"]) / "checkpoints"
    has_checkpoints = checkpoints.is_dir() and any(checkpoints.iterdir())
    if stage == "train" and has_checkpoints and not general.get("resume_training"):
        found.append(
            f"{checkpoints} already holds checkpoints from an earlier run. Training into it would mix two runs: "
            "pick a new run id, or continue that run with --resume."
        )
    if stage == "test" and not has_checkpoints:
        found.append(f"no checkpoints in {checkpoints}: training did not produce any")
    return found


def main() -> None:
    """Exit 1, listing every problem, when the config cannot run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--stage", choices=["train", "test"], required=True)
    args = parser.parse_args()
    try:
        found = problems(load_config(args.config), args.stage)
    except (ConfigError, ValueError, FileNotFoundError) as error:
        found = [f"the config cannot be loaded: {error}"]
    for line in found:
        sys.stderr.write(f"  {line}\n")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
