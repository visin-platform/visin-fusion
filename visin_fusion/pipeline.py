"""Run a config through train, test, visualize and benchmark from Python.

    import visin_fusion

    logdir = visin_fusion.run("my_config.json")                              # all four stages
    logdir = visin_fusion.run({"extends": "clftv2", ...}, stages=["train"])  # a config dict, some stages

This is what ``visin-fusion run`` does. Each stage runs as its own process (a failed or out-of-memory stage
cannot take the caller down, and GPU memory is released between stages), in order; the first that fails stops
the run with :class:`StageFailed`. The config is checked as a whole before any stage starts, and an invalid
one raises :class:`~visin_fusion.config.config_schema.ConfigError` or ``ValueError``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence

from visin_fusion.config.config import load_config, prepare_config

STAGE_ROOT = "visin_fusion.engine.stages"
STAGES = ("train", "test", "visualize", "benchmark")

logger = logging.getLogger(__name__)


class StageFailed(RuntimeError):
    """A stage process exited with a non-zero code; ``stage`` and ``returncode`` say which and how."""

    def __init__(self, stage: str, returncode: int) -> None:
        super().__init__(f"{stage} failed with exit code {returncode}")
        self.stage = stage
        self.returncode = returncode


def stage_command(stage: str, config_path: str, upload: bool = False, benchmark_device: str | None = None) -> list[str]:
    """The command line for one stage; every model shares each stage (stages/<stage>/common.py)."""
    command = [sys.executable, "-m", f"{STAGE_ROOT}.{stage}.common", "-c", config_path]
    if stage == "visualize" and upload:
        command.append("--upload")
    if stage == "benchmark" and benchmark_device:
        command += ["--single", "--device", benchmark_device]
    return command


def run(
    config: Mapping | str | os.PathLike,
    stages: Sequence[str] = STAGES,
    *,
    upload: bool = False,
    benchmark_device: str | None = None,
    output_dir: str | os.PathLike | None = None,
) -> str:
    """Run ``stages`` of ``config`` in order; returns the run's log directory.

    ``config`` is a config file or dict (its paths may be ``pathlib.Path``), resolved like
    ``visin-fusion run -c``: presets, ``extends``, the dataset manifest and the schema's defaults. Relative
    paths are relative to the working directory (or to the config file's folder for ``extends``). With
    ``output_dir``, a relative ``Log.logdir`` is placed under it. ``upload`` sends visualizations to Visin and
    ``benchmark_device`` (``"cpu"`` or ``"cuda"``) limits the benchmark to one device.
    """
    stages = list(stages)
    unknown = [s for s in stages if s not in STAGES]
    if unknown or not stages:
        raise ValueError(f"stages must be a non-empty selection of {list(STAGES)}, got {stages}")

    if isinstance(config, Mapping):
        resolved = prepare_config(json.loads(json.dumps(config, default=os.fspath)), os.getcwd())
    else:
        resolved = load_config(os.path.abspath(config))
    if output_dir and not os.path.isabs(resolved["Log"]["logdir"]):
        resolved["Log"]["logdir"] = os.path.join(os.path.abspath(output_dir), resolved["Log"]["logdir"])
    logdir = resolved["Log"]["logdir"]

    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="run-config-", delete=False) as f:
        json.dump(resolved, f, indent=2)
        config_path = f.name
    logger.info("Pipeline %s: %s; logs in %s", resolved["CLI"]["backbone"], ", ".join(stages), logdir)

    try:
        for stage in stages:
            command = stage_command(stage, config_path, upload, benchmark_device)
            logger.info("\n=== %s: %s", stage, " ".join(command[1:]))
            result = subprocess.run(command)
            if result.returncode != 0:
                raise StageFailed(stage, result.returncode)
    finally:
        os.remove(config_path)
    logger.info("\nDone: %s", ", ".join(stages))
    return logdir
