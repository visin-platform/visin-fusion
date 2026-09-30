#!/usr/bin/env python3
"""Run a config through its pipeline: train, test, visualize and benchmark.

Each stage is one implementation for every model; the model is the
config's CLI.backbone.
Stages run in order, each as its own process; the first one that fails stops the run
with its exit code.

    visin-fusion run -c config.json                          # all four stages
    visin-fusion run -c config.json --stages test,visualize  # some of them
    visin-fusion run -c config.json --upload                 # send visualizations to Visin

With --output-dir (or FUSION_OUTPUT_DIR), a relative Log.logdir is placed under it, so a
container can keep every run's logs and checkpoints on a mounted volume.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

from visin_fusion.config.config import load_config
from visin_fusion.config.config_schema import ConfigError

STAGE_ROOT = "visin_fusion.engine.stages"
STAGES = ("train", "test", "visualize", "benchmark")


def stage_command(stage, config_path, upload=False, benchmark_device=None):
    """The command line for one stage; every model shares each stage (stages/<stage>/common.py)."""
    command = [sys.executable, "-m", f"{STAGE_ROOT}.{stage}.common", "-c", config_path]
    if stage == "visualize" and upload:
        command.append("--upload")
    if stage == "benchmark" and benchmark_device:
        command += ["--single", "--device", benchmark_device]
    return command


def run(argv=None):
    parser = argparse.ArgumentParser(prog="visin-fusion run", description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True, help="config file")
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated stages to run, in order (default: {','.join(STAGES)})",
    )
    parser.add_argument("--upload", action="store_true", help="upload visualizations to Visin")
    parser.add_argument(
        "--benchmark-device",
        choices=["cpu", "cuda"],
        help="benchmark on this device only (default: CPU and GPU if available)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("FUSION_OUTPUT_DIR"),
        help="directory a relative Log.logdir is placed under (default: $FUSION_OUTPUT_DIR)",
    )
    args = parser.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages:
        parser.error("select at least one stage")
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        parser.error(f"unknown stages {unknown}; choose from {list(STAGES)}")

    # Child stages inherit the caller's working directory for relative dataset paths.
    args.config = os.path.abspath(args.config)
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)

    # The whole config is checked before any stage starts
    try:
        config = load_config(args.config)
    except (ConfigError, ValueError, FileNotFoundError) as e:
        sys.exit(f"{args.config}: {e}")
    backbone = config["CLI"]["backbone"]  # checked against the known models by load_config

    if args.output_dir and not os.path.isabs(config["Log"]["logdir"]):
        config["Log"]["logdir"] = os.path.join(args.output_dir, config["Log"]["logdir"])
    # Every stage gets the resolved config: the dataset manifest applied, the log directory placed
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="run-config-", delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name
    print(f"Pipeline {backbone}: {', '.join(stages)}; logs in {config['Log']['logdir']}", flush=True)

    try:
        for stage in stages:
            command = stage_command(stage, config_path, args.upload, args.benchmark_device)
            print(f"\n=== {stage}: {' '.join(command[1:])}", flush=True)
            result = subprocess.run(command)
            if result.returncode != 0:
                print(f"\n{stage} failed with exit code {result.returncode}", file=sys.stderr)
                sys.exit(result.returncode)
    finally:
        os.remove(config_path)  # the resolved config written above
    print(f"\nDone: {', '.join(stages)}")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "schema":
        if len(args) != 1:
            raise SystemExit("usage: visin-fusion schema")
        from visin_fusion.config.config_schema import export_schema

        schema = export_schema()
        print(json.dumps(schema, indent=2))
        return None
    if not args or args[0] in ("-h", "--help"):
        print("usage: visin-fusion run -c CONFIG [options] | visin-fusion schema")
        return None
    if args.pop(0) != "run":
        raise SystemExit("usage: visin-fusion run -c CONFIG [options] | visin-fusion schema")
    return run(args)


if __name__ == "__main__":
    main()
