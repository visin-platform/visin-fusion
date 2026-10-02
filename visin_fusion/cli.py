#!/usr/bin/env python3
"""Run a config through its pipeline: train, test, visualize and benchmark.

Each stage is one implementation for every model; the model is the
config's CLI.backbone.
Stages run in order, each as its own process; the first one that fails stops the run
with its exit code.

    visin-fusion run -c config.json                          # all four stages
    visin-fusion run -c config.json --stages test,visualize  # some of them
    visin-fusion run -c config.json --upload                 # send visualizations to Visin
    visin-fusion quickstart                                  # the whole pipeline on a bundled sample
    visin-fusion predict --checkpoint model.pth --input camera/ --output out/   # masks for new images

With --output-dir (or FUSION_OUTPUT_DIR), a relative Log.logdir is placed under it, so a
container can keep every run's logs and checkpoints on a mounted volume.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from visin_fusion.config.config_schema import ConfigError, export_schema
from visin_fusion.dataset_tools import dataset_stats, make_manifest, preview, project_lidar
from visin_fusion.inference import Predictor, save_prediction
from visin_fusion.logging_setup import configure_logging
from visin_fusion.pipeline import STAGES, StageFailed
from visin_fusion.pipeline import run as run_pipeline
from visin_fusion.sample import QUICKSTART_CONFIG
from visin_fusion.utils.helpers import replace_camera_folder


def run(argv=None):
    """Parse the options of ``visin-fusion run`` and run the pipeline (visin_fusion/pipeline.py)."""
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

    configure_logging()
    try:
        run_pipeline(
            args.config,
            stages,
            upload=args.upload,
            benchmark_device=args.benchmark_device,
            output_dir=args.output_dir,
        )
    except (ConfigError, ValueError, FileNotFoundError) as e:
        sys.exit(f"{args.config}: {e}")
    except StageFailed as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(e.returncode)


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def predict(argv=None):
    """Write a mask and an overlay for every camera image in ``--input`` (an image or a folder).

    A LiDAR projection, when the model reads one, is the file of the same name in ``--lidar-dir``, or
    in the ``lidar_png`` folder beside the ``camera`` folder.
    """
    parser = argparse.ArgumentParser(prog="visin-fusion predict", description=predict.__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="a checkpoint written by the train stage")
    parser.add_argument("-c", "--config", help="the training config, for checkpoints from before model_info")
    parser.add_argument("--input", required=True, help="a camera image or a folder of them")
    parser.add_argument("--lidar-dir", help="folder of LiDAR projections named like the camera images")
    parser.add_argument("--output", required=True, help="folder for <name>_mask.png and <name>_overlay.png")
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    source, output = Path(args.input), Path(args.output)
    images = _camera_images(source, output)
    if not images:
        sys.exit(f"No images in {args.input}")
    try:
        predictor = Predictor.from_checkpoint(args.checkpoint, config=args.config, device=args.device)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"{args.checkpoint}: {e}")
    lidars = _lidar_paths(images, predictor, args.lidar_dir)

    for index, (image, lidar) in enumerate(zip(images, lidars, strict=True), 1):
        mask = predictor.predict(None if predictor.mode == "lidar" else image, lidar)
        relative = image.relative_to(source).parent if source.is_dir() else Path()
        save_prediction(predictor, image, mask, output / relative, image.stem)
        print(f"{index}/{len(images)}: {image.name}", flush=True)
    print(f"Wrote {len(images)} masks to {args.output}; classes: {', '.join(predictor.class_names)}")


def _camera_images(source, output):
    """The images ``predict`` reads: ``source`` itself, or every image under it except earlier output."""
    if source.is_file():
        return [source]
    output = output.resolve()
    return sorted(
        p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and output not in p.resolve().parents
    )


def _lidar_paths(images, predictor, lidar_dir):
    """The LiDAR projection of each image (``None`` for a camera-only model); exits naming the first one missing."""
    if predictor.mode == "rgb":
        return [None] * len(images)
    layout = predictor.info.get("layout") or {}
    lidars = []
    for image in images:
        if lidar_dir:
            lidar = Path(lidar_dir) / image.name
        else:
            try:
                lidar = Path(
                    replace_camera_folder(str(image), layout.get("lidar", "lidar_png"), layout.get("camera", "camera"))
                )
            except ValueError:
                sys.exit(f"{image} is not in a 'camera' folder, so its LiDAR projection is unknown: use --lidar-dir")
        if not lidar.is_file():
            sys.exit(f"No LiDAR projection for {image} (looked for {lidar}); use --lidar-dir to say where they are")
        lidars.append(lidar)
    return lidars


def quickstart(argv=None):
    """Train, test, visualize and benchmark CLFTv2 on the sample dataset shipped with the package.

    Takes the options of ``run`` except ``-c``; results go to ``logs/quickstart`` in the working directory.
    """
    with tempfile.TemporaryDirectory(prefix="quickstart-") as tmp:
        path = Path(tmp) / "quickstart.json"
        path.write_text(json.dumps(QUICKSTART_CONFIG))
        return run(["-c", str(path), *(argv or [])])


def dataset(argv=None):
    """Prepare your own dataset: ``manifest`` writes dataset.json, ``project-lidar`` makes lidar_png, ``stats``
    computes the LiDAR normalization and class weights, ``preview`` renders annotations in color."""
    tools = {
        "manifest": make_manifest.main,
        "project-lidar": project_lidar.main,
        "stats": dataset_stats.main,
        "preview": preview.main,
    }
    args = list(argv or [])
    if not args or args[0] not in tools:
        raise SystemExit(f"usage: visin-fusion dataset {'|'.join(tools)} [options]  (add -h for a tool's options)")
    return tools[args[0]](args[1:])


def main(argv=None):
    """Dispatch ``visin-fusion <command>``: run, quickstart, predict, dataset or schema (``--help`` lists them)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "schema":
        if len(args) != 1:
            raise SystemExit("usage: visin-fusion schema")
        schema = export_schema()
        print(json.dumps(schema, indent=2))
        return None
    usage = (
        "usage: visin-fusion quickstart | visin-fusion run -c CONFIG [options] "
        "| visin-fusion predict --checkpoint FILE --input PATH "
        "--output DIR | visin-fusion dataset manifest|project-lidar|stats|preview | visin-fusion schema"
    )
    if not args or args[0] in ("-h", "--help"):
        print(usage)
        return None
    command = args.pop(0)
    if command == "predict":
        return predict(args)
    if command == "quickstart":
        return quickstart(args)
    if command == "dataset":
        return dataset(args)
    if command != "run":
        raise SystemExit(usage)
    return run(args)


if __name__ == "__main__":
    main()
