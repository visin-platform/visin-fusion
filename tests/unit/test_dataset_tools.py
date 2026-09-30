"""tools/make_sample_dataset.py and tools/make_manifest.py, run on the sample dataset itself."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "tests" / "data" / "zod_sample"


def dataset_config(tmp_path):
    """A config naming the sample's classes and normalization, as make_manifest needs."""
    from visin_fusion.config.config import load_config

    path = tmp_path / "zod_config.json"
    path.write_text(json.dumps(load_config(REPO / "configs" / "quickstart.json")))
    return path


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, check=True, capture_output=True, text=True)


def test_make_sample_dataset(tmp_path):
    run(
        "tools/make_sample_dataset.py",
        "--source",
        SAMPLE,
        "--splits",
        SAMPLE,
        "--output",
        tmp_path / "small",
        "--scale",
        2,
    )
    small = tmp_path / "small"
    frame = (small / "train.txt").read_text().split()[0]
    assert Image.open(small / frame).size == tuple(round(s / 2) for s in Image.open(SAMPLE / frame).size)
    # labels keep their class values (nearest-neighbour resampling)
    name = Path(frame).name
    original = set(np.unique(np.asarray(Image.open(SAMPLE / "annotation_camera_only" / name))))
    assert set(np.unique(np.asarray(Image.open(small / "annotation_camera_only" / name)))) <= original
    assert (small / "test_snow.txt").exists() and (small / "all.txt").exists()


def test_make_manifest_finds_splits_and_label_folders(tmp_path):
    dataset = tmp_path / "sample"
    shutil.copytree(SAMPLE, dataset)
    (dataset / "dataset.json").unlink()
    output = run("tools/make_manifest.py", "--root", dataset, "--config", dataset_config(tmp_path)).stdout
    manifest = json.loads((dataset / "dataset.json").read_text())
    assert manifest["name"] == "zod"
    assert manifest["splits"]["train"] == "train.txt"
    assert set(manifest["splits"]["test"]) == {"day_fair", "day_rain", "night_fair", "night_rain", "snow"}
    assert manifest["annotations"][0] == "annotation_camera_only"  # the config's choice comes first
    assert "missing" not in output


def test_make_manifest_reports_missing_files(tmp_path):
    dataset = tmp_path / "sample"
    shutil.copytree(SAMPLE, dataset)
    frame = Path((dataset / "train.txt").read_text().split()[0]).name
    (dataset / "lidar_png" / frame).unlink()
    output = run("tools/make_manifest.py", "--root", dataset, "--config", dataset_config(tmp_path)).stdout
    assert "1 files listed by the splits are missing" in output
