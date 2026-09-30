"""Dataset manifests: a dataset describes itself in ``dataset.json`` at its root.

A config then only has to point at the dataset (``Dataset.dataset_root``) and make the
experiment's choices; everything the manifest says fills in what the config leaves out.
Keys the config sets always win, so configs without a manifest work as before.

    {
        "format": 1,
        "name": "zod",
        "description": "Zenseact Open Dataset, 2,300 labelled frames",
        "layout": {"camera": "camera", "lidar": "lidar_png"},
        "annotations": ["annotation_camera_only", "annotation_lidar_only", "annotation_fusion"],
        "splits": {
            "train": "train.txt",
            "val": "validation.txt",
            "test": {"day_fair": "test_day_fair.txt", "snow": "test_snow.txt"},
            "visualization": "visualizations.txt"
        },
        "classes": [{"name": "background", "index": 0}, {"name": "vehicle", "index": 2}],
        "train_classes": [{"name": "background", "index": 0, "weight": 0.1, "dataset_mapping": [0],
                           "color": [0, 0, 0]}, ...],
        "normalization": {"lidar_mean": [0.25, 0.49, 0.50], "lidar_std": [0.23, 0.03, 0.17]}
    }

Split files list frames relative to the dataset root (e.g. ``camera/frame_000004.png`` or
``labeled/day/rain/camera/x.png``). The LiDAR projection and the annotation of a frame are
found by replacing its ``camera`` folder with the ``lidar`` folder or an annotation folder;
the first of ``annotations`` is the default. Split file names are relative to the root.
``train_classes`` is the class merging the dataset suggests for training; a config can set its own.
"""

import json
import os

MANIFEST = "dataset.json"
FORMAT = 1


def load_manifest(dataset_root):
    """The manifest at the dataset root, or None if it has none."""
    path = os.path.join(dataset_root, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported manifest format {manifest.get('format')!r} (expected {FORMAT})")
    return manifest


def apply_manifest(config):
    """Fill the config's Dataset section from the dataset's manifest, if it has one.

    Returns the config (changed in place). Keys already in the config are kept.
    """
    dataset = config.get("Dataset", {})
    root = dataset.get("dataset_root")
    manifest = load_manifest(root) if root else None
    if manifest is None:
        return config

    def in_root(name):
        return name if os.path.isabs(name) else os.path.join(root, name)

    splits = manifest.get("splits", {})
    defaults = {
        "name": manifest.get("name"),
        "dataset_classes": manifest.get("classes"),
        "train_classes": manifest.get("train_classes"),
        "layout": manifest.get("layout"),
        "split_dir": root,
        "train_split": in_root(splits["train"]) if "train" in splits else None,
        "val_split": in_root(splits["val"]) if "val" in splits else None,
        "test_splits": splits.get("test"),
        "visualization_split": splits.get("visualization"),
        "annotation_path": (manifest.get("annotations") or [None])[0],
    }
    for key, value in defaults.items():
        if value is not None:
            dataset.setdefault(key, value)

    transforms = dataset.setdefault("transforms", {})
    for key, value in (manifest.get("normalization") or {}).items():
        transforms.setdefault(key, value)

    annotations = manifest.get("annotations")
    if annotations and dataset.get("annotation_path") not in annotations:
        raise ValueError(
            f"Dataset.annotation_path {dataset.get('annotation_path')!r} is not one of the annotations "
            f"of {os.path.join(root, MANIFEST)}: {annotations}"
        )
    config["Dataset"] = dataset
    return config
