"""Render copyable model examples and downloadable JSON from the example configs.

Run after editing configs/examples: python tools/make_training_examples.py
Check for documentation drift: python tools/make_training_examples.py --check
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "clft": "CLFT",
    "clftv2": "CLFTv2",
    "deeplabv3plus": "DeepLabV3+",
    "maskformer": "MaskFormer",
    "mask2former": "Mask2Former",
}


DATASETS = {"zod": "ZOD", "waymo": "Waymo", "iseauto": "Iseauto"}
MODES = {"rgb": "RGB", "lidar": "LiDAR", "fusion": "Fusion"}
DIFFERING = ("Summary", "tags", "CLI", "Dataset", "Log")


def load(model, dataset, mode):
    """The checked-in example config as text and as data."""
    text = (ROOT / "configs/examples" / dataset / model / f"{mode}.json").read_text()
    return text, json.loads(text)


def check_only_expected_keys_differ(model):
    """The page says the files differ in a few keys; fail if a model's examples differ elsewhere."""
    _, reference = load(model, "zod", "fusion")
    for dataset in DATASETS:
        for mode in MODES:
            _, config = load(model, dataset, mode)
            for key in set(reference) | set(config):
                if key not in DIFFERING and reference.get(key) != config.get(key):
                    raise SystemExit(f"{model}/{dataset}/{mode}.json differs from zod/fusion in {key!r}")


def model_page(model, title):
    """One complete example, a table of every download, and what changes between them."""
    check_only_expected_keys_differ(model)
    text, _ = load(model, "zod", "fusion")
    lines = [
        f"# {title} examples",
        "",
        "Each example is a complete JSON file with one epoch, batch size 2, and no pretrained weight download.",
        "[Prepare your dataset](../../download-and-train.md) first, then activate `.venv` and run commands",
        "from the Fusion checkout.",
        "",
        "## The config",
        "",
        f"ZOD with both camera and LiDAR (`configs/examples/zod/{model}/fusion.json`):",
        "",
        f'```json title="configs/examples/zod/{model}/fusion.json"',
        *text.rstrip().splitlines(),
        "```",
        "",
        "## Other datasets and inputs",
        "",
        "Download the file for your dataset and input. Save it under any name and pass it to `-c`.",
        "",
        "| Dataset | Camera only | LiDAR only | Camera + LiDAR |",
        "| --- | --- | --- | --- |",
    ]
    for dataset, label in DATASETS.items():
        links = " | ".join(
            f"[{mode}.json](../../assets/configs/{dataset}/{model}/{mode}.json){{ download }}" for mode in MODES
        )
        lines.append(f"| {label} (`${dataset.upper()}_DATA_DIR`) | {links} |")
    lines += [
        "",
        "The files differ only in these keys:",
        "",
        "| Key | Values |",
        "| --- | --- |",
        "| `CLI.mode` | `rgb`, `lidar`, or `fusion` |",
        "| `Dataset.dataset_root` | `$ZOD_DATA_DIR`, `$WAYMO_DATA_DIR`, or `$ISEAUTO_DATA_DIR`: "
        "export the folder you prepared |",
        "| `Dataset.annotation_path` | ZOD: `annotation_camera_only`, `annotation_lidar_only`, "
        "`annotation_fusion`; Waymo and Iseauto: `annotation` |",
        f"| `Log.logdir` | `logs/<dataset>/{model}/<mode>`, so runs do not overwrite each other |",
        "| `Summary`, `tags` | the run's name and labels in Visin |",
        "",
        "## Run",
        "",
        '```bash title="Run all four stages"',
        f"visin-fusion run -c configs/examples/zod/{model}/fusion.json --upload --benchmark-device cuda",
        "```",
        "",
        "On a CPU host, replace `cuda` with `cpu`. `--upload` enables prediction-image uploads when Visin reporting",
        "is configured. One epoch is a pipeline check, not an accuracy baseline. `extends` names a built-in model",
        "preset, so no parent config file is required.",
        "",
        "**Next:** [Run and inspect the outputs](../../running.md#outputs) or "
        "[customize a run](../../configs.md#customize-an-example).",
        "",
    ]
    return "\n".join(lines)


def artifacts():
    """Produce model pages and matching downloads from the checked-in configs."""
    for model, title in MODELS.items():
        for dataset in DATASETS:
            for mode in MODES:
                relative = f"{dataset}/{model}/{mode}.json"
                yield ROOT / "docs/assets/configs" / relative, load(model, dataset, mode)[0]
        yield ROOT / f"docs/training/models/{model}.md", model_page(model, title)


def main():
    """Write generated documentation, or check it matches the config sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, content in artifacts():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        parser.exit(1, "Run python tools/make_training_examples.py; stale files:\n" + "\n".join(stale) + "\n")


if __name__ == "__main__":
    main()
