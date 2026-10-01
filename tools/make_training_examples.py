"""Render copyable model examples and downloadable JSON from the example configs.

Run after editing configs/examples: python tools/make_training_examples.py
Check for documentation drift: python tools/make_training_examples.py --check
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "clft": "CLFT",
    "clftv2": "CLFTv2",
    "deeplabv3plus": "DeepLabV3+",
    "maskformer": "MaskFormer",
    "mask2former": "Mask2Former",
}


def artifacts():
    """Produce model pages and matching downloads from the checked-in configs."""
    for model, title in MODELS.items():
        lines = [
            f"# {title} examples",
            "",
            "Select a dataset and input mode below. Each example is a complete JSON file",
            "with one epoch, batch size 2, and no pretrained weight download.",
            "",
            "[Prepare your dataset](../setup.md#next-prepare-a-dataset) first, then",
            "activate `.venv` and run commands from the Fusion checkout.",
            "",
        ]
        for dataset in ("zod", "waymo", "iseauto"):
            label = "ZOD" if dataset == "zod" else dataset.capitalize()
            lines += [
                f'=== "{label}"',
                "",
                f"    Export `{dataset.upper()}_DATA_DIR` using the [{label} setup](../datasets/{dataset}.md).",
                "",
            ]
            for mode in ("rgb", "lidar", "fusion"):
                relative = f"{dataset}/{model}/{mode}.json"
                source = ROOT / "configs/examples" / relative
                content = source.read_text()
                download = ROOT / "docs/assets/configs" / relative
                yield download, content
                lines += [
                    f'    === "{mode.upper() if mode != "fusion" else "Fusion"}"',
                    "",
                    f"        Save as `configs/examples/{relative}` or use the file already in the checkout.",
                    "",
                    f"        [Download JSON](../../assets/configs/{relative}){{ .md-button download }}",
                    "",
                    f'        ```json title="configs/examples/{relative}"',
                ]
                lines += ["        " + line for line in content.rstrip().splitlines()]
                lines += [
                    "        ```",
                    "",
                    '        ```bash title="Run all four stages"',
                    "        visin-fusion run \\",
                    f"          -c configs/examples/{relative} \\",
                    "          --upload --benchmark-device cuda",
                    "        ```",
                    "",
                ]
        lines += [
            "On a CPU host, replace `cuda` with `cpu`. `--upload` enables prediction-image",
            "uploads when Visin reporting is configured. One epoch is a pipeline check, not",
            "an accuracy baseline.",
            "",
            "Copied or downloaded configs can be saved under any filename: change `-c` to",
            "that path. `extends` names a built-in model preset, so no parent config file",
            "is required. Keep the dataset environment variable exported.",
            "",
            "**Next:** [Run selected stages](../run.md) or [customize a run](../customize.md).",
            "",
        ]
        yield ROOT / f"docs/training/models/{model}.md", "\n".join(lines)


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
