"""Write the Visin suite for a dataset's test splits: one condition per split, pinned by a digest of its frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import visin


def main() -> None:
    """Write ``--out`` for the dataset folder ``root``. The digest and counts come from the split files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.root / "dataset.json").read_text())
    frames = {
        name: [line.strip() for line in (args.root / path).read_text().splitlines() if line.strip()]
        for name, path in manifest["splits"]["test"].items()
    }
    suite = {
        "slug": args.slug,
        "version": args.version,
        "name": f"{manifest['name']} test sets",
        "protocol": {
            "task": "semantic-segmentation",
            "data": {
                "kind": "external",
                "label": f"{manifest['name']} test splits",
                "manifestSha256": visin.manifest_digest(frames),
            },
            "split": "test",
            "conditions": [{"name": name, "sampleCount": len(items)} for name, items in frames.items()],
            "metrics": [
                {"key": "mIoU_foreground", "direction": "max", "range": {"min": 0, "max": 1}, "headline": True},
                {"key": "mean_ap", "direction": "max", "range": {"min": 0, "max": 1}},
            ],
            "aggregation": "equal-mean-of-conditions",
            "evaluator": {"package": "visin-fusion"},
        },
    }
    args.out.write_text(json.dumps(suite, indent=2) + "\n")
    counts = ", ".join(f"{name} ({len(items)})" for name, items in frames.items())
    sys.stdout.write(f"wrote {args.out}: {counts}\n")


if __name__ == "__main__":
    main()
