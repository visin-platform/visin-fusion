"""Write one run's config: the template with this cluster's paths, suite and, when online, Hub repo laid over it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Write ``--out``. ``--hub-repo`` is left out when the compute nodes cannot reach the Hub; publish.sh uploads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--suite-file", required=True)
    parser.add_argument("--hub-repo")
    parser.add_argument("--resume", action="store_true", help="continue the logs already in --logdir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    general: dict[str, object] = {"suite_file": args.suite_file}
    if args.resume:
        general["resume_training"] = True
    if args.hub_repo:
        general["hub_repo"] = args.hub_repo
    config = {
        "extends": str(args.template.resolve()),
        "Dataset": {"dataset_root": args.dataset_root},
        "Log": {"logdir": args.logdir},
        "General": general,
    }
    args.out.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
