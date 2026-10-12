"""Print the local folder of a dataset root such as ``visin:zod`` or ``hf:org/name@commit``, downloading it first."""

from __future__ import annotations

import sys

from visin_fusion.data.visin_datasets import resolve_root


def main() -> None:
    """Resolve the root named on the command line, so a node without internet can use the folder it prints."""
    sys.stdout.write(f"{resolve_root(sys.argv[1])}\n")


if __name__ == "__main__":
    main()
