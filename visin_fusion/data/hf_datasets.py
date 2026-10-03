"""Datasets on the Hugging Face Hub (https://huggingface.co/datasets), downloaded once by ``huggingface_hub``.

A config names one instead of a folder:

    "Dataset": {"dataset_root": "hf:org/zod-png@3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"}

The repo must hold a ``dataset.json`` manifest at its root, as ``visin push`` and ``visin-fusion dataset``
write it. It is downloaded into the Hub cache (``HF_HOME``) on first use, and later runs use the cached copy.
Pin a commit after ``@``: a branch moves, so a run that names one is not reproducible. A private repo needs
your own ``HF_TOKEN``. Needs ``pip install visin-fusion[hf]``.

For a dataset Visin lists and has pointed at the Hub, ``visin:<name>`` does the same through Visin.
"""

from __future__ import annotations

import logging
import re

from visin_fusion.hub import HubRef, huggingface_module, parse_ref

logger = logging.getLogger(__name__)

PREFIX = "hf:"
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def resolve_root(root: str) -> str:
    """The local folder of the Hub dataset ``hf:org/name[@commit]`` names, downloaded (once) into the Hub cache."""
    ref: HubRef = parse_ref(root)
    if ref.filename:
        raise ValueError(f"{root!r}: a dataset root is a whole repo, not a file inside it")
    if not (ref.revision and COMMIT.match(ref.revision)):
        logger.warning(
            "%s is not pinned to a commit, so a later run may train on different data; use hf:%s@<40-character commit>",
            root,
            ref.repo,
        )
    return str(huggingface_module().snapshot_download(repo_id=ref.repo, repo_type="dataset", revision=ref.revision))
