"""Datasets from Visin (https://app.visin.eu/datasets), downloaded once by the visin package.

A config names one by name or id instead of a folder:

    "Dataset": {"dataset_root": "visin:zod"}

The first stage that needs it downloads and unpacks it (``visin.Datasets``: resumable, checked against
the size Visin reports), later runs use the local copy. Or beforehand, with the visin command:

    visin datasets
    visin download zod

The dataset service is VISIN_DATASET_URL, required for visin: roots. Datasets go to
VISIN_DATA_DIR, by default ~/.cache/visin/datasets (the image sets /data).
"""

from __future__ import annotations

import os
import sys
from typing import Any

from visin_fusion.integrations.settings import load_environment

PREFIX = "visin:"


def resolve_root(root: Any) -> Any:
    """``dataset_root`` as a local folder: ``visin:<name or id>`` is downloaded (once), anything else
    is returned as it is."""
    if not (isinstance(root, str) and root.startswith(PREFIX)):
        return root
    load_environment()
    url = os.environ.get("VISIN_DATASET_URL")
    if not url:
        raise ValueError("A visin: dataset requires VISIN_DATASET_URL; set it in VISIN_ENV_FILE or the environment")
    try:
        import visin
    except ImportError as exc:
        raise ImportError("A visin: dataset needs pip install visin-fusion[visin]") from exc

    visin.enable_console_logging(stream=sys.stdout)  # the download's progress, in order with the stage's
    with visin.Datasets(url=url) as datasets:
        return str(datasets.download(root[len(PREFIX) :]))
