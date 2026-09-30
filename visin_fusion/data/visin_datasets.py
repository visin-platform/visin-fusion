"""Datasets from Visin (https://app.visin.eu/datasets), downloaded once by the visin package.

A config names one by name or id instead of a folder:

    "Dataset": {"dataset_root": "visin:zod"}

The first stage that needs it downloads and unpacks it (``visin.Datasets``: resumable, checked against
the size Visin reports), later runs use the local copy. Or beforehand, with the visin command:

    visin datasets
    visin download zod

The dataset service is VISIN_DATASET_URL, by default this project's deployment below. Datasets go to
VISIN_DATA_DIR, by default ~/.cache/visin/datasets (the image sets /data).
"""

import os
import sys

PREFIX = "visin:"
# The dataset service this project uses unless VISIN_DATASET_URL says otherwise; the visin package
# has no default address, on purpose (like VISIN_URL in visin_fusion/integrations/visin.py)
DEFAULT_DATASET_URL = "https://dataset-api.visin.eu"


def resolve_root(root):
    """``dataset_root`` as a local folder: ``visin:<name or id>`` is downloaded (once), anything else
    is returned as it is."""
    if not (isinstance(root, str) and root.startswith(PREFIX)):
        return root
    try:
        import visin
    except ImportError as exc:
        raise ImportError("A visin: dataset needs pip install visin-fusion[visin]") from exc

    visin.enable_console_logging(stream=sys.stdout)  # the download's progress, in order with the stage's
    with visin.Datasets(url=os.environ.get("VISIN_DATASET_URL") or DEFAULT_DATASET_URL) as datasets:
        return str(datasets.download(root[len(PREFIX) :]))
