"""Inference checkpoints stored in Visin, with a stable file identity shared by exports and evaluations.

The optional Visin SDK is loaded only when publishing or downloading. Full training checkpoints retain their
optimizer state; an inference copy beside the logs keeps only weights, model_info, and the epoch. Once created,
it is reused rather than serialized again, since the SHA-256 names the actual published bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from visin_fusion.hub import CHECKPOINT_NAME, stage_for_hub
from visin_fusion.utils.helpers import get_best_checkpoint_path

logger = logging.getLogger(__name__)
SCHEME = "visin://"
EXTRA_HINT = "Visin models require the visin extra and a current SDK: pip install -U 'visin-fusion[visin]'"


def file_digest(path: str | Path) -> str:
    """The SHA-256 of a checkpoint without reading all its bytes into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_checkpoint(checkpoint: str | Path) -> Path:
    """The stable inference copy of a full training checkpoint, or that file itself when already inference-only.

    The copy is independent of checkpoint rotation. A digest of the original detects a changed file at the same
    path; its inference copy is replaced only then. The metadata also verifies that an old copy is intact.
    """
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "optimizer_state_dict" not in state and "scheduler_state_dict" not in state:
        return checkpoint
    source_digest = file_digest(checkpoint)
    folder = checkpoint.parent.parent / "published" / checkpoint.stem
    target = folder / CHECKPOINT_NAME
    record = folder / "publication.json"
    if record.is_file() and target.is_file():
        try:
            saved = json.loads(record.read_text())
            if saved.get("originalSha256") == source_digest and saved.get("sha256") == file_digest(target):
                return target
        except (OSError, ValueError, AttributeError):
            logger.warning("Rebuilding an unreadable publication record for %s", checkpoint)
    stage_for_hub(checkpoint, folder)
    record.write_text(
        json.dumps(
            {"originalSha256": source_digest, "sha256": file_digest(target), "epoch": state.get("epoch")}, indent=2
        )
    )
    return target


def publish_best_checkpoint(config: Mapping[str, Any]) -> str | None:
    """Save the best checkpoint through the current Visin run when General.publish_model is visin.

    Publication never raises into training. The inference copy persists under Log.logdir/published, where ONNX
    export and evaluation can reuse its exact bytes. Checkpoint rotation keeps the highest-mIoU checkpoint;
    publication runs after training while its Visin run is still open.
    """
    general = config["General"]
    if general.get("publish_model") != "visin":
        return None
    if not general.get("model_license"):
        logger.warning("General.model_license is unset: the stored model will be refused for download (409)")
    try:
        import visin

        run = visin.get_run()
        if not run.enabled or not hasattr(run, "save_model"):
            logger.warning("No enabled Visin run can save this model. %s", EXTRA_HINT)
            return None
        checkpoint = get_best_checkpoint_path(config)
        if not checkpoint:
            logger.warning("General.publish_model is set but no best checkpoint was found")
            return None
        path = inference_checkpoint(checkpoint)
        state = torch.load(path, map_location="cpu", weights_only=True)
        digest = run.save_model(
            path,
            epoch=state.get("epoch"),
            licence=general.get("model_license"),
            credit=general.get("model_credit"),
            model_info=state["model_info"],
        )
        if digest:
            logger.info("Saved epoch %s on Visin: %s (SHA-256 %s)", state.get("epoch"), path, digest)
        return str(digest) if digest else None
    except Exception as exc:
        logger.warning("Publishing the best checkpoint to Visin failed: %s", exc)
        return None


def download_checkpoint(ref: str, directory: str | Path) -> Path:
    """Download and verify visin://<training UUID>/<model id> using the optional SDK."""
    parts = ref.removeprefix(SCHEME).split("/")
    if not ref.startswith(SCHEME) or len(parts) != 2 or not all(parts):
        raise ValueError("Expected visin://<training UUID>/<model id>")
    try:
        import visin
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    if not hasattr(visin.Api, "download_model"):
        raise ImportError(EXTRA_HINT)
    with visin.Api() as api:
        return Path(api.download_model(parts[0], model=parts[1], directory=directory, file_format="pth"))


def publish_export(
    result: Mapping[str, Any], training_uuid: str, *, licence: str | None = None, credit: str | None = None
) -> str | None:
    """Queue a parity-checked ONNX export in its run, using the saved inference weights as its checkpoint identity."""
    try:
        import visin
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    if not hasattr(visin.Run, "save_model"):
        raise ImportError(EXTRA_HINT)
    if not result["parity"]["frames"]:
        raise ValueError("publishing an ONNX model requires parity on actual sample frames")
    with visin.Run.attach(training_uuid, mark_status=False) as run:
        return run.save_model(
            result["path"],
            epoch=result["epoch"],
            file_format="onnx",
            licence=licence,
            credit=credit,
            model_info=result["model_info"],
            source_sha256=result["sourceSha256"],
            parity=result["parity"],
        )
