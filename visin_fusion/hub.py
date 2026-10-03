"""Trained models and datasets on the Hugging Face Hub, behind the ``hf`` extra.

A model repo holds what a stranger needs to run the model and nothing else:

    checkpoint.pth   the weights, with ``model_info`` (no optimizer state)
    config.json      that same ``model_info``, so the Hub shows how the model was built

``Predictor.from_pretrained("hf://org/clftv2-zod")`` and ``visin-fusion predict --checkpoint hf://...``
read it. Setting ``General.hub_repo`` in a config makes the train stage publish its best checkpoint there
when it finishes, and Visin, when reporting is on, links the repo to the run.

A reference is ``hf://org/name``, optionally pinned to a commit with ``@<commit>`` and naming a file
inside the repo with a trailing path: ``hf://org/name@3f2a1c9/best.pth``. Pinning is how a result stays
reproducible: a branch moves, a commit does not.

``huggingface_hub`` is imported lazily, because it is optional (``pip install visin-fusion[hf]``). The
user's own token is read by that package (``HF_TOKEN`` or ``huggingface-cli login``); nothing here stores it.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from visin_fusion.utils.helpers import get_best_checkpoint_path

logger = logging.getLogger(__name__)

SCHEME = "hf://"
CHECKPOINT_NAME = "checkpoint.pth"
CONFIG_NAME = "config.json"
EXTRA_HINT = "Hugging Face support is an extra: pip install 'visin-fusion[hf]'"


@dataclass(frozen=True)
class HubRef:
    """A repo on the Hub, optionally pinned to a revision and naming one file in it."""

    repo: str
    revision: str | None = None
    filename: str | None = None


def parse_ref(ref: str) -> HubRef:
    """``hf://org/name[@revision][/path/in/repo]`` as a :class:`HubRef`; the ``hf://`` (or ``hf:``) is optional."""
    text = ref.removeprefix("hf:").removeprefix("//").strip("/")
    org, _, rest = text.partition("/")
    name, _, filename = rest.partition("/")
    name, _, revision = name.partition("@")
    if not (org and name):
        raise ValueError(f"{ref!r} is not a Hugging Face reference; expected hf://org/name[@commit][/file]")
    return HubRef(f"{org}/{name}", revision or None, filename or None)


def huggingface_module() -> Any:
    """``huggingface_hub``, imported on first use because it is an optional dependency."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    return huggingface_hub


def download_checkpoint(ref: HubRef | str) -> Path:
    """The local path of a checkpoint in a model repo, downloaded once into the Hub cache.

    A named file wins. A folder or whole repo uses its ``checkpoint.pth``, otherwise its only ``.pth``/``.pt``.
    Raises ``FileNotFoundError`` when the repo has none, or several and no file was named.
    """
    ref = parse_ref(ref) if isinstance(ref, str) else ref
    hub = huggingface_module()
    filename = ref.filename
    files = hub.HfApi().list_repo_files(ref.repo, revision=ref.revision)
    prefix = f"{filename.rstrip('/')}/" if filename and filename not in files else ""
    is_folder = bool(prefix and any(name.startswith(prefix) for name in files))
    if filename is None or is_folder:
        candidates = [name for name in files if name.startswith(prefix) and name.endswith((".pth", ".pt"))]
        conventional = f"{prefix}{CHECKPOINT_NAME}"
        if conventional in files:
            filename = conventional
        elif len(candidates) == 1:
            filename = candidates[0]
        else:
            found = ", ".join(candidates) or "none"
            raise FileNotFoundError(
                f"{ref.repo} has no {CHECKPOINT_NAME} and not exactly one .pth/.pt file (found: {found}); "
                f"name one: hf://{ref.repo}/<file>"
            )
    return Path(hub.hf_hub_download(ref.repo, filename, revision=ref.revision))


def stage_for_hub(checkpoint: str | Path, destination: str | Path) -> Path:
    """Write the folder a model repo should hold for ``checkpoint``: inference weights and ``config.json``.

    The optimizer and scheduler state a training checkpoint carries is left out; it is most of the file and
    useless to anyone but the run that wrote it. Returns ``destination``.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info = state["model_info"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": state.get("epoch"), "model_state_dict": state["model_state_dict"], "model_info": info},
        destination / CHECKPOINT_NAME,
    )
    (destination / CONFIG_NAME).write_text(json.dumps(info, indent=2, default=str))
    return destination


def push_best_checkpoint(config: Any) -> str | None:
    """Publish the run's best checkpoint to ``General.hub_repo``; return the commit, or ``None``.

    Does nothing when no repo is configured. A failure is logged, never raised: the model is trained and
    its checkpoints are on disk, and a refused upload should not turn that into a failed run. With Visin
    reporting on, the upload goes through ``Run.log_model``, so the run links the repo at that commit and
    Visin writes its model card; otherwise it is uploaded directly.
    """
    general = config["General"]
    repo = general.get("hub_repo")
    if not repo:
        return None
    checkpoint = get_best_checkpoint_path(config)
    if not checkpoint:
        logger.warning("General.hub_repo is set but no checkpoint was found to publish")
        return None
    try:
        with tempfile.TemporaryDirectory() as staging:
            stage_for_hub(checkpoint, staging)
            epoch = torch.load(checkpoint, map_location="cpu", weights_only=True).get("epoch")
            return _upload(Path(staging), str(repo), epoch, private=bool(general.get("hub_private", True)))
    except Exception as exc:
        logger.warning("Publishing the best checkpoint to %s failed: %s", repo, exc)
        return None


def _visin_run() -> Any:
    """The current Visin run when reporting is on and the installed ``visin`` can link models, else ``None``."""
    try:
        import visin
    except ImportError:
        return None
    get_run = getattr(visin, "get_run", None)
    run = get_run() if get_run else None
    return run if run is not None and run.enabled and hasattr(run, "log_model") else None


def _upload(folder: Path, repo: str, epoch: int | None, *, private: bool) -> str | None:
    """Upload ``folder`` to ``repo`` through the current Visin run when there is one, else directly."""
    run = _visin_run()
    if run is not None:
        revision = run.log_model(folder, repo, epoch=epoch, private=private)
        if revision:
            logger.info("Published the best checkpoint to %s at %s", repo, revision[:7])
        return str(revision) if revision else None
    hub = huggingface_module()
    api = hub.HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder))
    logger.info("Published the best checkpoint to %s at %s", repo, str(info.oid)[:7])
    return str(info.oid)
