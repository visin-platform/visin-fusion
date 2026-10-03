"""Predict segmentation masks with a trained checkpoint, without a config or the pipeline.

    from visin_fusion.inference import Predictor

    predictor = Predictor.from_checkpoint("logs/run/checkpoints/epoch_9_<uuid>.pth")
    mask = predictor.predict("camera/000001.png", "lidar_png/000001.png")   # [H, W] uint8 class indices
    overlay = predictor.overlay("camera/000001.png", mask)                   # [H, W, 3] uint8 RGB

Checkpoints written by this release carry what is needed to rebuild the model and preprocess its inputs
(``model_info``). For an older checkpoint pass the config that trained it: ``Predictor.from_checkpoint(path,
config=...)``, a config dict or the path of a config file.

A model published to the Hugging Face Hub loads the same way: ``Predictor.from_pretrained("hf://org/name")``.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from visin_fusion.config.config import load_config, prepare_config
from visin_fusion.config.model_info import config_from_info, model_info
from visin_fusion.data.preprocessing import ImageInput, Preprocessor
from visin_fusion.hub import HubRef, download_checkpoint, parse_ref
from visin_fusion.models.registry import from_config

logger = logging.getLogger(__name__)


def load_checkpoint(
    path: str | os.PathLike, config: Mapping | str | os.PathLike | None = None, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict]:
    """The model with a checkpoint's weights in eval mode, and its ``model_info``.

    The checkpoint is read with ``weights_only=True``. Without ``model_info`` in the file, ``config`` must
    be given (a config dict or file); the model must match the weights exactly.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    info = checkpoint.get("model_info")
    if info is None:
        if config is None:
            raise ValueError(f"{path} predates model_info; pass the config that trained it: config=...")
        info = model_info(_loaded_config(config))
    for module in info.get("plugins") or []:
        importlib.import_module(module)
    model = from_config(config_from_info(info), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), info


def _loaded_config(config: Mapping | str | os.PathLike) -> Mapping:
    return prepare_config(dict(config)) if isinstance(config, Mapping) else load_config(str(config))


class Predictor:
    """A model with the preprocessing and class colors it was trained with."""

    def __init__(self, model: torch.nn.Module, info: Mapping, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.info = info
        self.device = torch.device(device)
        self.mode = info["mode"]
        self.preprocessor = Preprocessor(info["transforms"])
        self.classes = sorted(info["train_classes"], key=lambda c: c["index"])
        self.palette = np.array([c["color"] for c in self.classes], dtype=np.uint8)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike,
        config: Mapping | str | os.PathLike | None = None,
        device: str | torch.device | None = None,
    ) -> Predictor:
        """Load a checkpoint; ``device`` defaults to CUDA when available, else the CPU."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, info = load_checkpoint(path, config, device)
        return cls(model, info, device)

    @classmethod
    def from_pretrained(
        cls,
        ref: str,
        *,
        revision: str | None = None,
        filename: str | None = None,
        config: Mapping | str | os.PathLike | None = None,
        device: str | torch.device | None = None,
    ) -> Predictor:
        """Load a model from a Hugging Face Hub repo, ``hf://org/name`` or ``org/name``.

        ``revision`` (or ``@commit`` in ``ref``) pins a commit, which is how a result stays reproducible;
        ``filename`` names the checkpoint when the repo does not hold exactly ``checkpoint.pth`` or one
        ``.pth``/``.pt`` file. The checkpoint is downloaded once into the Hub cache. A private repo needs your
        own ``HF_TOKEN``; the extra is ``pip install visin-fusion[hf]``.
        """
        found = parse_ref(ref)
        path = download_checkpoint(HubRef(found.repo, revision or found.revision, filename or found.filename))
        return cls.from_checkpoint(path, config=config, device=device)

    @property
    def class_names(self) -> list[str]:
        """Class names by mask value."""
        return [c["name"] for c in self.classes]

    @torch.inference_mode()
    def logits(self, rgb: ImageInput | None = None, lidar: ImageInput | None = None) -> torch.Tensor:
        """Class scores ``[classes, size, size]`` at the model's input size."""
        rgb_tensor, lidar_tensor = self._tensors(rgb, lidar)
        return self.model(rgb_tensor, lidar_tensor)[0]

    @torch.inference_mode()
    def predict(self, rgb: ImageInput | None = None, lidar: ImageInput | None = None) -> np.ndarray:
        """The class index of every pixel as ``[H, W]`` uint8, at the size of the input image.

        ``rgb`` is needed for the ``rgb`` and fusion modes, ``lidar`` for the ``lidar`` and fusion modes.
        Values are the model's class indices (``class_names``), not the dataset's.
        """
        size = _image_size(rgb if rgb is not None else lidar)
        scores = self.logits(rgb, lidar)
        scores = F.interpolate(scores[None], size=(size[1], size[0]), mode="bilinear", align_corners=False)[0]
        return scores.argmax(dim=0).to(torch.uint8).cpu().numpy()

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        """A mask as an ``[H, W, 3]`` RGB image in the classes' colors."""
        return self.palette[mask]

    def overlay(self, rgb: ImageInput, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """The camera image with the colorized mask blended over the classes other than background."""
        image = np.asarray(_as_image(rgb).convert("RGB"), dtype=np.float32)
        color = self.colorize(mask).astype(np.float32)
        blended = np.where((mask > 0)[..., None], alpha * color + (1 - alpha) * image, image)
        return blended.round().astype(np.uint8)

    def _tensors(
        self, rgb: ImageInput | None, lidar: ImageInput | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        rgb_tensor = self.preprocessor.load_rgb(rgb)[None].to(self.device) if rgb is not None else None
        lidar_tensor = self.preprocessor.load_lidar(lidar)[None].to(self.device) if lidar is not None else None
        return rgb_tensor, lidar_tensor


def _as_image(image: ImageInput) -> Image.Image:
    return image if isinstance(image, Image.Image) else Image.open(image)


def _image_size(image: ImageInput | None) -> tuple[int, int]:
    if image is None:
        raise ValueError("give rgb or lidar")
    return _as_image(image).size


def save_prediction(predictor: Predictor, rgb: ImageInput | None, mask: np.ndarray, output: Path, stem: str) -> None:
    """Write ``<stem>_mask.png`` and, with a camera image, ``<stem>_overlay.png``.

    The mask is a palette PNG: its pixel values are the class indices (read it back with ``np.array``),
    and any image viewer shows it in the classes' colors instead of as near-black.
    """
    output.mkdir(parents=True, exist_ok=True)
    indexed = Image.frombytes("P", (mask.shape[1], mask.shape[0]), np.ascontiguousarray(mask).tobytes())
    indexed.putpalette(predictor.palette.flatten().tolist(), rawmode="RGB")
    indexed.save(output / f"{stem}_mask.png")
    if rgb is not None:
        Image.fromarray(predictor.overlay(rgb, mask)).save(output / f"{stem}_overlay.png")
