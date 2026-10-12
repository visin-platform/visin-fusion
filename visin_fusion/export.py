"""Export a trained checkpoint to ONNX, so a consumer can run it without this package, PyTorch or a GPU.

    visin-fusion export --checkpoint logs/run/checkpoints/epoch_99_<uuid>.pth --output model.onnx --frames frames.txt

The exported graph takes the same tensors the model was trained on (a camera image and, for LiDAR models, a
projection, each resized to the model's input size and normalized as ``model_info`` records) and returns the
**probability** of every class at every pixel, ``[batch, classes, size, size]``: softmax is part of the graph, so
the most probable class is the class the model predicts and its probability is a confidence for free. Resizing and
normalization are not in the graph; a consumer applies the ``transforms`` written into the file's metadata.

An export is only offered once it has been shown to agree with the weights it came from: on the frames given, the
probabilities may differ by no more than ``MAX_ABS_DIFF`` and the predicted class of at most ``MAX_MISMATCH`` of the
pixels may differ (a near-tie can flip). A failed check deletes the file.

``onnx`` and ``onnxruntime`` are optional (``pip install visin-fusion[onnx]``) and imported only here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from visin_fusion._version import __version__
from visin_fusion.data.preprocessing import ImageInput, Preprocessor
from visin_fusion.inference import load_checkpoint
from visin_fusion.stored_models import file_digest, inference_checkpoint

logger = logging.getLogger(__name__)

OPSET = 17
MAX_ABS_DIFF = 1e-4
MAX_MISMATCH = 1e-4
# A single-file ONNX is one protobuf, which cannot pass 2 GiB; larger weights need external data (several
# files), which is not a single asset a consumer can fetch.
MAX_SINGLE_FILE_BYTES = 2 * 1024**3
EXTRA_HINT = "ONNX export is an extra: pip install 'visin-fusion[onnx]'"
OUTPUT_NAME = "probabilities"

# What each mode reads. Anything else (cross_fusion) reads both.
_MODE_INPUTS: dict[str, tuple[str, ...]] = {"rgb": ("rgb",), "lidar": ("lidar",)}


class ExportError(RuntimeError):
    """The checkpoint could not be exported, or the export did not agree with it."""


def inputs_for(mode: str) -> tuple[str, ...]:
    """The named inputs of an exported model of ``mode``, in the order the graph takes them."""
    return _MODE_INPUTS.get(mode, ("rgb", "lidar"))


class Probabilities(torch.nn.Module):
    """A model that returns class probabilities, taking only the inputs its mode reads."""

    def __init__(self, model: torch.nn.Module, mode: str) -> None:
        super().__init__()
        self.model = model
        self.inputs = inputs_for(mode)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """``[batch, classes, H, W]`` probabilities for the tensors of ``self.inputs``, in that order."""
        given = dict(zip(self.inputs, tensors, strict=True))
        return self.model(given.get("rgb"), given.get("lidar")).softmax(dim=1)


def _runtime() -> tuple[Any, Any]:
    try:
        import onnx  # noqa: PLC0415 - optional extra
        import onnxruntime  # noqa: PLC0415 - optional extra
    except ImportError as exc:
        raise ExportError(EXTRA_HINT) from exc
    return onnx, onnxruntime


def _frame_tensors(
    preprocessor: Preprocessor, mode: str, frame: tuple[ImageInput, ImageInput | None]
) -> list[torch.Tensor]:
    camera, lidar = frame
    tensors: dict[str, torch.Tensor] = {}
    if "rgb" in inputs_for(mode):
        tensors["rgb"] = preprocessor.load_rgb(camera)[None]
    if "lidar" in inputs_for(mode):
        if lidar is None:
            raise ExportError(f"{camera} has no LiDAR projection, which this {mode} model reads")
        tensors["lidar"] = preprocessor.load_lidar(lidar)[None]
    return [tensors[name] for name in inputs_for(mode)]


def export_onnx(
    checkpoint: str | Path,
    output: str | Path,
    *,
    frames: Sequence[tuple[ImageInput, ImageInput | None]] = (),
    size: int | None = None,
    opset: int = OPSET,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Write ``checkpoint`` as ONNX to ``output`` and prove it agrees with the weights.

    ``frames`` are ``(camera, lidar)`` pairs the two are compared on (``lidar`` is ``None`` for a camera-only
    model); with none, they are compared on seeded random inputs, which says the graph was exported faithfully
    but not that it is right on real frames, and ``parity["frames"]`` is ``0``. ``size`` defaults to the
    model's own input size.

    Returns ``path``, ``size`` and ``sha256`` of the file, the ``inputs`` names, ``opset``, ``model_info``, and
    ``parity`` (``maxAbsDiff``, ``argmaxMismatchFraction``, ``frames``). Raises :class:`ExportError`.
    """
    onnx, onnxruntime = _runtime()
    output = Path(output)
    checkpoint = inference_checkpoint(checkpoint)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights_digest = file_digest(checkpoint)
    if source_sha256 and source_sha256 != weights_digest:
        raise ExportError("source_sha256 does not match the inference checkpoint")
    model, info = load_checkpoint(checkpoint, device="cpu")
    if sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) >= MAX_SINGLE_FILE_BYTES:
        raise ExportError("the weights need ONNX external data (over 2 GiB); single-file fp32 export is unsupported")
    mode = info["mode"]
    side = int(size if size is not None else info["transforms"]["resize"])
    if side < 1:
        raise ExportError("the model input size must be positive")
    if Path(output).resolve() == Path(checkpoint).resolve():
        raise ExportError("the export output must differ from its checkpoint")
    info = {**info, "transforms": {**info["transforms"], "resize": side}}
    wrapper = Probabilities(model, mode).eval()
    inputs = wrapper.inputs
    example = tuple(torch.zeros(1, 3, side, side) for _ in inputs)

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                example,
                str(output),
                input_names=list(inputs),
                output_names=[OUTPUT_NAME],
                opset_version=opset,
                dynamic_axes={name: {0: "batch"} for name in (*inputs, OUTPUT_NAME)},
                dynamo=False,
                external_data=False,
            )
    except Exception as exc:  # the exporter raises whatever the traced model does
        output.unlink(missing_ok=True)
        raise ExportError(f"{info.get('backbone', 'this model')} could not be exported to ONNX: {exc}") from exc

    if output.stat().st_size > MAX_SINGLE_FILE_BYTES:
        output.unlink()
        raise ExportError("the exported model is over 2 GiB, which a single ONNX file cannot hold")

    graph = onnx.load(str(output))
    for key, value in {
        "visin_fusion_version": __version__,
        "visin_fusion_model_info": json.dumps(info, default=str),
        "visin_fusion_inputs": ",".join(inputs),
        "visin_fusion_output": f"{OUTPUT_NAME}: softmax over classes, [batch, classes, {side}, {side}]",
        "visin_fusion_source_sha256": weights_digest,
    }.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(graph, str(output))

    try:
        parity = _parity(onnxruntime, wrapper, output, info, frames, side)
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise ExportError(f"ONNX parity could not be checked: {exc}") from exc
    if (
        not np.isfinite(parity["maxAbsDiff"])
        or parity["maxAbsDiff"] >= MAX_ABS_DIFF
        or parity["argmaxMismatchFraction"] > MAX_MISMATCH
    ):
        output.unlink()
        raise ExportError(
            f"the ONNX model disagrees with the weights: probabilities differ by up to {parity['maxAbsDiff']:.2e} "
            f"(limit {MAX_ABS_DIFF:.0e}) and the class of {parity['argmaxMismatchFraction']:.2%} of pixels "
            f"(limit {MAX_MISMATCH:.2%}); not written"
        )
    logger.info("Exported %s: %d bytes, parity %s", output, output.stat().st_size, parity)
    return {
        "path": str(output),
        "epoch": checkpoint_state.get("epoch"),
        "sourceSha256": weights_digest,
        "size": output.stat().st_size,
        "sha256": file_digest(output),
        "inputs": list(inputs),
        "opset": opset,
        "model_info": info,
        "parity": parity,
    }


def _parity(
    onnxruntime: Any,
    wrapper: Probabilities,
    path: Path,
    info: Mapping[str, Any],
    frames: Sequence[tuple[ImageInput, ImageInput | None]],
    side: int,
) -> dict[str, Any]:
    """How far the exported graph is from the weights, over ``frames``: worst probability gap and class mismatch."""
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batches: list[list[torch.Tensor]]
    if frames:
        preprocessor = Preprocessor(info["transforms"])
        batches = [_frame_tensors(preprocessor, info["mode"], frame) for frame in frames]
    else:
        generator = torch.Generator().manual_seed(0)
        batches = [[torch.randn(1, 3, side, side, generator=generator) for _ in wrapper.inputs] for _ in range(2)]

    worst, mismatched, pixels = 0.0, 0, 0
    for tensors in batches:
        with torch.inference_mode():
            expected = wrapper(*tensors).numpy()
        got = session.run([OUTPUT_NAME], {name: t.numpy() for name, t in zip(wrapper.inputs, tensors, strict=True)})[0]
        if not np.isfinite(expected).all() or not np.isfinite(got).all():
            raise ExportError("parity produced nonfinite probabilities")
        worst = max(worst, float(np.abs(expected - got).max()))
        differing = expected.argmax(axis=1) != got.argmax(axis=1)
        mismatched += int(differing.sum())
        pixels += differing.size
    return {
        "maxAbsDiff": worst,
        "argmaxMismatchFraction": mismatched / pixels if pixels else 0.0,
        "frames": len(frames),
    }
