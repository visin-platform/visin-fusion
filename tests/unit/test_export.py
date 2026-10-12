"""ONNX exports contain probabilities and metadata, and pass an independent Runtime parity check."""

import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from visin_fusion import cli, export
from visin_fusion.stored_models import file_digest


class TinySegmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 3, 1)

    def forward(self, rgb, lidar):
        if rgb is not None and lidar is not None:
            return self.layer(rgb + lidar)
        return self.layer(rgb if rgb is not None else lidar)


@pytest.fixture
def model(tmp_path, monkeypatch):
    net = TinySegmentation().eval()
    info = {
        "mode": "rgb",
        "backbone": "test",
        "transforms": {"resize": 8, "image_mean": [0.5] * 3, "image_std": [0.2] * 3},
    }
    path = tmp_path / "weights.pth"
    torch.save({"epoch": 7, "model_info": info, "model_state_dict": net.state_dict()}, path)
    monkeypatch.setattr(export, "load_checkpoint", lambda checkpoint, device: (net, info))
    return net, info, path


@pytest.mark.parametrize(
    ("mode", "names"), [("rgb", ["rgb"]), ("lidar", ["lidar"]), ("cross_fusion", ["rgb", "lidar"])]
)
def test_real_onnx_runtime_matches_pytorch_and_graph_records_preprocessing(model, tmp_path, mode, names):
    onnx = pytest.importorskip("onnx")
    runtime = pytest.importorskip("onnxruntime")
    _net, info, path = model
    info["mode"] = mode
    camera = Image.fromarray(np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3))
    result = export.export_onnx(path, tmp_path / "model.onnx", frames=[(camera, camera)])
    assert result["inputs"] == names and result["epoch"] == 7
    assert result["sourceSha256"] == file_digest(path)
    assert result["parity"]["frames"] == 1
    assert result["parity"]["maxAbsDiff"] < 1e-4
    assert result["parity"]["argmaxMismatchFraction"] <= 1e-4
    graph = onnx.load(result["path"])
    onnx.checker.check_model(graph)
    assert any(node.op_type == "Softmax" for node in graph.graph.node)
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    assert json.loads(metadata["visin_fusion_model_info"])["transforms"]["resize"] == 8
    assert metadata["visin_fusion_source_sha256"] == file_digest(path)
    session = runtime.InferenceSession(result["path"], providers=["CPUExecutionProvider"])
    probabilities = session.run(None, {name: np.zeros((1, 3, 8, 8), dtype=np.float32) for name in names})[0]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


def test_failed_or_nonfinite_parity_deletes_the_export(model, tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    for diff in (1e-3, float("nan"), 1e-4):
        monkeypatch.setattr(
            export, "_parity", lambda *args, gap=diff: {"maxAbsDiff": gap, "argmaxMismatchFraction": 0, "frames": 1}
        )
        target = tmp_path / "bad.onnx"
        with pytest.raises(export.ExportError, match="disagrees"):
            export.export_onnx(model[2], target)
        assert not target.exists()


def test_missing_export_extra_has_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(export.ExportError, match=r"visin-fusion\[onnx\]"):
        export._runtime()


def test_cli_resolves_frame_list_and_requires_real_frames_for_publication(model, tmp_path, capsys):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    camera = tmp_path / "camera"
    camera.mkdir()
    Image.new("RGB", (16, 12), "red").save(camera / "frame.png")
    frames = tmp_path / "frames.txt"
    frames.write_text("camera/frame.png\n")
    cli.main(["export", "--checkpoint", str(model[2]), "--frames", str(frames), "--output", str(tmp_path / "out.onnx")])
    assert json.loads(capsys.readouterr().out)["parity"]["frames"] == 1
    with pytest.raises(SystemExit):
        cli.main(["export", "--checkpoint", str(model[2]), "--publish", "--run", "run"])
    with pytest.raises(SystemExit):
        cli.main(["export", "--checkpoint", str(model[2]), "--size", "0"])


def test_runtime_nonfinite_probabilities_are_never_accepted_as_zero_difference(model, tmp_path):
    from types import SimpleNamespace

    session = SimpleNamespace(run=lambda *args: [np.full((1, 3, 8, 8), np.nan, dtype=np.float32)])
    runtime = SimpleNamespace(InferenceSession=lambda *args, **kwargs: session)
    wrapper = export.Probabilities(model[0], "rgb")
    with pytest.raises(export.ExportError, match="nonfinite probabilities"):
        export._parity(runtime, wrapper, tmp_path / "unused.onnx", model[1], [], 8)
