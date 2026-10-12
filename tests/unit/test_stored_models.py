"""Published weights retain one identity across checkpoint rotation, SDK publication and verified downloads."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from visin_fusion import stored_models
from visin_fusion.inference import Predictor


@pytest.fixture
def checkpoint(tmp_path):
    folder = tmp_path / "logs" / "checkpoints"
    folder.mkdir(parents=True)
    path = folder / "epoch_7_abc.pth"
    torch.save(
        {
            "epoch": 7,
            "model_state_dict": {"weight": torch.ones(2, 3)},
            "model_info": {"backbone": "clft", "mode": "rgb"},
            "optimizer_state_dict": {"state": {1: torch.zeros(10)}},
        },
        path,
    )
    return path


def test_inference_copy_is_stable_and_survives_rotation(checkpoint):
    target = stored_models.inference_checkpoint(checkpoint)
    digest = stored_models.file_digest(target)
    assert set(torch.load(target, weights_only=True)) == {"epoch", "model_state_dict", "model_info"}
    assert stored_models.inference_checkpoint(checkpoint) == target
    assert stored_models.file_digest(target) == digest
    checkpoint.unlink()
    assert stored_models.inference_checkpoint(target) == target
    assert torch.load(target, weights_only=True)["epoch"] == 7


def test_changed_weights_or_corrupt_copy_rebuild_the_inference_file(checkpoint):
    target = stored_models.inference_checkpoint(checkpoint)
    first = stored_models.file_digest(target)
    state = torch.load(checkpoint, weights_only=True)
    state["model_state_dict"]["weight"] += 1
    torch.save(state, checkpoint)
    assert stored_models.inference_checkpoint(checkpoint) == target
    assert stored_models.file_digest(target) != first
    target.write_bytes(b"broken")
    stored_models.inference_checkpoint(checkpoint)
    assert torch.equal(torch.load(target, weights_only=True)["model_state_dict"]["weight"], torch.full((2, 3), 2.0))
    target.with_name("publication.json").write_text("[")
    stored_models.inference_checkpoint(checkpoint)
    assert json.loads(target.with_name("publication.json").read_text())["epoch"] == 7


def test_training_publishes_the_best_inference_copy_through_the_enabled_run(checkpoint, monkeypatch):
    run = SimpleNamespace(
        enabled=True, save_model=Mock(side_effect=lambda path, **kwargs: stored_models.file_digest(path))
    )
    monkeypatch.setitem(sys.modules, "visin", SimpleNamespace(get_run=lambda: run))
    monkeypatch.setattr(stored_models, "get_best_checkpoint_path", lambda config: checkpoint)
    config = {"General": {"publish_model": "visin", "model_license": "cc-by-4.0", "model_credit": "TalTech"}}
    digest = stored_models.publish_best_checkpoint(config)
    saved = run.save_model.call_args
    assert digest == stored_models.file_digest(saved.args[0])
    assert saved.kwargs == {
        "epoch": 7,
        "licence": "cc-by-4.0",
        "credit": "TalTech",
        "model_info": {"backbone": "clft", "mode": "rgb"},
    }
    assert "optimizer_state_dict" not in torch.load(saved.args[0], weights_only=True)


def test_publication_does_nothing_without_configuration_and_logs_refusals(checkpoint, monkeypatch, caplog):
    assert stored_models.publish_best_checkpoint({"General": {}}) is None
    config = {"General": {"publish_model": "visin"}}
    monkeypatch.setitem(sys.modules, "visin", SimpleNamespace(get_run=lambda: SimpleNamespace(enabled=False)))
    assert stored_models.publish_best_checkpoint(config) is None
    assert "model_license is unset" in caplog.text
    assert "No enabled Visin run" in caplog.text
    run = SimpleNamespace(enabled=True, save_model=Mock(side_effect=OSError("offline")))
    monkeypatch.setitem(sys.modules, "visin", SimpleNamespace(get_run=lambda: run))
    monkeypatch.setattr(stored_models, "get_best_checkpoint_path", lambda config: None)
    assert stored_models.publish_best_checkpoint(config) is None
    monkeypatch.setattr(stored_models, "get_best_checkpoint_path", lambda config: checkpoint)
    assert stored_models.publish_best_checkpoint(config) is None
    assert "Publishing the best checkpoint to Visin failed" in caplog.text


def test_visin_predictor_verifies_download_before_loading_and_removes_temporary_bytes(tmp_path, monkeypatch):
    seen = []

    def download(ref, folder):
        path = tmp_path / "downloaded.pth"
        seen.append((ref, folder))
        return path

    monkeypatch.setattr(stored_models, "download_checkpoint", download)
    monkeypatch.setattr(Predictor, "from_checkpoint", lambda path, **kwargs: (path, kwargs))
    result = Predictor.from_pretrained("visin://run/model", device="cpu")
    assert result[0] == tmp_path / "downloaded.pth"
    assert seen[0][0] == "visin://run/model"
    assert not __import__("pathlib").Path(seen[0][1]).exists()
    with pytest.raises(ValueError, match="pinned by model id"):
        Predictor.from_pretrained("visin://run/model", revision="main")


@pytest.mark.parametrize("ref", ["visin://run", "visin:///model", "visin://run/model/extra", "hf://run/model"])
def test_visin_reference_requires_run_and_model_ids(ref, tmp_path):
    with pytest.raises(ValueError, match="Expected visin://"):
        stored_models.download_checkpoint(ref, tmp_path)


def test_download_uses_sdk_model_selection_and_digest_verification(tmp_path, monkeypatch):
    api = Mock()
    api.__enter__ = Mock(return_value=api)
    api.__exit__ = Mock(return_value=False)
    api.download_model.return_value = tmp_path / "model.pth"
    module = SimpleNamespace(Api=Mock(return_value=api))
    monkeypatch.setitem(sys.modules, "visin", module)
    assert stored_models.download_checkpoint("visin://run/model", tmp_path) == tmp_path / "model.pth"
    api.download_model.assert_called_once_with("run", model="model", directory=tmp_path, file_format="pth")
    monkeypatch.setitem(sys.modules, "visin", None)
    with pytest.raises(ImportError, match="extra"):
        stored_models.download_checkpoint("visin://run/model", tmp_path)


def test_onnx_publication_keeps_source_digest_parity_and_run_status(monkeypatch):
    calls = []

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def save_model(self, path, **kwargs):
            calls.append((path, kwargs))
            return "export-digest"

        @staticmethod
        def attach(uuid, **kwargs):
            assert uuid == "run" and kwargs == {"mark_status": False}
            return Run()

    monkeypatch.setitem(sys.modules, "visin", SimpleNamespace(Run=Run))
    result = {
        "path": "model.onnx",
        "epoch": 7,
        "model_info": {"mode": "rgb"},
        "sourceSha256": "a" * 64,
        "parity": {"frames": 48, "maxAbsDiff": 1e-6, "argmaxMismatchFraction": 0.0},
    }
    assert stored_models.publish_export(result, "run", licence="cc-by-4.0", credit="TalTech") == "export-digest"
    assert calls[0][1]["source_sha256"] == result["sourceSha256"]
    assert calls[0][1]["parity"] == result["parity"]
    result["parity"]["frames"] = 0
    with pytest.raises(ValueError, match="actual sample frames"):
        stored_models.publish_export(result, "run")
