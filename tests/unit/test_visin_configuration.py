"""Reporting requires caller-owned addresses and preserves downloaded dataset provenance."""

import json

import pytest

from visin_fusion.integrations import visin as integration


def test_token_without_address_is_refused_at_stage_start(monkeypatch):
    monkeypatch.setattr(integration, "_environment_ready", False)
    monkeypatch.setattr(integration, "load_environment", lambda: None)
    monkeypatch.setenv("VISIN_TOKEN", "pipeline-key")
    monkeypatch.delenv("VISIN_URL", raising=False)
    monkeypatch.delenv("VISIN_API_URL", raising=False)
    with pytest.raises(ValueError, match="VISIN_TOKEN requires VISIN_URL"):
        integration.prepare_environment()


def test_dataset_reference_comes_from_the_download_marker(tmp_path):
    (tmp_path / ".visin-dataset.json").write_text(
        json.dumps({"id": "dataset-id", "name": "ZOD", "revision": "version"})
    )
    root = tmp_path / "zod"
    root.mkdir()
    assert integration._dataset_reference(str(root), "zod") == {
        "source": "visin",
        "id": "dataset-id",
        "name": "ZOD",
        "revision": "version",
    }
    assert integration._dataset_reference(None, "local") == "local"


@pytest.mark.parametrize(
    "contents", ["{", "[]", "{}", '{"id": 42, "name": "ZOD"}', '{"id": "dataset-id", "name": "ZOD", "revision": []}']
)
def test_bad_dataset_provenance_does_not_abort_training(tmp_path, caplog, contents):
    (tmp_path / ".visin-dataset.json").write_text(contents)
    assert integration._dataset_reference(str(tmp_path), "local") == "local"
    assert "dataset provenance" in caplog.text


COMMIT = "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_a_hub_dataset_is_recorded_with_its_repo_and_the_commit_it_holds(separator):
    root = separator.join(["", "cache", "hub", "datasets--acme--zod-png", "snapshots", COMMIT])
    assert integration._dataset_reference(root, "zod") == {"source": "hf", "name": "acme/zod-png", "revision": COMMIT}


def test_a_cache_folder_that_names_no_commit_is_only_a_label(tmp_path):
    root = tmp_path / "datasets--acme--zod-png" / "snapshots" / "main"
    root.mkdir(parents=True)
    assert integration._dataset_reference(str(root), "zod") == "zod"
