"""Visualization uploads include only rendered files and carry frame metadata."""

import json
from unittest.mock import Mock

import pytest
from PIL import Image

from visin_fusion.integrations.visualization_uploader import prediction_dataset, queue_visualizations


def test_queue_visualizations_uploads_existing_kinds(tmp_path, caplog):
    for kind in ("segment", "overlay"):
        directory = tmp_path / kind
        directory.mkdir()
        Image.new("RGB", (16, 12)).save(directory / "frame.png")
    run = Mock()
    assert queue_visualizations(run, 4, "epoch-id", str(tmp_path), "frame.png") == 2
    assert run.upload_visualization.call_count == 2
    assert run.upload_visualization.call_args_list[0].args == (4, str(tmp_path / "segment" / "frame.png"), "segment")
    assert run.upload_visualization.call_args_list[1].kwargs == {
        "epoch_uuid": "epoch-id",
        "metadata": {"image_name": "frame.png", "source": "visualization_script"},
    }
    assert "Skipping compare" in caplog.text


def test_prediction_outputs_report_dataset_frame_geometry_and_confidence_encoding(tmp_path):
    folder = tmp_path / "confidence"
    folder.mkdir()
    Image.new("L", (1363, 768), 200).save(folder / "sq11_000301.png")
    run = Mock()
    dataset = {"source": "visin", "id": "sample", "archiveRevision": "sample-v1"}
    queue_visualizations(run, 99, "epoch", str(tmp_path), "sq11_000301.png", predicted_on=dataset)
    assert run.upload_visualization.call_args.kwargs["metadata"]["prediction"] == {
        "dataset": dataset,
        "frame": "sq11_000301",
        "width": 1363,
        "height": 768,
        "confidenceEncoding": "uint8",
    }


def test_frame_list_dataset_is_read_from_its_download_marker(tmp_path):
    (tmp_path / ".visin-dataset.json").write_text(
        json.dumps({"id": "sample", "revision": "rev", "archiveRevision": "archive"})
    )
    lists = tmp_path / "splits"
    lists.mkdir()
    assert prediction_dataset(lists / "sample.txt") == {
        "source": "visin",
        "id": "sample",
        "revision": "rev",
        "archiveRevision": "archive",
    }


@pytest.mark.parametrize("contents", ["{", "[]", "{}", '{"id": 1}'])
def test_missing_or_invalid_prediction_marker_does_not_invent_training_provenance(tmp_path, contents):
    assert prediction_dataset(tmp_path / "frames.txt") is None
    (tmp_path / ".visin-dataset.json").write_text(contents)
    assert prediction_dataset(tmp_path / "frames.txt") is None
