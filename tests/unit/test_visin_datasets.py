"""`visin:` dataset roots, downloaded through the visin package from a local stand-in for Visin's
dataset service."""

import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from visin_fusion.config.config import prepare_config
from visin_fusion.data import visin_datasets
from visin_fusion.sample import SAMPLE_DIR

SAMPLE = SAMPLE_DIR


def sample_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in SAMPLE.rglob("*"):
            if path.is_file():
                archive.write(path, Path("zod_sample") / path.relative_to(SAMPLE))
    return buffer.getvalue()


@pytest.fixture
def visin(tmp_path, monkeypatch):
    """A Visin dataset API serving the sample dataset as 'ZOD'; counts the zip downloads."""
    archive = sample_zip()
    state = {"downloads": 0, "size": len(archive)}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, content_type="application/json", headers=()):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            base = f"http://127.0.0.1:{self.server.server_port}"
            if self.path.startswith("/api/datasets?"):
                data = [{"id": "abc123", "name": "ZOD", "archive": {"size": state["size"]}}]
                self.send(200, json.dumps({"success": True, "data": data}).encode())
            elif self.path == "/api/datasets/abc123/download":
                data = {"downloadUrl": f"{base}/files/zod.zip", "filename": "zod_dataset.zip"}
                self.send(200, json.dumps({"success": True, "data": data}).encode())
            elif self.path == "/files/zod.zip":
                state["downloads"] += 1
                start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0)
                if start:
                    self.send(
                        206,
                        archive[start:],
                        "application/zip",
                        [("Content-Range", f"bytes {start}-{len(archive) - 1}/{len(archive)}")],
                    )
                else:
                    self.send(200, archive, "application/zip")
            else:
                self.send(404, b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("VISIN_DATASET_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("VISIN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("VISIN_TOKEN", raising=False)
    yield state, archive, tmp_path / "data"
    server.shutdown()


def test_a_config_can_name_a_visin_dataset(visin):
    state, _, data = visin
    config = prepare_config({"extends": "clftv2", "Dataset": {"dataset_root": "visin:zod"}})
    root = Path(config["Dataset"]["dataset_root"])
    assert root.is_relative_to(data) and (root / "dataset.json").exists()
    assert config["Dataset"]["name"] == "zod"  # from the downloaded dataset's manifest
    assert Path(config["Dataset"]["train_split"]).exists()
    assert state["downloads"] == 1


def test_a_downloaded_dataset_is_used_again_without_downloading(visin):
    state, _, _ = visin
    first = visin_datasets.resolve_root("visin:zod")
    assert visin_datasets.resolve_root("visin:ZOD") == first
    assert state["downloads"] == 1


def test_this_projects_dataset_service_is_the_default(monkeypatch):
    used = {}

    class Recording:
        def __init__(self, url=None, **_kwargs):
            used["url"] = url

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def download(self, ref):
            return Path("/data") / ref

    import visin

    monkeypatch.delenv("VISIN_DATASET_URL", raising=False)
    monkeypatch.setattr(visin, "Datasets", Recording)
    assert visin_datasets.resolve_root("visin:zod") == "/data/zod"
    assert used["url"] == visin_datasets.DEFAULT_DATASET_URL


def test_other_roots_are_left_alone(tmp_path):
    assert visin_datasets.resolve_root(str(tmp_path)) == str(tmp_path)
    assert visin_datasets.resolve_root(None) is None
