"""Visin credentials belong to the calling application, not the installed package."""

import os
import sys

import pytest

from visin_fusion.integrations import settings


def test_external_env_file_is_used_from_any_working_directory(tmp_path, monkeypatch):
    credentials = tmp_path / "outside" / "visin.env"
    credentials.parent.mkdir()
    credentials.write_text("VISIN_TOKEN=from-file\nVISIN_URL=https://example.invalid\n")
    app = tmp_path / "application"
    app.mkdir()
    (app / ".env").write_text("VISIN_TOKEN=wrong-file\n")
    monkeypatch.chdir(app)
    monkeypatch.setenv("VISIN_ENV_FILE", str(credentials))
    monkeypatch.delenv("VISIN_TOKEN", raising=False)
    monkeypatch.delenv("VISIN_URL", raising=False)

    assert settings.env_file() == credentials
    assert settings.pipeline_key_present()
    settings.load_environment()
    assert os.environ["VISIN_TOKEN"] == "from-file"
    assert os.environ["VISIN_URL"] == "https://example.invalid"


def test_exported_variables_win_over_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("VISIN_TOKEN=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VISIN_ENV_FILE", raising=False)
    monkeypatch.setenv("VISIN_TOKEN", "from-shell")
    assert settings.env_file() == tmp_path / ".env"
    assert settings.pipeline_key_present()
    settings.load_environment()
    assert os.environ["VISIN_TOKEN"] == "from-shell"


def test_no_package_or_legacy_env_file_is_searched(tmp_path, monkeypatch):
    legacy = tmp_path / "integrations"
    legacy.mkdir()
    (legacy / ".env").write_text("VISIN_TOKEN=legacy-secret\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VISIN_ENV_FILE", raising=False)
    monkeypatch.delenv("VISIN_TOKEN", raising=False)
    assert settings.env_file() is None
    assert settings.pipeline_key_present() is False


def test_explicit_missing_file_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIN_ENV_FILE", str(tmp_path / "missing.env"))
    with pytest.raises(FileNotFoundError, match="VISIN_ENV_FILE"):
        settings.pipeline_key_present()


def test_token_detection_without_optional_dotenv(tmp_path, monkeypatch):
    path = tmp_path / "visin.env"
    path.write_text('export VISIN_TOKEN="file-key"\n')
    monkeypatch.setenv("VISIN_ENV_FILE", str(path))
    monkeypatch.delenv("VISIN_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", None)
    assert settings.pipeline_key_present()
    path.write_text("VISIN_TOKEN=\n")
    assert not settings.pipeline_key_present()
