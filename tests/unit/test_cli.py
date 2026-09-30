"""The installed command validates once and gives each stage a resolved config."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from visin_fusion import cli

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "zod_sample"


def config_file(tmp_path, **overrides):
    config = {"extends": "clftv2", "Dataset": {"dataset_root": str(SAMPLE)}, "Log": {"logdir": "run-logs"}}
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_run_resolves_config_and_orders_selected_stages(tmp_path, monkeypatch):
    path = config_file(tmp_path)
    calls = []

    def fake_run(command):
        calls.append((command, json.loads(Path(command[command.index("-c") + 1]).read_text())))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli.run(
        [
            "-c",
            str(path),
            "--stages",
            "test, visualize,benchmark",
            "--upload",
            "--benchmark-device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert [cmd[2] for cmd, _ in calls] == [
        "visin_fusion.engine.stages.test.common",
        "visin_fusion.engine.stages.visualize.common",
        "visin_fusion.engine.stages.benchmark.common",
    ]
    assert "--upload" in calls[1][0]
    assert calls[2][0][-3:] == ["--single", "--device", "cpu"]
    assert all(config["Log"]["logdir"] == str(tmp_path / "run-logs") for _, config in calls)
    assert not Path(calls[0][0][4]).exists()  # resolved temporary config was removed


def test_run_stops_at_failed_stage_and_removes_config(tmp_path, monkeypatch):
    path = config_file(tmp_path, Log={"logdir": str(tmp_path / "absolute")})
    commands = []

    def fail(command):
        commands.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, "run", fail)
    with pytest.raises(SystemExit) as error:
        cli.run(["-c", str(path), "--stages", "train,test"])
    assert error.value.code == 7
    assert len(commands) == 1
    assert not Path(commands[0][4]).exists()


@pytest.mark.parametrize("stages", ["", "not-a-stage"])
def test_run_rejects_bad_stage_selection(tmp_path, stages):
    with pytest.raises(SystemExit) as error:
        cli.run(["-c", str(config_file(tmp_path)), "--stages", stages])
    assert error.value.code == 2


def test_run_reports_invalid_config(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{}")
    with pytest.raises(SystemExit, match="Invalid config"):
        cli.run(["-c", str(path), "--stages", "test"])


def test_main_schema_help_and_usage(capsys):
    cli.main(["schema"])
    schema = json.loads(capsys.readouterr().out)
    assert schema["x-schema-version"] == "2.0"
    cli.main(["--help"])
    assert "visin-fusion run" in capsys.readouterr().out
    cli.main([])
    assert "visin-fusion schema" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="usage: visin-fusion schema"):
        cli.main(["schema", "extra"])
    with pytest.raises(SystemExit, match="usage: visin-fusion run"):
        cli.main(["unknown"])
