"""The installed command validates once and gives each stage a resolved config."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from visin_fusion import cli, pipeline
from visin_fusion.sample import SAMPLE_DIR

SAMPLE = SAMPLE_DIR


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

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
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

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
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
    with pytest.raises(SystemExit, match="usage: visin-fusion quickstart"):
        cli.main(["unknown"])


def test_the_pipeline_runs_from_python_with_a_config_dict(tmp_path, monkeypatch):
    import visin_fusion

    commands = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda c: commands.append(c) or SimpleNamespace(returncode=0))
    config = {"extends": "clftv2", "Dataset": {"dataset_root": str(SAMPLE)}, "Log": {"logdir": "mine"}}
    logdir = visin_fusion.run(config, ["train", "test"], output_dir=tmp_path)
    assert logdir == str(tmp_path / "mine")
    assert [c[2].split(".")[-2] for c in commands] == ["train", "test"]


def test_a_failed_stage_raises_with_its_name_and_code(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, "run", lambda c: SimpleNamespace(returncode=3))
    with pytest.raises(pipeline.StageFailed) as error:
        pipeline.run(config_file(tmp_path), ["test"])
    assert (error.value.stage, error.value.returncode) == ("test", 3)


def test_unknown_stages_are_refused_before_anything_runs(tmp_path):
    with pytest.raises(ValueError, match="stages must be"):
        pipeline.run(config_file(tmp_path), ["train", "deploy"])


@pytest.mark.parametrize(("configured", "frames", "expected"), [(None, 4, 2), (None, 1, 1), (3, 4, 3), (0, 100, 0)])
def test_default_data_loader_workers_never_exceed_the_batches(monkeypatch, configured, frames, expected):
    from visin_fusion.utils import helpers

    monkeypatch.setattr(helpers.os, "cpu_count", lambda: 64)
    general = {"batch_size": 2, **({"num_workers": configured} if configured is not None else {})}
    assert helpers.num_workers({"General": general}, frames) == expected


def test_a_config_dict_may_hold_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, "run", lambda c: SimpleNamespace(returncode=0))
    config = {"extends": "clftv2", "Dataset": {"dataset_root": SAMPLE}, "Log": {"logdir": tmp_path / "logs"}}
    assert pipeline.run(config, ["test"]) == str(tmp_path / "logs")
