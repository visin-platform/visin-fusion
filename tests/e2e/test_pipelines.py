"""End-to-end: every model trains, tests, visualizes and benchmarks on the sample dataset.

Each case runs the four stage scripts as subprocesses on visin_fusion/sample/zod_sample for one epoch,
with all outputs in a temporary log directory. The config is written the way a user writes one:
it extends the model's preset (visin_fusion/config/presets/), names the dataset only by its root (the rest
comes from visin_fusion/sample/zod_sample/dataset.json), and overrides a few training settings.

    pytest tests/e2e                              # all models and modes, Visin offline
    pytest tests/e2e --models clftv2 --modes fusion # one model, one mode
    pytest tests/e2e --visin online               # report to the Visin project of VISIN_TOKEN
"""

import glob
import json
import os
import subprocess
from pathlib import Path

import pytest

from visin_fusion.config.config_schema import BACKBONES
from visin_fusion.pipeline import stage_command
from visin_fusion.sample import SAMPLE_DIR

REPO = Path(__file__).resolve().parents[2]
SAMPLE = SAMPLE_DIR
WEATHER_CONDITIONS = ["day_fair", "day_rain", "night_fair", "night_rain", "snow"]
VISUALIZATION_KINDS = ["segment", "overlay", "compare", "correct_only"]
STAGE_TIMEOUT = 30 * 60  # seconds; CPU runs of the larger models are slow


# Test name -> the model's preset in visin_fusion/config/presets/ (the stages are the same for every model)
PRESETS = {
    "clft": "clft",
    "clftv2": "clftv2",
    "maskformer": "maskformer",
    "mask2former": "mask2former",
    "deeplab": "deeplabv3plus",
}
MODELS = PRESETS  # the names --models accepts
# A few timing runs are enough to check the benchmark works
BENCHMARK_ARGS = ["--num-runs", "5", "--warmup-runs", "1"]


def preset(model):
    with open(REPO / "visin_fusion" / "config" / "presets" / f"{PRESETS[model]}.json") as f:
        return json.load(f)


def command(model, stage, config_path, **options):
    extra = BENCHMARK_ARGS if stage == "benchmark" else []
    return stage_command(stage, str(config_path), **options) + extra


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:

        def option(name):
            return [value.strip() for value in metafunc.config.getoption(name).split(",") if value.strip()]

        models, modes = option("--models"), option("--modes")
        unknown = set(models) - set(MODELS)
        if unknown:
            raise pytest.UsageError(f"--models: unknown {sorted(unknown)}; choose from {sorted(MODELS)}")
        cases = [(model, mode) for model in models for mode in modes]
        metafunc.parametrize("case", cases, ids=[f"{model}-{mode}" for model, mode in cases])


def read_lines(path):
    with open(path) as f:
        return [line for line in f.read().splitlines() if line.strip()]


def make_config(model, mode, logdir, device):
    """A user's config: the model's preset, the sample dataset, one short epoch."""
    backbone = preset(model)["CLI"]["backbone"]
    section = BACKBONES[backbone][0]
    config = {
        "extends": PRESETS[model],
        "Summary": f"e2e: {model} {mode}",
        "tags": ["e2e", model, mode],
        "Dataset": {"dataset_root": str(SAMPLE)},
        "Log": {"logdir": str(logdir)},
        "General": {
            "device": device,
            "epochs": 1,
            "batch_size": 2,
            "early_stop_patience": 1,
            "max_checkpoints": 1,
            "num_workers": 0,  # worker processes only add start-up time on a handful of frames
        },
        # One epoch, no warmup (schedules follow General.epochs)
        section: {"warmup_epochs": 0},
    }
    if mode != "fusion":  # the presets' default mode
        config["CLI"] = {"mode": mode}
    return config


def visin_env(mode, spool_dir):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if mode == "disabled":
        env["VISIN_MODE"] = "disabled"
    elif mode == "offline":
        # Offline needs a server configured to be on at all; nothing is sent to it.
        env.update(VISIN_MODE="offline", VISIN_DIR=str(spool_dir))
        env.setdefault("VISIN_TOKEN", "e2e-offline")
        env.setdefault("VISIN_URL", "http://127.0.0.1:9")
    else:
        env["VISIN_MODE"] = "online"  # token and URL from exported variables or a caller-owned env file
    return env


def run_stage(name, cmd, env, logdir):
    """Run one stage's command from the repo root; fail with the end of its output."""
    log_path = Path(logdir) / f"{name}.log"
    with open(log_path, "w") as log:
        result = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=STAGE_TIMEOUT,
        )
    output = log_path.read_text()
    if result.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        pytest.fail(f"{name} exited with {result.returncode} (full log: {log_path})\n{tail}")
    return output


def only(pattern):
    matches = glob.glob(str(pattern))
    assert len(matches) == 1, f"expected one {pattern}, found {matches}"
    return matches[0]


@pytest.mark.e2e
def test_pipeline(case, tmp_path, device, visin_mode):
    model, mode = case
    logdir = tmp_path / "logs"
    logdir.mkdir()
    spool_dir = tmp_path / "visin"
    config = make_config(model, mode, logdir, device)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    env = visin_env(visin_mode, spool_dir)
    upload = visin_mode != "disabled"
    bench_device = "cuda" if device.startswith("cuda") else "cpu"

    # Train: one epoch log and one checkpoint, named by the same epoch UUID
    run_stage("train", command(model, "train", config_path), env, logdir)
    epoch_log = only(logdir / "epochs" / "epoch_0_*.json")
    with open(epoch_log) as f:
        epoch = json.load(f)
    training_uuid, epoch_uuid = epoch["training_uuid"], epoch["epoch_uuid"]
    only(logdir / "checkpoints" / f"epoch_0_{epoch_uuid}.pth")

    # Test: every weather condition of the sample was evaluated
    run_stage("test", command(model, "test", config_path), env, logdir)
    with open(only(logdir / "test_results" / "*.json")) as f:
        tested = json.dumps(json.load(f))
    missing = [c for c in WEATHER_CONDITIONS if c not in tested]
    assert not missing, f"test results lack conditions {missing}"

    # Visualize: every kind of image for every frame in the visualization list
    run_stage("visualize", command(model, "visualize", config_path, upload=upload), env, logdir)
    frames = len(read_lines(SAMPLE / "visualizations.txt"))
    for kind in VISUALIZATION_KINDS:
        images = glob.glob(str(logdir / "visualizations" / kind / "*"))
        assert len(images) == frames, f"{kind}: {len(images)} images, expected {frames}"

    # Benchmark: one results file that measured this model
    run_stage("benchmark", command(model, "benchmark", config_path, benchmark_device=bench_device), env, logdir)
    with open(only(logdir / "benchmark" / "*.json")) as f:
        measured = json.load(f)["results"]
    assert [r["backbone"] for r in measured] == [preset(model)["CLI"]["backbone"]]
    assert measured[0]["fps"] > 0

    if visin_mode == "offline":
        check_offline_reports(spool_dir, training_uuid, frames)
    elif visin_mode == "online":
        check_online_reports(training_uuid)


def check_offline_reports(spool_dir, training_uuid, frames):
    """Everything was kept on disk for `visin sync`, under the training's UUID."""
    import visin

    pending = visin.pending(spool_dir)
    assert list(pending) == [training_uuid], f"reports kept for runs {list(pending)}, expected {training_uuid}"
    # the run, its config, an epoch, a test result, a benchmark, and the visualizations
    minimum = 5 + frames * len(VISUALIZATION_KINDS)
    assert pending[training_uuid] >= minimum, f"{pending[training_uuid]} reports kept, expected >= {minimum}"


def check_online_reports(training_uuid):
    """Visin has the run with its epoch and test result."""
    import visin

    import visin_fusion.integrations.visin  # noqa: F401  loads caller env and the default URL

    api = visin.Api()
    training = api.training(training_uuid)
    assert training, f"training {training_uuid} not found in Visin"
    assert api.epochs(training_uuid), "no epochs in Visin"
    assert api.test_results(training_uuid), "no test results in Visin"


@pytest.mark.e2e
def test_any_dataset_name(tmp_path, device):
    """Nothing depends on the dataset being zod, waymo or iseauto: a renamed copy trains and tests."""
    import shutil

    dataset = tmp_path / "custom"
    shutil.copytree(SAMPLE, dataset)
    manifest = json.loads((dataset / "dataset.json").read_text())
    manifest["name"] = "custom_dataset"
    (dataset / "dataset.json").write_text(json.dumps(manifest))

    logdir = tmp_path / "logs"
    logdir.mkdir()
    config = make_config("clft", "rgb", logdir, device)
    config["Dataset"]["dataset_root"] = str(dataset)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    env = visin_env("disabled", tmp_path / "visin")

    # Before, training failed (no IoU function for unknown names) and CLFT's tester shifted class
    # indices only for known names
    run_stage("train", command("clft", "train", config_path), env, logdir)
    run_stage("test", command("clft", "test", config_path), env, logdir)
    with open(only(logdir / "test_results" / "*.json")) as f:
        tested = f.read()
    assert all(condition in tested for condition in WEATHER_CONDITIONS)
