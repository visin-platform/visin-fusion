"""Models and datasets on the Hugging Face Hub, against a stand-in for ``huggingface_hub``."""

import json
import logging
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from visin_fusion import __version__, cli, hub, space
from visin_fusion.config.config import prepare_config
from visin_fusion.data import hf_datasets
from visin_fusion.hub import HubRef, download_checkpoint, parse_ref, push_best_checkpoint, stage_for_hub
from visin_fusion.inference import Predictor, load_checkpoint
from visin_fusion.models.registry import from_config
from visin_fusion.sample import SAMPLE_DIR
from visin_fusion.utils.helpers import save_model_dict

COMMIT = "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"
FRAME = "camera/frame_000004.png"


class FakeHub(types.ModuleType):
    """Records what was asked of the Hub and serves a checkpoint, files and a dataset from disk."""

    def __init__(self):
        super().__init__("huggingface_hub")
        self.files = [hub.CHECKPOINT_NAME, hub.CONFIG_NAME]
        self.checkpoint = None
        self.dataset = None
        self.downloads = []
        self.snapshots = []
        self.created = []
        self.uploads = []
        self.failure = None

    def hf_hub_download(self, repo, filename, revision=None):
        self.downloads.append((repo, filename, revision))
        return str(self.checkpoint)

    def snapshot_download(self, **kwargs):
        self.snapshots.append(kwargs)
        return str(self.dataset)

    def HfApi(self):
        return FakeApi(self)


class FakeApi:
    def __init__(self, fake):
        self.fake = fake

    def list_repo_files(self, repo, revision=None):
        return self.fake.files

    def create_repo(self, **kwargs):
        self.fake.created.append(kwargs)

    def upload_folder(self, **kwargs):
        if self.fake.failure:
            raise self.fake.failure
        folder = Path(kwargs["folder_path"])
        self.fake.uploads.append({**kwargs, "files": sorted(path.name for path in folder.iterdir())})
        return types.SimpleNamespace(oid=COMMIT)


@pytest.fixture
def fake_hub(monkeypatch):
    fake = FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


@pytest.fixture(scope="module")
def config():
    config = prepare_config({"extends": "clftv2", "Dataset": {"dataset_root": str(SAMPLE_DIR)}})
    config["CLFTv2"]["pretrained"] = False
    return config


@pytest.fixture(scope="module")
def trained(config, tmp_path_factory):
    """A log directory holding one checkpoint and its epoch log, as the train stage leaves them."""
    logdir = tmp_path_factory.mktemp("run")
    config = {**config, "Log": {"logdir": str(logdir)}}
    model = from_config(config, pretrained=False)
    save_model_dict(config, 3, model, torch.optim.SGD(model.parameters(), lr=0.1), epoch_uuid="abc")
    epochs = logdir / "epochs"
    epochs.mkdir()
    (epochs / "epoch_3_abc.json").write_text(
        json.dumps({"training_uuid": "run-1", "results": {"val": {"mean_iou": 0.61}}})
    )
    return config, next((logdir / "checkpoints").glob("*.pth"))


def settings(config, logdir, **general):
    return {**config, "General": {**config["General"], **general}, "Log": {"logdir": str(logdir)}}


class TestReferences:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("hf://acme/clftv2-zod", HubRef("acme/clftv2-zod")),
            ("acme/clftv2-zod", HubRef("acme/clftv2-zod")),
            ("hf:acme/zod-png", HubRef("acme/zod-png")),
            (f"hf://acme/clftv2-zod@{COMMIT}", HubRef("acme/clftv2-zod", COMMIT)),
            ("hf://acme/m@v1/weights/best.pth", HubRef("acme/m", "v1", "weights/best.pth")),
            ("hf://acme/m/best.pth", HubRef("acme/m", None, "best.pth")),
        ],
    )
    def test_a_reference_names_the_repo_the_commit_and_the_file(self, text, expected):
        assert parse_ref(text) == expected

    @pytest.mark.parametrize("text", ["", "hf://", "hf://acme", "hf://acme/"])
    def test_anything_else_is_refused_with_the_expected_form(self, text):
        with pytest.raises(ValueError, match=r"hf://org/name\[@commit\]\[/file\]"):
            parse_ref(text)


class TestDownload:
    def test_the_conventional_checkpoint_is_downloaded_at_the_commit(self, fake_hub, tmp_path):
        fake_hub.checkpoint = tmp_path / "checkpoint.pth"
        assert download_checkpoint(f"hf://acme/m@{COMMIT}") == fake_hub.checkpoint
        assert fake_hub.downloads == [("acme/m", "checkpoint.pth", COMMIT)]

    def test_a_repo_with_one_checkpoint_under_another_name_uses_it(self, fake_hub, tmp_path):
        fake_hub.files = ["README.md", "epoch_9.pt"]
        fake_hub.checkpoint = tmp_path / "epoch_9.pt"
        download_checkpoint("hf://acme/m")
        assert fake_hub.downloads == [("acme/m", "epoch_9.pt", None)]

    def test_a_named_file_needs_no_guessing(self, fake_hub, tmp_path):
        fake_hub.files = ["a.pth", "b.pth"]
        fake_hub.checkpoint = tmp_path / "b.pth"
        download_checkpoint("hf://acme/m/b.pth")
        assert fake_hub.downloads == [("acme/m", "b.pth", None)]

    @pytest.mark.parametrize("files", [["README.md"], ["a.pth", "b.pth"]])
    def test_no_checkpoint_or_several_is_an_error_that_says_which_to_name(self, fake_hub, files):
        fake_hub.files = files
        with pytest.raises(FileNotFoundError, match=r"hf://acme/m/<file>"):
            download_checkpoint("hf://acme/m")

    def test_a_missing_extra_is_named(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ImportError, match=r"visin-fusion\[hf\]"):
            download_checkpoint("hf://acme/m")


class TestStaging:
    def test_inference_weights_and_config_are_staged_without_the_optimizer(self, trained, tmp_path):
        _, checkpoint = trained
        staged = stage_for_hub(checkpoint, tmp_path / "repo")
        assert sorted(path.name for path in staged.iterdir()) == ["checkpoint.pth", "config.json"]
        state = torch.load(staged / "checkpoint.pth", weights_only=True)
        assert set(state) == {"epoch", "model_state_dict", "model_info"} and state["epoch"] == 3
        assert json.loads((staged / "config.json").read_text())["backbone"] == "clftv2"
        assert (staged / "checkpoint.pth").stat().st_size < checkpoint.stat().st_size
        assert load_checkpoint(staged / "checkpoint.pth")[1]["backbone"] == "clftv2"


class TestPredictor:
    def test_a_hub_model_predicts_like_a_local_checkpoint(self, fake_hub, trained, tmp_path):
        _, checkpoint = trained
        fake_hub.checkpoint = stage_for_hub(checkpoint, tmp_path / "repo") / "checkpoint.pth"
        predictor = Predictor.from_pretrained(f"hf://acme/m@{COMMIT}", device="cpu")
        assert fake_hub.downloads == [("acme/m", "checkpoint.pth", COMMIT)]
        camera = SAMPLE_DIR / FRAME
        mask = predictor.predict(camera, SAMPLE_DIR / FRAME.replace("camera", "lidar_png", 1))
        assert mask.shape == Image.open(camera).size[::-1]

    def test_a_revision_and_file_can_be_given_beside_the_reference(self, fake_hub, trained, tmp_path):
        _, checkpoint = trained
        fake_hub.checkpoint = checkpoint
        Predictor.from_pretrained("acme/m", revision="v2", filename="epoch.pth", device="cpu")
        assert fake_hub.downloads == [("acme/m", "epoch.pth", "v2")]

    def test_the_command_line_reads_a_hub_model(self, fake_hub, trained, tmp_path, capsys):
        _, checkpoint = trained
        fake_hub.checkpoint = stage_for_hub(checkpoint, tmp_path / "repo") / "checkpoint.pth"
        output = tmp_path / "out"
        cli.predict(
            [
                "--checkpoint", f"hf://acme/m@{COMMIT}",
                "--input", str(SAMPLE_DIR / FRAME),
                "--output", str(output),
                "--device", "cpu",
            ]
        )  # fmt: skip
        assert (output / "frame_000004_mask.png").exists()
        assert "Wrote 1 masks" in capsys.readouterr().out

    def test_the_command_line_says_when_the_extra_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(SystemExit, match=r"visin-fusion\[hf\]"):
            cli.predict(["--checkpoint", "hf://acme/m", "--input", str(SAMPLE_DIR / FRAME), "--output", str(tmp_path)])


class TestDatasets:
    def test_a_pinned_hub_dataset_is_downloaded_as_a_dataset_root(self, fake_hub):
        fake_hub.dataset = SAMPLE_DIR
        config = prepare_config({"extends": "clftv2", "Dataset": {"dataset_root": f"hf:acme/zod@{COMMIT}"}})
        assert Path(config["Dataset"]["dataset_root"]) == SAMPLE_DIR
        assert fake_hub.snapshots == [{"repo_id": "acme/zod", "repo_type": "dataset", "revision": COMMIT}]

    def test_an_unpinned_dataset_works_but_warns_that_a_later_run_may_differ(self, fake_hub, caplog):
        fake_hub.dataset = SAMPLE_DIR
        with caplog.at_level(logging.WARNING, logger="visin_fusion.data.hf_datasets"):
            assert hf_datasets.resolve_root("hf:acme/zod") == str(SAMPLE_DIR)
        assert "not pinned to a commit" in caplog.text

    def test_a_file_inside_a_repo_is_not_a_dataset_root(self, fake_hub):
        with pytest.raises(ValueError, match="whole repo"):
            hf_datasets.resolve_root("hf:acme/zod/train.txt")

    def test_a_missing_extra_is_named(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ImportError, match=r"visin-fusion\[hf\]"):
            hf_datasets.resolve_root(f"hf:acme/zod@{COMMIT}")


class TestPublishing:
    def test_nothing_is_published_without_a_repo(self, fake_hub, trained):
        config, _ = trained
        assert push_best_checkpoint(config) is None
        assert not fake_hub.uploads

    def test_the_best_checkpoint_is_staged_and_uploaded_to_a_private_repo(self, fake_hub, trained, monkeypatch):
        config, _ = trained
        monkeypatch.setattr(hub, "_visin_run", lambda: None)
        revision = push_best_checkpoint(settings(config, config["Log"]["logdir"], hub_repo="acme/m"))
        assert revision == COMMIT
        assert fake_hub.created == [{"repo_id": "acme/m", "repo_type": "model", "private": True, "exist_ok": True}]
        assert fake_hub.uploads[0]["files"] == ["checkpoint.pth", "config.json"]

    def test_a_public_repo_must_be_asked_for(self, fake_hub, trained, monkeypatch):
        config, _ = trained
        monkeypatch.setattr(hub, "_visin_run", lambda: None)
        push_best_checkpoint(settings(config, config["Log"]["logdir"], hub_repo="acme/m", hub_private=False))
        assert fake_hub.created[0]["private"] is False

    def test_with_visin_reporting_on_the_run_uploads_and_links_it(self, fake_hub, trained, monkeypatch):
        config, _ = trained
        calls = []

        class Run:
            def log_model(self, folder, repo, **kwargs):
                calls.append((sorted(path.name for path in Path(folder).iterdir()), repo, kwargs))
                return COMMIT

        monkeypatch.setattr(hub, "_visin_run", lambda: Run())
        assert push_best_checkpoint(settings(config, config["Log"]["logdir"], hub_repo="acme/m")) == COMMIT
        assert calls == [(["checkpoint.pth", "config.json"], "acme/m", {"epoch": 3, "private": True})]
        assert not fake_hub.uploads

    def test_a_refused_upload_is_logged_and_never_fails_the_run(self, fake_hub, trained, monkeypatch, caplog):
        config, _ = trained
        monkeypatch.setattr(hub, "_visin_run", lambda: None)
        fake_hub.failure = RuntimeError("403 forbidden")
        with caplog.at_level(logging.WARNING, logger="visin_fusion.hub"):
            assert push_best_checkpoint(settings(config, config["Log"]["logdir"], hub_repo="acme/m")) is None
        assert "403 forbidden" in caplog.text

    def test_no_checkpoint_to_publish_is_said_not_raised(self, fake_hub, config, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="visin_fusion.hub"):
            assert push_best_checkpoint(settings(config, tmp_path, hub_repo="acme/m")) is None
        assert "no checkpoint was found" in caplog.text

    def test_the_visin_run_is_used_only_when_reporting_is_on_and_visin_can_link_models(self, monkeypatch):
        class Visin(types.ModuleType):
            def __init__(self, run):
                super().__init__("visin")
                self.run = run

            def get_run(self):
                return self.run

        run = types.SimpleNamespace(enabled=True, log_model=lambda *a, **k: None)
        for module, expected in [
            (Visin(run), run),
            (Visin(types.SimpleNamespace(enabled=False, log_model=print)), None),
            (Visin(types.SimpleNamespace(enabled=True)), None),
            (types.ModuleType("visin"), None),
            (None, None),
        ]:
            monkeypatch.setitem(sys.modules, "visin", module)
            assert hub._visin_run() is expected


class TestSpace:
    def test_the_demo_app_loads_the_pinned_model_and_names_the_hub_repo(self, tmp_path):
        folder = space.build_space(HubRef("acme/clftv2-zod", COMMIT), tmp_path / "demo")
        app = (folder / "app.py").read_text()
        assert f'MODEL = "hf://acme/clftv2-zod@{COMMIT}"' in app
        assert "Predictor.from_pretrained(MODEL" in app
        compile(app, "app.py", "exec")
        assert "sdk: gradio" in (folder / "README.md").read_text()
        assert "  - acme/clftv2-zod" in (folder / "README.md").read_text()
        assert (folder / "requirements.txt").read_text() == f"visin-fusion[hf]>={__version__}\ngradio\n"

    def test_a_model_without_a_commit_is_pinned_to_its_newest_one(self, fake_hub):
        fake_hub.sha = COMMIT
        files = []
        original = FakeApi.upload_folder

        def capture(self, **kwargs):
            files.append((Path(kwargs["folder_path"]) / "app.py").read_text())
            return original(self, **kwargs)

        FakeApi.model_info = lambda self, repo: types.SimpleNamespace(sha=self.fake.sha)
        try:
            FakeApi.upload_folder = capture
            url = space.publish_space("hf://acme/clftv2-zod", "acme/clftv2-zod-demo")
        finally:
            FakeApi.upload_folder = original
            del FakeApi.model_info
        assert url == "https://huggingface.co/spaces/acme/clftv2-zod-demo"
        assert f"@{COMMIT}" in files[0]
        assert fake_hub.created == [
            {
                "repo_id": "acme/clftv2-zod-demo",
                "repo_type": "space",
                "space_sdk": "gradio",
                "private": False,
                "exist_ok": True,
            }
        ]

    def test_a_space_can_be_private(self, fake_hub):
        space.publish_space(f"hf://acme/m@{COMMIT}", "acme/demo", private=True)
        assert fake_hub.created[0]["private"] is True

    def test_the_command_prints_the_space_url_and_explains_a_missing_extra(self, fake_hub, capsys, monkeypatch):
        cli.main(["space", "--model", f"hf://acme/m@{COMMIT}", "--space", "acme/demo"])
        assert capsys.readouterr().out.strip() == "https://huggingface.co/spaces/acme/demo"
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(SystemExit, match=r"visin-fusion\[hf\]"):
            cli.main(["space", "--model", "hf://acme/m", "--space", "acme/demo"])


@pytest.mark.parametrize("revision", [COMMIT, None])
def test_space_preserves_the_checkpoint_path_in_platform_references(fake_hub, monkeypatch, revision):
    apps = []
    upload = FakeApi.upload_folder

    def capture(self, **kwargs):
        apps.append((Path(kwargs["folder_path"]) / "app.py").read_text())
        return upload(self, **kwargs)

    monkeypatch.setattr(FakeApi, "upload_folder", capture)
    monkeypatch.setattr(FakeApi, "model_info", lambda self, repo: types.SimpleNamespace(sha=COMMIT), raising=False)
    ref = "hf://acme/m" + (f"@{revision}" if revision else "") + "/checkpoints/best.pth"
    space.publish_space(ref, "acme/demo")
    assert f'MODEL = "hf://acme/m@{COMMIT}/checkpoints/best.pth"' in apps[0]
    compile(apps[0], "app.py", "exec")


def test_space_escapes_checkpoint_paths_in_generated_python(tmp_path):
    folder = space.build_space(HubRef("acme/m", COMMIT, 'checkpoints/"best" model.pth'), tmp_path)
    app = (folder / "app.py").read_text()
    compile(app, "app.py", "exec")
    model_line = next(line for line in app.splitlines() if line.startswith("MODEL = "))
    assert json.loads(model_line.removeprefix("MODEL = ")) == f'hf://acme/m@{COMMIT}/checkpoints/"best" model.pth'


@pytest.mark.parametrize("filename", ["runs/one/checkpoint.pth", "runs/one/best.pt"])
def test_a_platform_folder_reference_selects_a_checkpoint_inside_that_folder(fake_hub, tmp_path, filename):
    fake_hub.files = ["checkpoint.pth", "runs/other/best.pth", filename]
    fake_hub.checkpoint = tmp_path / "best.pth"
    download_checkpoint(f"hf://acme/m@{COMMIT}/runs/one")
    assert fake_hub.downloads == [("acme/m", filename, COMMIT)]


def test_a_folder_with_several_checkpoints_needs_an_explicit_file(fake_hub):
    fake_hub.files = ["checkpoint.pth", "runs/one/a.pt", "runs/one/b.pt"]
    with pytest.raises(FileNotFoundError, match="not exactly one"):
        download_checkpoint(f"hf://acme/m@{COMMIT}/runs/one")
    assert not fake_hub.downloads
