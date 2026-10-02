"""visin_fusion/config/config_schema.py: every shipped config is valid, and typical mistakes are caught."""

import copy
import json
import os
from pathlib import Path

import pytest

from visin_fusion.config.config import prepare_config, resolve_extends
from visin_fusion.config.config_schema import Config, ConfigError, validate_config

REPO = Path(__file__).resolve().parents[2]
# Dataset fragments are inputs to manifest generation, not complete run configs.
CONFIGS = sorted(p for p in (REPO / "configs").rglob("*.json") if "datasets" not in p.relative_to(REPO).parts)
QUICKSTART = REPO / "configs" / "quickstart.json"


def load(path):
    path = Path(path)
    with open(path) as f:
        config = resolve_extends(json.load(f), str(path.parent))
    root = config.get("Dataset", {}).get("dataset_root")
    if root and not Path(os.path.expandvars(root)).is_absolute():
        config["Dataset"]["dataset_root"] = str(REPO / root)
    return prepare_config(config, str(path.parent))


@pytest.fixture
def dataset_roots(tmp_path, monkeypatch):
    """Provide the manifests external example datasets supply in a real run."""
    for path in sorted((REPO / "configs" / "datasets").glob("*.json")):
        dataset = json.loads(path.read_text())["Dataset"]
        root = tmp_path / path.stem
        root.mkdir()
        annotations = (
            ["annotation_fusion", "annotation_camera_only", "annotation_lidar_only"]
            if path.stem == "zod"
            else [dataset["annotation_path"]]
        )
        manifest = {
            "format": 1,
            "name": dataset["name"],
            "classes": dataset["dataset_classes"],
            "train_classes": dataset["train_classes"],
            "normalization": dataset["transforms"],
            "annotations": annotations,
            "splits": {"train": "train.txt", "val": "validation.txt"},
        }
        (root / "dataset.json").write_text(json.dumps(manifest))
        monkeypatch.setenv(f"{path.stem.upper()}_DATA_DIR", str(root))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(Path(p).relative_to(REPO)))
def test_shipped_config_is_valid(path, dataset_roots):
    config = load(path)
    validate_config(config)


@pytest.fixture
def config():
    return load(QUICKSTART)


def invalid(config, match):
    with pytest.raises(ConfigError, match=match):
        validate_config(config)


def test_typo_in_a_key(config):
    config["General"]["epcohs"] = config["General"].pop("epochs")
    invalid(config, r"General\.epcohs")


def test_unknown_section(config):
    config["Genral"] = {}
    invalid(config, "Genral")


def test_mode_the_backbone_does_not_have(config):
    config["CLI"]["mode"] = "fusion"  # validate_config alone does not translate; utils.config does
    invalid(config, "mode 'fusion' is not one of clftv2's modes")


def test_unknown_mode(config):
    config["CLI"]["mode"] = "thermal"
    invalid(config, r"CLI\.mode")


def test_missing_model_section(config):
    del config["CLFTv2"]
    invalid(config, "needs a 'CLFTv2' section")


def test_train_class_gap(config):
    config["Dataset"]["train_classes"][-1]["index"] += 1
    invalid(config, "no gaps or duplicates")


def test_train_class_maps_unknown_dataset_class(config):
    config["Dataset"]["train_classes"][1]["dataset_mapping"] = [99]
    invalid(config, r"maps dataset classes \[99\]")


def test_probability_out_of_range(config):
    config["Dataset"]["transforms"]["p_flip"] = 1.5
    invalid(config, r"transforms\.p_flip")


def test_every_problem_is_listed(config):
    broken = copy.deepcopy(config)
    broken["General"]["epochs"] = 0
    broken["General"]["batch_size"] = "eight"
    with pytest.raises(ConfigError) as error:
        validate_config(broken)
    assert "General.epochs" in str(error.value) and "General.batch_size" in str(error.value)


def test_json_schema_exports():
    schema = Config.model_json_schema()
    assert set(schema["required"]) >= {"CLI", "General", "Log", "Dataset"}


def test_config_reference_is_up_to_date():
    from make_config_reference import OUTPUT, render

    assert Path(OUTPUT).read_text() == render(), "run: python tools/make_config_reference.py"


def test_the_repository_quickstart_config_is_the_packaged_one():
    from visin_fusion.sample import QUICKSTART_CONFIG

    on_disk = json.loads(QUICKSTART.read_text())
    expected = {**QUICKSTART_CONFIG, "Dataset": on_disk["Dataset"]}
    assert on_disk == expected
