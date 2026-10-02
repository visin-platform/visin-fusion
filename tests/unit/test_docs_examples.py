"""The documentation's examples run: config JSON validates, and Python blocks marked ``<!-- doctest -->`` execute.

A block is marked by putting ``<!-- doctest -->`` on the line before its fence. Marked blocks must be
self-contained (random tensors, ``pretrained=False``), since they run for real.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from visin_fusion.config.config import prepare_config
from visin_fusion.sample import SAMPLE_DIR

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
PYTHON_BLOCK = re.compile(r"<!-- doctest -->\n```python\n(.*?)```", re.DOTALL)
JSON_BLOCK = re.compile(r"```json[^\n]*\n(.*?)```", re.DOTALL)


def config_blocks() -> list:
    found = []
    for page in PAGES:
        for index, text in enumerate(JSON_BLOCK.findall(page.read_text())):
            try:
                block = json.loads(text)
            except json.JSONDecodeError:
                continue
            extends = block.get("extends", "") if isinstance(block, dict) else ""
            if extends and not extends.startswith("."):
                found.append(pytest.param(page, text, id=f"{page.name}-{index}"))
    return found


def python_blocks() -> list:
    found = []
    for page in PAGES:
        for index, text in enumerate(PYTHON_BLOCK.findall(page.read_text())):
            found.append(pytest.param(text, id=f"{page.name}-{index}"))
    return found


def test_the_docs_have_examples_to_check():
    assert len(config_blocks()) >= 2
    assert len(python_blocks()) >= 3


@pytest.mark.parametrize(("page", "text"), config_blocks())
def test_a_config_in_the_docs_is_valid(page, text, tmp_path):
    """A config that names its own classes must work with no dataset.json; one that does not relies on it."""
    root = tmp_path / "my_dataset"
    keep_manifest = "train_classes" not in text
    shutil.copytree(SAMPLE_DIR, root, ignore=None if keep_manifest else shutil.ignore_patterns("dataset.json"))
    text = text.replace("/data/my_dataset", str(root)).replace("my_package.models", "tests.unit.custom_model_plugin")
    prepare_config(json.loads(text))


@pytest.mark.parametrize("code", python_blocks())
def test_a_marked_python_example_runs(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "<docs example>", "exec"), {"__name__": "__docs__"})  # noqa: S102
