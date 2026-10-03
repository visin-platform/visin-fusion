"""pyproject.toml: a plain `pip install visin-fusion` installs everything the command line imports."""

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "visin_fusion"
OPTIONAL = {"visin", "dotenv", "huggingface_hub"}
DISTRIBUTIONS = {"PIL": "pillow", "cv2": "opencv-python", "GPUtil": "gputil", "pynvml": "nvidia-ml-py"}


def imported_modules() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                found.setdefault(name.split(".")[0], path)
    return found


def test_every_third_party_import_is_a_core_dependency():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower().replace("_", "-") for d in project["dependencies"]}
    third_party = {m: p for m, p in imported_modules().items() if m not in sys.stdlib_module_names | {"visin_fusion"}}
    missing = {
        module: str(path.relative_to(ROOT))
        for module, path in third_party.items()
        if module not in OPTIONAL and DISTRIBUTIONS.get(module, module).lower() not in declared
    }
    assert not missing, f"imported but not in [project].dependencies: {missing}"
