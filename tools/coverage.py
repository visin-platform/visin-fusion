"""Run the full-package unit and end-to-end coverage gate locally."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="visin-fusion-coverage-") as directory:
        env = os.environ.copy()
        env["COVERAGE_FILE"] = str(Path(directory) / ".coverage")
        env["COVERAGE_PROCESS_START"] = str(ROOT / ".coveragerc")
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(ROOT / "tools" / "coverage_startup"), str(ROOT), env.get("PYTHONPATH")))
        )
        commands = [
            ["run", "-m", "pytest", "tests/unit"],
            [
                "run",
                "-m",
                "pytest",
                "tests/e2e",
                "--models",
                "clft,clftv2,maskformer,mask2former,deeplab",
                "--modes",
                "fusion",
                "--device",
                "cpu",
                "--visin",
                "offline",
            ],
            ["combine"],
            ["report", "--fail-under=90.1"],
        ]
        for args in commands:
            subprocess.run([sys.executable, "-m", "coverage", *args], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
