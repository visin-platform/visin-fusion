"""Compatibility imports for visin_fusion.engine."""
from pathlib import Path
from visin_fusion.engine import __path__ as _target_path
__path__ = [str(Path(__file__).parent), *_target_path]
