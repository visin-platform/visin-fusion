"""Compatibility imports for visin_fusion.models."""
from pathlib import Path
from visin_fusion.models import __path__ as _target_path
__path__ = [str(Path(__file__).parent), *_target_path]
