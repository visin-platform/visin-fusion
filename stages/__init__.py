"""Compatibility imports for visin_fusion.engine.stages."""
from pathlib import Path
from visin_fusion.engine.stages import __path__ as _target_path
__path__ = [str(Path(__file__).parent), *_target_path]
