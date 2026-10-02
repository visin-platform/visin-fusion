"""Camera and LiDAR fusion models and training tools.

``visin_fusion.run`` runs a config through the pipeline (see :mod:`visin_fusion.pipeline`); it is loaded on
first use, so importing the package for its version stays cheap.
"""

from visin_fusion._version import __version__

__all__ = ["__version__", "run"]


def __getattr__(name: str):
    """Load ``run`` (and keep ``import visin_fusion`` light) on first access."""
    if name == "run":
        from visin_fusion.pipeline import run

        return run
    raise AttributeError(f"module 'visin_fusion' has no attribute {name!r}")
