"""Pipeline events and callbacks.

A callback subclasses :class:`Callback` and overrides the events it cares about. Name it in a config to have
every stage load it (stages are separate processes, so it is named, not passed):

    "General": {"callbacks": ["my_package.logging:MyCallback"]}

It is built as ``MyCallback(config)``, with the resolved config, after the Visin callback when reporting is on.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from visin_fusion.integrations.settings import pipeline_key_present


@dataclass(frozen=True)
class RunStart:
    """Training is starting. A callback that creates the run stores its UUID in ``state["training_uuid"]``."""

    config: Mapping
    state: dict


@dataclass(frozen=True)
class EpochEnd:
    """An epoch finished; ``results`` is ``{"train": {...}, "val": {...}, "system_info": {...}}``."""

    config: Mapping
    epoch: int
    epoch_uuid: str | None
    results: Mapping
    learning_rate: float
    epoch_time: float


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint was written for ``epoch``."""

    config: Mapping
    epoch: int
    epoch_uuid: str | None


@dataclass(frozen=True)
class TestEnd:
    """A checkpoint was tested; ``results`` maps each test set (and ``overall``) to its scores."""

    config: Mapping
    epoch: int
    epoch_uuid: str | None
    results: Mapping
    test_uuid: str | None = None

    __test__ = False


@dataclass(frozen=True)
class Visualization:
    """One image was rendered into ``output_dir``."""

    config: Mapping
    epoch: int
    epoch_uuid: str | None
    output_dir: str
    image_name: str


@dataclass(frozen=True)
class Benchmark:
    """Benchmark results, linked to the run and checkpoint they measured."""

    results: Sequence[Mapping]
    system_info: Mapping
    training_uuid: str | None = None
    epoch: int | None = None
    epoch_uuid: str | None = None


@dataclass(frozen=True)
class RunEnd:
    """The stage finished; ``error`` is the exception that ended it, or ``None``."""

    config: Mapping
    error: BaseException | None = None


EVENT_HANDLERS = {
    RunStart: "on_run_start",
    EpochEnd: "on_epoch_end",
    Checkpoint: "on_checkpoint",
    TestEnd: "on_test_end",
    Visualization: "on_visualization",
    Benchmark: "on_benchmark",
    RunEnd: "on_run_end",
}


class Callback:
    """Override the ``on_*`` methods for the events you care about; each receives one event dataclass."""

    def on_run_start(self, event: RunStart) -> None:
        """Training is starting."""

    def on_epoch_end(self, event: EpochEnd) -> None:
        """An epoch finished."""

    def on_checkpoint(self, event: Checkpoint) -> None:
        """A checkpoint was written."""

    def on_test_end(self, event: TestEnd) -> None:
        """A checkpoint was tested."""

    def on_visualization(self, event: Visualization) -> None:
        """An image was rendered."""

    def on_benchmark(self, event: Benchmark) -> None:
        """Benchmark results are ready."""

    def on_run_end(self, event: RunEnd) -> None:
        """The stage finished."""


class Events:
    """The callbacks of a run; ``emit`` hands an event dataclass to each callback's matching method."""

    def __init__(self, callbacks: Sequence[Callback] = ()) -> None:
        self.callbacks = list(callbacks)

    def emit(self, event: object) -> None:
        """Call ``on_<event>`` of every callback with ``event``."""
        handler = EVENT_HANDLERS[type(event)]
        for callback in self.callbacks:
            getattr(callback, handler)(event)


def load_callback(spec: str, config) -> Callback:
    """Build the callback ``"package.module:ClassName"`` names, or explain what is wrong with the setting."""
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(f'General.callbacks: {spec!r} must look like "package.module:ClassName"')
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"General.callbacks: cannot load {spec!r}: {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, Callback)):
        raise TypeError(f"General.callbacks: {spec!r} is not a subclass of visin_fusion.engine.callbacks.Callback")
    return cls(config)


def configured_callbacks(config=None, *, visin=True):
    """The callbacks a run uses: Visin reporting when a pipeline key is set, then ``General.callbacks``.

    Local runs need no Visin packages. ``visin=False`` leaves Visin out, for a stage that reports to it only
    on request (visualize uploads only with ``--upload``) but still informs the config's own callbacks.
    """
    callbacks = []
    if visin and pipeline_key_present():
        try:
            from visin_fusion.integrations.callback import VisinCallback
        except ImportError as exc:
            raise ImportError("A Visin pipeline key is set; pip install visin-fusion[visin]") from exc
        callbacks.append(VisinCallback(config))
    callbacks += [load_callback(spec, config) for spec in (config or {}).get("General", {}).get("callbacks", [])]
    return Events(callbacks)
