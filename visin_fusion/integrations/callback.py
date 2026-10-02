"""Visin reporting for engine events."""

from __future__ import annotations

from visin_fusion.engine.callbacks import (
    Benchmark,
    Callback,
    EpochEnd,
    RunEnd,
    RunStart,
    TestEnd,
    Visualization,
)
from visin_fusion.integrations import visin as service
from visin_fusion.integrations.visualization_uploader import queue_visualizations


class VisinCallback(Callback):
    """Reports a run's epochs, test results, visualizations and benchmarks to Visin."""

    def __init__(self, config=None):
        self.config = config
        self.run = None

    def on_run_start(self, event: RunStart) -> None:
        """Create (or resume) the Visin run and hand its UUID back through ``event.state``."""
        self.config = event.config
        self.run, event.state["training_uuid"] = service.start_training_run(
            self.config, model=self.config["CLI"]["backbone"]
        )
        self.run.__enter__()

    def on_epoch_end(self, event: EpochEnd) -> None:
        """Log the epoch's metrics to the run."""
        if self.run is not None:
            self.run.log_epoch(
                event.epoch, event.results, learning_rate=event.learning_rate, epoch_time=event.epoch_time
            )

    def on_test_end(self, event: TestEnd) -> None:
        """Report test results on the tested checkpoint's epoch."""
        service.report_test_results(
            event.config, event.epoch, event.epoch_uuid, event.results, test_uuid=event.test_uuid
        )

    def on_visualization(self, event: Visualization) -> None:
        """Queue one rendered image for upload to the checkpoint's epoch."""
        if self.run is None:
            self.run = service.attach_to_training(event.config["Log"]["logdir"], epoch_uuid=event.epoch_uuid)
        queue_visualizations(self.run, event.epoch, event.epoch_uuid, event.output_dir, event.image_name)

    def on_benchmark(self, event: Benchmark) -> None:
        """Report benchmark results, linked to the run they measured."""
        service.report_benchmark(
            event.results,
            event.system_info,
            training_uuid=event.training_uuid,
            epoch=event.epoch,
            epoch_uuid=event.epoch_uuid,
        )

    def on_run_end(self, event: RunEnd) -> None:
        """Close the run, marking it failed when the stage ended with an error."""
        if self.run is not None:
            error = event.error
            self.run.__exit__(type(error) if error else None, error, None)
            self.run = None
