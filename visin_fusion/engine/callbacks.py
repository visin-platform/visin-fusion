"""Pipeline events and optional reporting callbacks."""

from visin_fusion.integrations.settings import pipeline_key_present


class Callback:
    """Override any event method needed by an integration."""

    def on_run_start(self, **event):
        pass

    def on_epoch_end(self, **event):
        pass

    def on_checkpoint(self, **event):
        pass

    def on_test_end(self, **event):
        pass

    def on_visualization(self, **event):
        pass

    def on_benchmark(self, **event):
        pass

    def on_run_end(self, **event):
        pass


class Events:
    def __init__(self, callbacks=()):
        self.callbacks = list(callbacks)

    def emit(self, event, **data):
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                method(**data)


# Retain this name for integrations and tests that used the earlier helper.
_pipeline_key_present = pipeline_key_present


def configured_callbacks(config=None):
    """Load Visin only for configured pipelines; local runs need no Visin packages."""
    callbacks = []
    if _pipeline_key_present():
        try:
            from visin_fusion.integrations.callback import VisinCallback
        except ImportError as exc:
            raise ImportError("A Visin pipeline key is set; pip install visin-fusion[visin]") from exc
        callbacks.append(VisinCallback(config))
    return Events(callbacks)
