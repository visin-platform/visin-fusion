"""Visin reporting for engine events."""
from visin_fusion.engine.callbacks import Callback
from visin_fusion.integrations import visin as service
from visin_fusion.integrations.visualization_uploader import queue_visualizations


class VisinCallback(Callback):
    def __init__(self, config=None):
        self.config = config
        self.run = None

    def on_run_start(self, **event):
        self.config = event['config']
        self.run, event['state']['training_uuid'] = service.start_training_run(
            self.config, model=self.config['CLI']['backbone'])
        self.run.__enter__()

    def on_epoch_end(self, **event):
        if self.run is not None:
            self.run.log_epoch(event['epoch'], event['results'],
                               learning_rate=event.get('learning_rate'),
                               epoch_time=event.get('epoch_time'))

    def on_test_end(self, **event):
        service.report_test_results(event['config'], event['epoch'], event['epoch_uuid'],
                                    event['results'], test_uuid=event.get('test_uuid'))

    def on_visualization(self, **event):
        if self.run is None:
            self.run = service.attach_to_training(event['config']['Log']['logdir'],
                                                  epoch_uuid=event['epoch_uuid'])
        queue_visualizations(self.run, event['epoch'], event['epoch_uuid'],
                             event['output_dir'], event['image_name'])

    def on_benchmark(self, **event):
        service.report_benchmark(event['results'], event['system_info'],
                                 training_uuid=event.get('training_uuid'),
                                 epoch=event.get('epoch'), epoch_uuid=event.get('epoch_uuid'))

    def on_run_end(self, **event):
        if self.run is not None:
            error = event.get('error')
            self.run.__exit__(type(error) if error else None, error, None)
            self.run = None
