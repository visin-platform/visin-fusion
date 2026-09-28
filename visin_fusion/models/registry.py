"""Model names and the config adapter used by the command line pipeline."""
from torch import nn

from .api import CLFT, CLFTv2, DeepLabV3Plus, Mask2FormerFusion, MaskFormerFusion

MODEL_CLASSES = {
    'clft': CLFT,
    'clftv2': CLFTv2,
    'maskformer': MaskFormerFusion,
    'mask2former': Mask2FormerFusion,
    'deeplabv3plus': DeepLabV3Plus,
}
SECTIONS = {
    'clft': 'CLFT', 'clftv2': 'CLFTv2', 'maskformer': 'MaskFormer',
    'mask2former': 'Mask2Former', 'deeplabv3plus': 'DeepLabV3Plus',
}


def model_section(config):
    return config[SECTIONS[config['CLI']['backbone']]]


def from_config(config, pretrained=None):
    """Construct a public model from a validated pipeline config."""
    name = config['CLI']['backbone']
    cls = MODEL_CLASSES[name]
    section = model_section(config)
    options = dict(section)
    options['num_classes'] = len(config['Dataset']['train_classes'])
    options['mode'] = config['CLI']['mode']
    options['training_options'] = dict(section)
    if pretrained is not None:
        options['pretrained'] = pretrained
    if name == 'clft':
        options['image_size'] = config['Dataset']['transforms']['resize']
    if name in ('clft', 'clftv2'):
        options['reassembles'] = section['reassembles']
    if name in ('maskformer', 'mask2former'):
        options['backbone'] = section['model_timm']
    if name == 'deeplabv3plus':
        options['fusion_strategy'] = (section.get('fusion_strategy') or
                                      section.get('fusion_type') or 'residual_average')
    # Only constructor arguments are forwarded; training options stay on the model.
    import inspect
    names = inspect.signature(cls).parameters
    return cls(**{key: value for key, value in options.items() if key in names})


build_model = from_config  # compatibility for older stage scripts


def forward(model, config, rgb, lidar):
    return model.raw_forward(rgb, lidar)


def segmentation(outputs, config):
    return outputs[0] if config['CLI']['backbone'] == 'deeplabv3plus' else outputs[1]


def segment(model, config, rgb, lidar):
    return model(rgb, lidar)


class Segmenter(nn.Module):
    def __init__(self, model, config=None):
        super().__init__()
        self.model = model

    def forward(self, rgb, lidar):
        return self.model(rgb, lidar)


def training_setup(config, model, device):
    """Compatibility adapter; the selected model owns its defaults."""
    classes = sorted(config['Dataset']['train_classes'], key=lambda c: c['index'])
    weights = [item['weight'] for item in classes]
    return model.training_setup(weights, device=device, epochs=config['General']['epochs'])
