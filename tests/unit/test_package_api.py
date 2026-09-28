"""Public package boundaries: model calls, events, dataset hooks, and schema export."""
import inspect
import sys

import pytest
import torch

from visin_fusion.config.config_schema import SCHEMA_VERSION, export_schema
from visin_fusion.data.resolvers import register_resolver, resolve_root
from visin_fusion.engine import callbacks
from visin_fusion.models import CLFT, DeepLabV3Plus, Mask2FormerFusion, MaskFormerFusion, CLFTv2
from visin_fusion.models.api import FusionModel


def test_public_models_have_python_constructors_and_uniform_forward():
    for cls in (CLFT, CLFTv2, MaskFormerFusion, Mask2FormerFusion, DeepLabV3Plus):
        assert issubclass(cls, FusionModel)
        assert 'config' not in inspect.signature(cls).parameters
        assert hasattr(cls, 'training_setup')
    model = DeepLabV3Plus(num_classes=3, mode='rgb', pretrained=False).eval()
    inputs = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert model(inputs, inputs).shape == (1, 3, 64, 64)


def test_local_events_and_dataset_paths_need_no_visin(monkeypatch):
    monkeypatch.setattr(callbacks, '_pipeline_key_present', lambda: False)
    events = callbacks.configured_callbacks()
    assert events.callbacks == []
    events.emit('on_test_end', results={})
    assert resolve_root('/data/sample') == '/data/sample'
    register_resolver('example', lambda root: '/cache/' + root.split(':', 1)[1])
    assert resolve_root('example:sample') == '/cache/sample'


def test_pipeline_key_without_visin_extra_gives_install_hint(monkeypatch):
    monkeypatch.setattr(callbacks, '_pipeline_key_present', lambda: True)
    monkeypatch.setitem(sys.modules, 'visin_fusion.integrations.callback', None)
    with pytest.raises(ImportError, match=r'pip install visin-fusion\[visin\]'):
        callbacks.configured_callbacks()


def test_exported_schema_has_a_stable_version():
    schema = export_schema()
    assert schema['x-schema-version'] == SCHEMA_VERSION
    assert SCHEMA_VERSION in schema['$id']
    assert 'CLI' in schema['$defs']
