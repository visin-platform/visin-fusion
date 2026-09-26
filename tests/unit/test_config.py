"""utils/config.py: extends, presets, defaults and the default log directory."""
import json
from pathlib import Path

import pytest

from utils.config import load_config, merge, prepare_config, resolve_extends
from utils.config_schema import ConfigError

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / 'tests' / 'data' / 'zod_sample'
PRESETS = sorted(p.stem for p in (REPO / 'configs' / 'presets').glob('*.json'))


def test_merge_keeps_what_the_override_leaves_out():
    base = {'General': {'epochs': 200, 'batch_size': 8}, 'tags': ['a'], 'CLI': {'mode': 'rgb'}}
    merged = merge(base, {'General': {'epochs': 5}, 'tags': ['b']})
    assert merged == {'General': {'epochs': 5, 'batch_size': 8}, 'tags': ['b'], 'CLI': {'mode': 'rgb'}}
    assert base['General']['epochs'] == 200  # not changed in place


def test_extends_a_preset():
    config = resolve_extends({'extends': 'swin', 'General': {'epochs': 3}})
    assert config['CLI']['backbone'] == 'swin_fusion'
    assert config['General']['epochs'] == 3 and config['General']['batch_size'] == 8
    assert 'extends' not in config


def test_extends_a_relative_file_that_extends_a_preset(tmp_path):
    (tmp_path / 'base.json').write_text(json.dumps({'extends': 'deeplabv3plus', 'General': {'epochs': 7}}))
    (tmp_path / 'run.json').write_text(json.dumps({'extends': './base.json', 'CLI': {'mode': 'rgb'},
                                                   'Dataset': {'dataset_root': str(SAMPLE)}}))
    config = load_config(tmp_path / 'run.json')
    assert config['CLI'] == {'backbone': 'deeplabv3plus', 'mode': 'rgb'}
    assert config['General']['epochs'] == 7


def test_extends_cycle_is_an_error(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'extends': './b.json'}))
    (tmp_path / 'b.json').write_text(json.dumps({'extends': './a.json'}))
    with pytest.raises(ValueError, match='extends itself'):
        load_config(tmp_path / 'a.json')


def test_unknown_preset_names_the_presets():
    with pytest.raises(ValueError, match="no preset 'swim'"):
        resolve_extends({'extends': 'swim'})


@pytest.mark.parametrize('preset', PRESETS)
def test_every_preset_runs_on_a_dataset_with_a_manifest(preset):
    config = prepare_config({'extends': preset, 'Dataset': {'dataset_root': str(SAMPLE)}})
    assert config['Dataset']['train_classes']  # suggested by the manifest
    assert config['General']['resume_training'] is False  # schema default


def test_schema_defaults_fill_only_missing_keys():
    config = prepare_config({'extends': 'swin', 'Dataset': {'dataset_root': str(SAMPLE)},
                             'General': {'seed': 7}})
    assert config['General']['seed'] == 7
    assert config['General']['model_path'] == ''
    assert 'source_classes' not in config['General']  # optional, no default: stays absent
    assert 'CLFT' not in config  # another model's section is not added


def test_default_logdir_names_dataset_and_model():
    config = prepare_config({'extends': 'swin', 'CLI': {'mode': 'rgb'}, 'Dataset': {'dataset_root': str(SAMPLE)}})
    assert config['Log']['logdir'] == 'logs/zod/swin_fusion-rgb'


def test_preparing_twice_changes_nothing():
    once = load_config(REPO / 'configs' / 'quickstart.json')
    assert prepare_config(json.loads(json.dumps(once))) == once


def test_mistake_in_a_user_config_is_reported():
    with pytest.raises(ConfigError, match='General.epcohs'):
        prepare_config({'extends': 'swin', 'Dataset': {'dataset_root': str(SAMPLE)}, 'General': {'epcohs': 3}})


@pytest.mark.parametrize('preset, word', [('swin', 'cross_fusion'), ('deeplabv3plus', 'fusion')])
@pytest.mark.parametrize('mode', ['fusion', 'cross_fusion'])
def test_either_fusion_word_works_for_every_model(preset, word, mode):
    config = prepare_config({'extends': preset, 'CLI': {'mode': mode}, 'Dataset': {'dataset_root': str(SAMPLE)}})
    assert config['CLI']['mode'] == word


def test_dataset_root_uses_environment_variables(monkeypatch):
    monkeypatch.setenv('DATA_ROOT', str(SAMPLE.parent))
    config = prepare_config({'extends': 'swin', 'Dataset': {'dataset_root': '$DATA_ROOT/zod_sample'}})
    assert config['Dataset']['dataset_root'] == str(SAMPLE)
    assert config['Dataset']['name'] == 'zod'  # its manifest was found there


def test_unset_environment_variable_is_named(monkeypatch):
    monkeypatch.delenv('NO_SUCH_ROOT', raising=False)
    with pytest.raises(ValueError, match='NO_SUCH_ROOT'):
        prepare_config({'extends': 'swin', 'Dataset': {'dataset_root': '$NO_SUCH_ROOT/zod'}})
