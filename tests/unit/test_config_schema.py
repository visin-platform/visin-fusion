"""utils/config_schema.py: every shipped config is valid, and typical mistakes are caught."""
import copy
import glob
import json
from pathlib import Path

import pytest

from utils.config_schema import Config, ConfigError, validate_config
from utils.config import resolve_extends
from utils.dataset_manifest import apply_manifest

REPO = Path(__file__).resolve().parents[2]
# Complete configs; presets are tested with a dataset in test_config.py
CONFIGS = sorted(p for p in glob.glob(str(REPO / 'configs' / '**' / '*.json'), recursive=True)
                 if '/presets/' not in p)
QUICKSTART = REPO / 'configs' / 'quickstart.json'


def load(path):
    with open(path) as f:
        config = resolve_extends(json.load(f))
    if 'dataset_root' in config.get('Dataset', {}) and not Path(config['Dataset']['dataset_root']).is_absolute():
        config['Dataset']['dataset_root'] = str(REPO / config['Dataset']['dataset_root'])
    return apply_manifest(config)


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: str(Path(p).relative_to(REPO)))
def test_shipped_config_is_valid(path):
    config = load(path)
    validate_config(config)


@pytest.fixture
def config():
    return load(QUICKSTART)


def invalid(config, match):
    with pytest.raises(ConfigError, match=match):
        validate_config(config)


def test_typo_in_a_key(config):
    config['General']['epcohs'] = config['General'].pop('epochs')
    invalid(config, r'General\.epcohs')


def test_unknown_section(config):
    config['Genral'] = {}
    invalid(config, 'Genral')


def test_mode_the_backbone_does_not_have(config):
    config['CLI']['mode'] = 'fusion'  # validate_config alone does not translate; utils.config does
    invalid(config, "mode 'fusion' is not one of swin_fusion's modes")


def test_unknown_mode(config):
    config['CLI']['mode'] = 'thermal'
    invalid(config, r'CLI\.mode')


def test_missing_model_section(config):
    del config['SwinFusion']
    invalid(config, "needs a 'SwinFusion' section")


def test_train_class_gap(config):
    config['Dataset']['train_classes'][-1]['index'] += 1
    invalid(config, 'no gaps or duplicates')


def test_train_class_maps_unknown_dataset_class(config):
    config['Dataset']['train_classes'][1]['dataset_mapping'] = [99]
    invalid(config, r'maps dataset classes \[99\]')


def test_probability_out_of_range(config):
    config['Dataset']['transforms']['p_flip'] = 1.5
    invalid(config, r'transforms\.p_flip')


def test_every_problem_is_listed(config):
    broken = copy.deepcopy(config)
    broken['General']['epochs'] = 0
    broken['General']['batch_size'] = 'eight'
    with pytest.raises(ConfigError) as error:
        validate_config(broken)
    assert 'General.epochs' in str(error.value) and 'General.batch_size' in str(error.value)


def test_json_schema_exports():
    schema = Config.model_json_schema()
    assert set(schema['required']) >= {'CLI', 'General', 'Log', 'Dataset'}


def test_config_reference_is_up_to_date():
    import sys
    sys.path.insert(0, str(REPO / 'tools'))
    from make_config_reference import OUTPUT, render
    assert Path(OUTPUT).read_text() == render(), 'run: python tools/make_config_reference.py'
