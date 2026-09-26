"""utils/splits.py: one rule for where test and visualization split files are."""
import pytest

# splits.test_splits is called through the module: imported bare, pytest would collect it as a test
from utils import splits
from utils.splits import WEATHER_TEST_SPLITS, split_dir, visualization_split


def make_config(tmp_path, **dataset):
    return {'Dataset': {'val_split': str(tmp_path / 'validation.txt'), **dataset}}


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('camera/frame_000001.png\n')
    return path


def test_split_dir_defaults_to_the_val_split_directory(tmp_path):
    assert split_dir(make_config(tmp_path)) == str(tmp_path)


def test_split_dir_can_be_set(tmp_path):
    assert split_dir(make_config(tmp_path, split_dir='/data/splits')) == '/data/splits'


def test_default_weather_splits_skip_missing_files(tmp_path):
    touch(tmp_path / 'test_day_fair.txt')
    touch(tmp_path / 'test_snow.txt')
    assert splits.test_splits(make_config(tmp_path)) == {
        'day_fair': str(tmp_path / 'test_day_fair.txt'),
        'snow': str(tmp_path / 'test_snow.txt'),
    }


def test_all_default_weather_splits(tmp_path):
    for name in WEATHER_TEST_SPLITS.values():
        touch(tmp_path / name)
    assert list(splits.test_splits(make_config(tmp_path))) == list(WEATHER_TEST_SPLITS)


def test_configured_splits_resolve_relative_and_keep_absolute(tmp_path):
    touch(tmp_path / 'test.txt')
    elsewhere = touch(tmp_path / 'other' / 'hard.txt')
    config = make_config(tmp_path, test_splits={'test': 'test.txt', 'hard': str(elsewhere)})
    assert splits.test_splits(config) == {'test': str(tmp_path / 'test.txt'), 'hard': str(elsewhere)}


def test_configured_split_that_is_missing_fails(tmp_path):
    config = make_config(tmp_path, test_splits={'test': 'test.txt'})
    with pytest.raises(FileNotFoundError, match='test.txt'):
        splits.test_splits(config)


def test_visualization_split_finds_either_name(tmp_path):
    touch(tmp_path / 'visualization.txt')
    assert visualization_split(make_config(tmp_path)) == str(tmp_path / 'visualization.txt')


def test_visualization_split_can_be_set(tmp_path):
    config = make_config(tmp_path, split_dir='/data/splits', visualization_split='vis.txt')
    assert visualization_split(config) == '/data/splits/vis.txt'
