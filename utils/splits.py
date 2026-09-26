"""Where a config's split files are: one rule for every train, test and visualize script.

Split files list frames (e.g. ``camera/frame_000004.png``) relative to ``Dataset.dataset_root``.
All keys below are optional, so configs that only name ``train_split`` and ``val_split`` keep
working unchanged:

    "Dataset": {
        "train_split": "/data/zod/train.txt",
        "val_split": "/data/zod/validation.txt",
        "split_dir": "/data/zod",                     # default: the directory of val_split
        "test_splits": {"day_fair": "test_day_fair.txt", "snow": "test_snow.txt"},
                                                      # default: the five weather splits below
        "visualization_split": "visualizations.txt"   # default: visualizations.txt or
    }                                                 #          visualization.txt in split_dir

Relative file names in ``test_splits`` and ``visualization_split`` are resolved against
``split_dir``; absolute paths are used as they are.
"""
import os

# The test sets of ZOD, Waymo and iseAuto, used when a config names none
WEATHER_TEST_SPLITS = {
    'day_fair': 'test_day_fair.txt',
    'day_rain': 'test_day_rain.txt',
    'night_fair': 'test_night_fair.txt',
    'night_rain': 'test_night_rain.txt',
    'snow': 'test_snow.txt',
}
VISUALIZATION_SPLITS = ('visualizations.txt', 'visualization.txt')


def split_dir(config):
    """Directory the test and visualization split files are in."""
    dataset = config['Dataset']
    return dataset.get('split_dir') or os.path.dirname(dataset['val_split'])


def _resolve(config, name):
    return name if os.path.isabs(name) else os.path.join(split_dir(config), name)


def test_splits(config):
    """The test sets to evaluate, as ``{name: path}``.

    Configured test splits must all exist. Of the default weather splits, missing ones are
    skipped, since not every dataset has every condition.
    """
    configured = config['Dataset'].get('test_splits')
    splits = {name: _resolve(config, path) for name, path in (configured or WEATHER_TEST_SPLITS).items()}
    missing = {name: path for name, path in splits.items() if not os.path.exists(path)}
    if configured and missing:
        raise FileNotFoundError(f"Dataset.test_splits: files not found: {missing}")
    for name, path in missing.items():
        print(f"Skipping test split {name}: file not found ({path})")
    return {name: path for name, path in splits.items() if name not in missing}


def visualization_split(config):
    """Split file listing the frames to visualize."""
    configured = config['Dataset'].get('visualization_split')
    if configured:
        return _resolve(config, configured)
    candidates = [_resolve(config, name) for name in VISUALIZATION_SPLITS]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])
