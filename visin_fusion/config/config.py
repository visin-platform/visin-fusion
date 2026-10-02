"""Loading a run's config: every stage script and `visin-fusion run` go through here.

A config may build on another with ``extends``: a preset name from visin_fusion/config/presets/ (``"clftv2"``)
or a path relative to the config (``"./base.json"``). The config's own keys are merged over the
one it extends, dictionaries key by key, everything else replaced:

    {"extends": "clftv2",
     "CLI": {"mode": "rgb"},
     "Dataset": {"dataset_root": "/data/zod"},
     "General": {"epochs": 50}}

Then the dataset's manifest fills in what the dataset describes (visin_fusion/config/dataset_manifest.py),
``CLI.mode`` "fusion" or "cross_fusion" becomes the word the model uses, the log directory defaults
to logs/<dataset>/<backbone>-<mode>, and the schema checks the result (visin_fusion/config/config_schema.py) and
fills in its defaults for keys still missing.
"""

import copy
import importlib
import json
import os

from pydantic import BaseModel

from visin_fusion.config.config_schema import BACKBONES, Config, ConfigError, validate_config
from visin_fusion.config.dataset_manifest import apply_manifest, load_manifest
from visin_fusion.data.resolvers import resolve_root

PRESETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


def merge(base, override):
    """``override`` over ``base``: dictionaries merged key by key, other values replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _extended_path(name, base_dir):
    if name.endswith(".json"):
        return os.path.normpath(os.path.join(base_dir, name))
    path = os.path.join(PRESETS, f"{name}.json")
    if not os.path.exists(path):
        known = sorted(f[:-5] for f in os.listdir(PRESETS) if f.endswith(".json"))
        raise ValueError(f"extends: no preset {name!r}; presets: {known}")
    return path


def resolve_extends(config, base_dir=".", _seen=()):
    """The config with everything it extends merged in, and no ``extends`` key left."""
    name = config.get("extends")
    if not name:
        return config
    path = _extended_path(name, base_dir)
    if path in _seen:
        raise ValueError(f"extends: {path} extends itself through {list(_seen)}")
    with open(path) as f:
        base = resolve_extends(json.load(f), os.path.dirname(path), (*_seen, path))
    own = {key: value for key, value in config.items() if key != "extends"}
    return merge(base, own)


def _expand_dataset_root(config):
    """Environment variables and ~ in Dataset.dataset_root, e.g. "$DATA_ROOT/zod" (the image sets
    DATA_ROOT=/data), so one config works on every machine; "visin:zod" is fetched from Visin."""
    dataset = config.get("Dataset", {})
    root = dataset.get("dataset_root")
    if not root:
        return
    expanded = os.path.expanduser(os.path.expandvars(root))
    if "$" in expanded:
        raise ValueError(f"Dataset.dataset_root {root!r}: environment variable not set ({expanded})")
    # "visin:<name or id>": downloaded from Visin the first time, then the local copy
    dataset["dataset_root"] = resolve_root(expanded)


def _normalize_mode(config):
    """'fusion' and 'cross_fusion' name the same mode: accept either, give each model its own word."""
    cli = config.get("CLI", {})
    if cli.get("mode") in ("fusion", "cross_fusion") and cli.get("backbone") in BACKBONES:
        cli["mode"] = next(m for m in BACKBONES[cli["backbone"]][1] if m in ("fusion", "cross_fusion"))


def _default_logdir(config):
    log = config.setdefault("Log", {})
    if not log.get("logdir"):
        dataset, cli = config.get("Dataset", {}), config.get("CLI", {})
        log["logdir"] = os.path.join(
            "logs", dataset.get("name", "dataset"), f"{cli.get('backbone', 'model')}-{cli.get('mode', 'mode')}"
        )


def _fill_defaults(config, model):
    """Add the schema's defaults for keys the config leaves out (never None: absent stays absent)."""
    for name in type(model).model_fields:
        value = getattr(model, name)
        if name not in config:
            if value is not None:
                config[name] = value.model_dump() if isinstance(value, BaseModel) else copy.deepcopy(value)
        elif isinstance(value, BaseModel) and isinstance(config[name], dict):
            _fill_defaults(config[name], value)
        elif isinstance(value, list) and isinstance(config[name], list):
            for item, item_config in zip(value, config[name], strict=True):
                if isinstance(item, BaseModel) and isinstance(item_config, dict):
                    _fill_defaults(item_config, item)


def _import_plugins(config):
    """Import the modules named in ``plugins``, so models they register are known before the config is checked."""
    for name in config.get("plugins") or []:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise ValueError(f"plugins: cannot import {name!r}: {exc}") from exc


def prepare_config(config, base_dir="."):
    """A config ready for a run: extends resolved, manifest applied, default log directory,
    checked against the schema, schema defaults filled in."""
    config = resolve_extends(config, base_dir)
    _import_plugins(config)
    _expand_dataset_root(config)
    config = apply_manifest(config)
    _normalize_mode(config)
    _default_logdir(config)
    try:
        validate_config(config)
    except ConfigError as e:
        root = config.get("Dataset", {}).get("dataset_root")
        if root and "Dataset.name" in str(e) and load_manifest(root) is None:
            raise ConfigError(
                f"{e}\n  (no dataset.json in {root}: name the dataset's splits and classes in the "
                f"config, or write one with tools/make_manifest.py)"
            ) from None
        raise
    _fill_defaults(config, Config.model_validate(config))
    return config


def load_config(path):
    """The config at ``path``, prepared for a run."""
    with open(path) as f:
        return prepare_config(json.load(f), os.path.dirname(os.path.abspath(path)))
