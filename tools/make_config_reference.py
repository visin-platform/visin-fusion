#!/usr/bin/env python3
"""Write docs/reference/config.md from the config schema (visin_fusion/config/config_schema.py).

python tools/make_config_reference.py            # write it
python tools/make_config_reference.py --check    # fail if it is out of date (CI)
"""

import argparse
import os
import sys
import types
import typing
from pathlib import Path

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from visin_fusion.config.config_schema import (
    BACKBONES,
    CLI,
    SCHEMA_VERSION,
    Config,
    Dataset,
    DatasetClass,
    General,
    Log,
    TrainClass,
    Transforms,
)

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "reference", "config.md")
SECTIONS = [
    ("Top level", Config, "The sections of a config and its name."),
    ("CLI", CLI, "The model and its inputs."),
    ("General", General, "Training settings."),
    ("Log", Log, "Where a run writes."),
    (
        "Dataset",
        Dataset,
        "The data, and the classes to learn. With a `dataset.json` at `dataset_root`, "
        "most of this comes from the manifest.",
    ),
    ("Dataset.transforms", Transforms, "Input size, augmentation and normalization."),
    ("Dataset.dataset_classes[]", DatasetClass, "The dataset's label values."),
    ("Dataset.train_classes[]", TrainClass, "The classes the model learns."),
]


def type_name(annotation):
    """A type as a short phrase; never a '|', which would split a table cell."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        present = [a for a in args if a is not type(None)]
        return " or ".join(type_name(a) for a in present) + (" (optional)" if len(present) < len(args) else "")
    if origin is typing.Literal:
        return ", ".join(f"`{value}`" for value in args)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "section"
    if origin is not None:
        return f"{origin.__name__} of {', '.join(type_name(a) for a in args)}"
    return getattr(annotation, "__name__", str(annotation))


def default(field):
    if field.is_required():
        return "**required**"
    value = field.default
    if value is PydanticUndefined or value is None:
        return ""
    return f"`{value!r}`".replace("'", '"') if value != "" else '`""`'


def render():
    lines = [
        "# Config reference",
        "",
        "Generated from `visin_fusion/config/config_schema.py` by `tools/make_config_reference.py`; do not edit.",
        f"Schema version: `{SCHEMA_VERSION}`. Export JSON Schema with `visin-fusion schema`.",
        "",
        "Unknown keys in these sections are errors. Model sections (`CLFT`, `CLFTv2`, `MaskFormer`, "
        "`Mask2Former`, `DeepLabV3Plus`) take the settings of their model; start from its preset in "
        "`visin_fusion/config/presets/`.",
        "",
    ]
    lines += ["| `CLI.backbone` | Model section | Modes |", "| --- | --- | --- |"]
    lines += [
        f"| `{b}` | `{section}` | {', '.join(f'`{m}`' for m in modes)} |" for b, (section, modes) in BACKBONES.items()
    ]
    for title, model, intro in SECTIONS:
        lines += [
            "",
            f"## {title}",
            "",
            intro,
            "",
            "| Key | Type | Default | Description |",
            "| --- | --- | --- | --- |",
        ]
        for name, field in model.model_fields.items():
            annotation = typing.get_type_hints(model).get(name, field.annotation)
            lines.append(f"| `{name}` | {type_name(annotation)} | {default(field)} | {field.description or ''} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = Path(OUTPUT).read_text() if os.path.exists(OUTPUT) else ""
        if current != text:
            sys.exit(f"{OUTPUT} is out of date: run python tools/make_config_reference.py")
        return
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write(text)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
