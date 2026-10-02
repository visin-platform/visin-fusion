"""visin-fusion dataset preview: annotations rendered in color, not as near-black class indices."""

import numpy as np
from PIL import Image

from visin_fusion import cli
from visin_fusion.dataset_tools.preview import annotation_colors, render_preview
from visin_fusion.sample import SAMPLE_DIR

CLASSES = [
    {"name": "background", "index": 0, "dataset_mapping": [0], "color": [0, 0, 0]},
    {"name": "vehicle", "index": 1, "dataset_mapping": [2, 3], "color": [128, 0, 128]},
]


def test_every_mapped_value_gets_its_class_color_and_the_rest_gray():
    table = annotation_colors(CLASSES)
    assert table[2].tolist() == table[3].tolist() == [128, 0, 128]
    assert table[1].tolist() == [128, 128, 128]


def test_the_preview_has_three_panels_and_a_legend_naming_unmapped_values():
    camera = np.zeros((40, 80, 3), dtype=np.uint8)
    annotation = np.zeros((40, 80), dtype=np.uint8)
    annotation[:10] = 2
    annotation[10:20] = 1
    preview = render_preview(camera, annotation, CLASSES)
    assert preview.shape[1] == 480 * 3
    assert preview.shape[0] == 240 + 36
    legend = preview[240:]
    assert (legend == np.array([128, 128, 128])).all(axis=-1).any()


def test_the_command_writes_a_preview_per_frame(tmp_path):
    cli.main(["dataset", "preview", "--root", str(SAMPLE_DIR), "--frames", "2", "--output", str(tmp_path)])
    previews = sorted(tmp_path.glob("*_preview.png"))
    assert len(previews) == 2
    assert Image.open(previews[0]).mode == "RGB"


def test_a_mapping_above_255_does_not_break_a_frame_without_it():
    classes = [*CLASSES, {"name": "rare", "index": 2, "dataset_mapping": [300], "color": [1, 2, 3]}]
    preview = render_preview(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint16), classes)
    assert preview.ndim == 3
