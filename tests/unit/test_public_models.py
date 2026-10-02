"""visin_fusion.models: the public classes check their inputs and accept either spelling of the fusion mode."""

import pytest
import torch

from visin_fusion.models import CLFTv2, DeepLabV3Plus, MaskFormerFusion

IMAGE = torch.randn(1, 3, 256, 256)


class ProjectConfig:
    """An object a weights-only load refuses."""


def small(cls=CLFTv2, **kwargs):
    return cls(num_classes=4, pretrained=False, **kwargs).eval()


@pytest.mark.parametrize("cls", [CLFTv2, MaskFormerFusion, DeepLabV3Plus])
@pytest.mark.parametrize("mode", ["rgb", "lidar"])
def test_a_single_stream_model_needs_only_its_own_input(cls, mode):
    model = small(cls, mode=mode)
    own = {"rgb": IMAGE} if mode == "rgb" else {"lidar": IMAGE}
    with torch.inference_mode():
        assert model(**own).shape == (1, 4, 256, 256)


def test_a_missing_input_is_named():
    with pytest.raises(ValueError, match="needs rgb and lidar; lidar is None"):
        small(mode="cross_fusion")(IMAGE)
    with pytest.raises(ValueError, match="needs rgb; rgb is None"):
        small(mode="rgb")(lidar=IMAGE)


def test_inputs_must_be_three_channel_batches_of_equal_shape():
    model = small(mode="cross_fusion")
    with pytest.raises(ValueError, match=r"rgb must be shaped \[batch, 3, height, width\]"):
        model(IMAGE[0], IMAGE)
    with pytest.raises(ValueError, match="same shape"):
        model(IMAGE, torch.randn(1, 3, 128, 128))


@pytest.mark.parametrize(
    ("cls", "own", "other"), [(DeepLabV3Plus, "fusion", "cross_fusion"), (CLFTv2, "cross_fusion", "fusion")]
)
def test_either_spelling_of_the_fusion_mode_is_accepted(cls, own, other):
    assert small(cls, mode=other).mode == own


def test_an_unknown_mode_lists_the_choices():
    with pytest.raises(ValueError, match="must be one of"):
        small(mode="radar")


def test_from_pretrained_reads_a_state_dict_and_refuses_pickled_objects(tmp_path):
    model = small(mode="rgb")
    good = tmp_path / "good.pth"
    torch.save({"model_state_dict": model.state_dict()}, good)
    assert CLFTv2.from_pretrained(str(good), num_classes=4, mode="rgb").mode == "rgb"

    bad = tmp_path / "bad.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": ProjectConfig()}, bad)
    with pytest.raises(ValueError, match="holds more than weights"):
        CLFTv2.from_pretrained(str(bad), num_classes=4, mode="rgb")


def test_clft_rejects_an_image_size_its_backbone_cannot_take():
    from visin_fusion.models import CLFT

    with pytest.raises(ValueError, match=r"takes 384x384 inputs, but image_size is 256"):
        CLFT(num_classes=4, image_size=256, pretrained=False)
