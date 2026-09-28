"""Small tensor checks for CLFTv2 fusion layers and their supported fusion policies."""

import pytest
import torch
from torch import nn

from visin_fusion.models import clftv2


def test_readout_layers_and_reassembly():
    tokens = torch.tensor([[[2., 4.], [6., 8.], [10., 12.]]])
    assert clftv2.Read_ignore(start_index=1)(tokens).tolist() == [[[6., 8.], [10., 12.]]]
    assert clftv2.Read_add(start_index=1)(tokens).tolist() == [[[8., 12.], [12., 16.]]]
    assert clftv2.Read_add(start_index=2)(tokens).tolist() == [[[14., 18.]]]
    assert clftv2.Read_projection(2)(tokens).shape == (1, 2, 2)

    spatial = torch.randn(1, 2, 2, 3)
    assert clftv2.Resample(4, 2, 3)(spatial).shape == (1, 3, 8, 12)
    lazy = clftv2.Resample(8, None, 3)
    assert lazy(spatial).shape == (1, 3, 16, 24)
    assert lazy.emb_dim == 2
    assert clftv2.SpatialReassemble('ignore', 4, 2, 3)(spatial).shape == (1, 3, 8, 12)
    with pytest.raises(AssertionError):
        clftv2.Resample(3, 2, 3)


def test_residual_unit_keeps_shape_and_supports_gradients():
    x = torch.randn(1, 4, 3, 3, requires_grad=True)
    result = clftv2.ResidualConvUnit(4)(x * 1)
    result.sum().backward()
    assert result.shape == x.shape
    assert x.grad is not None


def test_cross_attention_preserves_small_and_large_resolution(monkeypatch):
    attention = clftv2.CrossAttention(4)
    small = torch.randn(1, 4, 3, 5)
    assert attention(small, small).shape == small.shape

    class IdentityAttention(nn.Module):
        def forward(self, query, key, value):
            assert query.shape[1] == 64 * 64
            return query, None

    monkeypatch.setattr(attention, 'multihead_attn', IdentityAttention())
    large = torch.randn(1, 4, 65, 66)
    assert attention(large, large).shape == large.shape


@pytest.mark.parametrize('strategy', [
    'cross_attention', 'gmf', 'spatial_fusion', 'gated_average',
    'ln_gated_average', 'sigmoid_gated_average', 'residual_average',
    'simple_average', 'adder_fusion', 'simple_average_w_prev', 'resconv_average',
])
def test_fusion_strategies_accept_both_modalities(strategy):
    fusion = clftv2.Fusion(4, strategy).eval()
    rgb, lidar = torch.randn(1, 4, 2, 3), torch.randn(1, 4, 2, 3)
    previous = torch.randn(1, 4, 1, 2)
    with torch.no_grad():
        result = fusion(rgb, lidar, previous_stage=previous, modal='cross_fusion')
    assert result.shape == rgb.shape
    assert torch.isfinite(result).all()


def test_fusion_modal_and_average_semantics():
    fusion = clftv2.Fusion(4, 'simple_average')
    fusion.res_conv_rgb = nn.Identity()
    fusion.res_conv_xyz = nn.Identity()
    fusion.res_conv2 = nn.Identity()
    rgb, lidar = torch.ones(1, 4, 2, 2), torch.full((1, 4, 2, 2), 3.)
    prev = torch.full((1, 4, 1, 1), 5.)
    assert torch.equal(fusion(rgb, lidar, prev, 'rgb'), rgb + 5)
    assert torch.equal(fusion(rgb, lidar, prev, 'lidar'), lidar + 5)
    assert torch.equal(fusion(rgb, lidar, prev, 'cross_fusion'), torch.full_like(rgb, 2.))
    fusion.fusion_strategy = 'simple_average_w_prev'
    assert torch.equal(fusion(rgb, lidar, prev, 'cross_fusion'), torch.full_like(rgb, 9.))
    fusion.fusion_strategy = 'missing'
    with pytest.raises(ValueError, match='Unknown fusion strategy'):
        fusion(rgb, lidar, modal='cross_fusion')
    with pytest.raises(ValueError, match='fusion_strategy'):
        clftv2.Fusion(4)


@pytest.mark.parametrize('model_type', ['segmentation', 'depth', 'full'])
def test_clftv2_heads_and_spatial_backbone_formats(monkeypatch, model_type):
    class Backbone(nn.Module):
        def forward(self, image):
            # No feature_info: use the explicitly supplied embedding dimensions.
            return [nn.functional.adaptive_avg_pool2d(image.mean(dim=1, keepdim=True), (2, 2)).expand(-1, 4, -1, -1),
                    nn.functional.adaptive_avg_pool2d(image.mean(dim=1, keepdim=True), (1, 1)).expand(-1, 4, -1, -1).permute(0, 2, 3, 1)]

    monkeypatch.setattr(clftv2.timm, 'create_model', lambda *args, **kwargs: Backbone())
    model = clftv2.CLFTv2Network(emb_dims=[4, 4], resample_dim=4,
                                       reassemble_s=[4, 8], nclasses=3,
                                       type=model_type, pretrained=False,
                                       fusion_strategy='simple_average').eval()
    rgb = torch.randn(1, 3, 8, 10)
    with torch.no_grad():
        depth, segmentation = model(rgb, rgb, modal='cross_fusion')
    assert (depth is not None) == (model_type in ('depth', 'full'))
    assert (segmentation is not None) == (model_type in ('segmentation', 'full'))
    if depth is not None:
        assert depth.shape == (1, 1, 8, 10)
    if segmentation is not None:
        assert segmentation.shape == (1, 3, 8, 10)
    with pytest.raises(ValueError, match='Invalid modal'):
        model(rgb, rgb, modal='radar')
    with pytest.raises(ValueError, match='emb_dims'):
        clftv2.CLFTv2Network(emb_dims=None, pretrained=False,
                                   fusion_strategy='simple_average')
