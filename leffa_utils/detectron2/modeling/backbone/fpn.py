# Copyright (c) Facebook, Inc. and its affiliates.
import math
import torch
import torch.nn.functional as F
from torch import nn
import fvcore.nn.weight_init as weight_init

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone

__all__ = [
    "FPN",
    "LastLevelMaxPool",
    "LastLevelP6P7",
    "build_resnet_fpn_backbone",
    "build_retinanet_resnet_fpn_backbone",
]


class FPN(Backbone):
    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
    ):
        super().__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        _assert_strides_are_log2_contiguous(strides)

        self.lateral_convs = []
        self.output_convs = []
        use_bias = norm == ""

        for idx, in_channels in enumerate(in_channels_per_feature):
            stage = int(math.log2(strides[idx]))
            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=get_norm(norm, out_channels)
            )
            output_conv = Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=get_norm(norm, out_channels)
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module(f"fpn_lateral{stage}", lateral_conv)
            self.add_module(f"fpn_output{stage}", output_conv)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.lateral_convs = self.lateral_convs[::-1]
        self.output_convs = self.output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        self._out_feature_strides = {f"p{int(math.log2(s))}": s for s in strides}

        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides[f"p{s + 1}"] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        for idx in range(1, len(self.in_features)):
            feat_name = self.in_features[-idx - 1]
            features = bottom_up_features[feat_name]
            top_down = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
            lateral = self.lateral_convs[idx](features)
            prev_features = lateral + top_down
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, self.output_convs[idx](prev_features))

        if self.top_block is not None:
            top_input = bottom_up_features.get(self.top_block.in_feature) or results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_input))

        assert len(results) == len(self._out_features)
        return {f: r for f, r in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    for i in range(1, len(strides)):
        assert strides[i] == 2 * strides[i - 1], f"Strides {strides[i-1]} and {strides[i]} are not log2 contiguous."


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for m in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(m)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_resnet_backbone(cfg, input_shape)
    return FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )


@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    return FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, cfg.MODEL.FPN.OUT_CHANNELS),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
