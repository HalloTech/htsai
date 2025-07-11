# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from typing import Dict, List, Optional
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import ImageList, Instances

from .. import (
    build_densepose_data_filter,
    build_densepose_embedder,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
)

# Register Decoder if not already registered
if "Decoder" not in globals():
    class Decoder(nn.Module):
        def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
            super(Decoder, self).__init__()
            self.in_features      = in_features
            feature_strides       = {k: v.stride for k, v in input_shape.items()}
            feature_channels      = {k: v.channels for k, v in input_shape.items()}
            num_classes           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES
            conv_dims             = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
            self.common_stride    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
            norm                  = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM

            self.scale_heads = []
            for in_feature in self.in_features:
                head_ops = []
                head_length = max(
                    1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
                )
                for k in range(head_length):
                    conv = Conv2d(
                        feature_channels[in_feature] if k == 0 else conv_dims,
                        conv_dims,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=not norm,
                        norm=get_norm(norm, conv_dims),
                        activation=F.relu,
                    )
                    weight_init.c2_msra_fill(conv)
                    head_ops.append(conv)
                    if feature_strides[in_feature] != self.common_stride:
                        head_ops.append(
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                        )
                self.scale_heads.append(nn.Sequential(*head_ops))
                self.add_module(in_feature, self.scale_heads[-1])
            self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
            weight_init.c2_msra_fill(self.predictor)

        def forward(self, features: List[torch.Tensor]):
            for i, _ in enumerate(self.in_features):
                if i == 0:
                    x = self.scale_heads[i](features[i])
                else:
                    x = x + self.scale_heads[i](features[i])
            x = self.predictor(x)
            return x

# Register DensePoseROIHeads if not already registered
class DensePoseROIHeads(StandardROIHeads):
        def __init__(self, cfg, input_shape):
            super().__init__(cfg, input_shape)
            self._init_densepose_head(cfg, input_shape)

        def _init_densepose_head(self, cfg, input_shape):
            self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
            if not self.densepose_on:
                return
            self.densepose_data_filter = build_densepose_data_filter(cfg)
            dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
            dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
            dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
            self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON

            if self.use_decoder:
                dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
            else:
                dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
            in_channels = [input_shape[f].channels for f in self.in_features][0]

            if self.use_decoder:
                self.decoder = Decoder(cfg, input_shape, self.in_features)

            self.densepose_pooler = ROIPooler(
                output_size=dp_pooler_resolution,
                scales=dp_pooler_scales,
                sampling_ratio=dp_pooler_sampling_ratio,
                pooler_type=dp_pooler_type,
            )
            self.densepose_head = build_densepose_head(cfg, in_channels)
            self.densepose_predictor = build_densepose_predictor(
                cfg, self.densepose_head.n_out_channels
            )
            self.densepose_losses = build_densepose_losses(cfg)
            self.embedder = build_densepose_embedder(cfg)

        def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
            if not self.densepose_on:
                return {} if self.training else instances

            features_list = [features[f] for f in self.in_features]
            if self.training:
                proposals, _ = select_foreground_proposals(instances, self.num_classes)
                features_list, proposals = self.densepose_data_filter(features_list, proposals)
                if len(proposals) > 0:
                    proposal_boxes = [x.proposal_boxes for x in proposals]

                    if self.use_decoder:
                        features_list = [self.decoder(features_list)]

                    features_dp = self.densepose_pooler(features_list, proposal_boxes)
                    densepose_head_outputs = self.densepose_head(features_dp)
                    densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
                    densepose_loss_dict = self.densepose_losses(
                        proposals, densepose_predictor_outputs, embedder=self.embedder
                    )
                    return densepose_loss_dict
            else:
                pred_boxes = [x.pred_boxes for x in instances]

                if self.use_decoder:
                    features_list = [self.decoder(features_list)]

                features_dp = self.densepose_pooler(features_list, pred_boxes)
                if len(features_dp) > 0:
                    densepose_head_outputs = self.densepose_head(features_dp)
                    densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
                else:
                    densepose_predictor_outputs = None

                densepose_inference(densepose_predictor_outputs, instances)
                return instances

        def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
        ):
            instances, losses = super().forward(images, features, proposals, targets)
            del targets, images
            if self.training:
                losses.update(self._forward_densepose(features, instances))
            return instances, losses

        def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Instances]
        ):
            instances = super().forward_with_given_boxes(features, instances)
            instances = self._forward_densepose(features, instances)
            return instances
        
        
if "DensePoseROIHeads" not in ROI_HEADS_REGISTRY._obj_map:
    ROI_HEADS_REGISTRY.register(DensePoseROIHeads)
