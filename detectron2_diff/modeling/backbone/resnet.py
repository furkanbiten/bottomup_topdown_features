#https://github.com/airsplay/py-bottom-up-attention/commit/c053e580da10da7e6639d3b26d2cc5b58207877a#diff-69978e00fc9e5d13821dd5d741cd8716

import torch.nn.functional as F
from detectron2.layers import FrozenBatchNorm2d
from detectron2.modeling import BACKBONE_REGISTRY, ResNet, ResNetBlockBase, make_stage  #as in trident_backbone.py in detectron2
from detectron2.modeling.backbone.resnet import BasicStem, BottleneckBlock, DeformBottleneckBlock

__all__ = ["BasicStemCaffeeMaxPool", "build_resnet_backbone_caffe_maxpool"]


class BasicStemCaffeeMaxPool(BasicStem):
    def __init__(self, caffe_maxpool=False, **kwargs):
        super().__init__(**kwargs)
        self.caffe_maxpool = caffe_maxpool
        if self.caffe_maxpool:
            print("Modifications for VG in ResNet Backbone (modeling/backbone/resnet.py):\n\tUsing pad 0 in stem max_pool instead of pad 1.\n")

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        if self.caffe_maxpool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
        else:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

@BACKBONE_REGISTRY.register()
def build_resnet_backbone_caffe_maxpool(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStemCaffeeMaxPool(  #modified BasicStem -> BasicStemCaffeeMaxPool
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        caffe_maxpool=cfg.MODEL.CAFFE_MAXPOOL, #added
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "bottleneck_channels": bottleneck_channels,
            "out_channels": out_channels,
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation,
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)