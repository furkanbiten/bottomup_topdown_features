#https://github.com/airsplay/py-bottom-up-attention/commit/c053e580da10da7e6639d3b26d2cc5b58207877a#diff-7f4b8ab7b2d687039a81de274c1562e5

from typing import Dict, List
from torch import nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import RPN_HEAD_REGISTRY, ProposalNetwork, build_anchor_generator
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead

@RPN_HEAD_REGISTRY.register()
class StandardRPNHeadForVGWithHiddenDim(StandardRPNHead):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(StandardRPNHead, self).__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        if cfg.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS == -1:
            hid_channels = in_channels
        else:
            hid_channels = cfg.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS
            print("Modifications for VG in RPN (modeling/proposal_generator/rpn.py):\n"
                  "\tUse hidden dim %d instead fo the same dim as Res4 (%d).\n" % (hid_channels, in_channels))

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, hid_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(hid_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            hid_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
