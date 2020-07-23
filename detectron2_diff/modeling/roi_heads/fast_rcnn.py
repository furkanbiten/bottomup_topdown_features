import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg,
            box_dim=4, use_attr=False, num_attrs=-1):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        self.use_attr = use_attr
        if use_attr:
            print("Modifications for VG in RoI heads (modeling/roi_heads/fast_rcnn.py))\n"
                    f"\tEmbedding: {num_classes + 1} --> {input_size // 8}"
                    f"\tLinear: {input_size + input_size // 8} --> {input_size // 4}"
                    f"\tLinear: {input_size // 4} --> {num_attrs + 1}"
                  )
            print()
            self.cls_embedding = nn.Embedding(num_classes + 1, input_size // 8)
            self.fc_attr = nn.Linear(input_size + input_size // 8, input_size // 4)
            self.attr_score = nn.Linear(input_size // 4, num_attrs + 1)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        if self.use_attr:
            _, max_class = scores.max(-1)      # [b, c] --> [b]
            cls_emb = self.cls_embedding(max_class)   # [b] --> [b, 256]
            x = torch.cat([x, cls_emb], -1)     # [b, 2048] + [b, 256] --> [b, 2304]
            x = self.fc_attr(x)
            x = F.relu(x)
            attr_scores = self.attr_score(x)
            return scores, attr_scores, proposal_deltas
        else:
            return scores, proposal_deltas