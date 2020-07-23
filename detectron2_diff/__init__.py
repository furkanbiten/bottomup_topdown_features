from .config.defaults import add_detectron2_diff_config
from .modeling.backbone.resnet import BasicStemCaffeeMaxPool, build_resnet_backbone_caffe_maxpool
from .modeling.proposal_generator.rpn import StandardRPNHeadForVGWithHiddenDim
from .modeling.roi_heads.roi_heads import Res5ROIHeadsForVGStride