#https://github.com/airsplay/py-bottom-up-attention/commit/c053e580da10da7e6639d3b26d2cc5b58207877a#diff-ba16c545429f1c96b5ee0248c1d56469

from detectron2.config import CfgNode as CN

def add_detectron2_diff_config(cfg):
    _C = cfg
    # Caffe Options
    _C.MODEL.BACKBONE.NAME = "build_resnet_backbone_caffe_maxpool"
    _C.MODEL.CAFFE_MAXPOOL = False

    # RPN options
    _C.MODEL.RPN.HEAD_NAME = "StandardRPNHeadForVGWithHiddenDim"
    _C.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS = -1

    # ROI options
    _C.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsForVGStride"  #will be overwritten by Base-RCNN-C4.yaml
    _C.MODEL.ROI_BOX_HEAD.RES5HALVE = True

def add_detectron2_diff_config_att(cfg):
    _C = cfg
    # Caffe Options
    _C.MODEL.BACKBONE.NAME = "build_resnet_backbone_caffe_maxpool"
    _C.MODEL.CAFFE_MAXPOOL = False

    # RPN options
    _C.MODEL.RPN.HEAD_NAME = "StandardRPNHeadForVGWithHiddenDim"
    _C.MODEL.PROPOSAL_GENERATOR.HID_CHANNELS = -1

    # ROI options
    _C.MODEL.ROI_HEADS.NAME = "Res5ROIHeadsForVGStride"  #will be overwritten by Base-RCNN-C4.yaml
    _C.MODEL.ROI_BOX_HEAD.RES5HALVE = True
    _C.MODEL.ROI_BOX_HEAD.ATTR = True
    _C.MODEL.ROI_BOX_HEAD.NUM_ATTRS = 400
    _C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
    _C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
        (10.0, 10.0, 5.0, 5.0),
        (20.0, 20.0, 10.0, 10.0),
        (30.0, 30.0, 15.0, 15.0),
    )
    _C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0