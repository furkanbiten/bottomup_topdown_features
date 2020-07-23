# import some common libraries
import numpy as np
import cv2
import torch
import os
import io
import argparse
import pickle
import json
import tqdm

# import detectron2
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2_diff.utils.visualizer import Visualizer
# from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2_diff.config.defaults import add_detectron2_diff_config, add_detectron2_diff_config_att

# Show the image in ipynb
from IPython.display import clear_output, Image, display
import PIL.Image
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def get_images(img_list, ix, batch_size):
    raw_images, fnames = [], []
    for fname in img_list[ix:ix + batch_size]:
        # if os.path.splitext(fname)[-1] == '.jpg':
        im = cv2.imread("data/images/" + fname)
        raw_images.append(im)
        fnames.append(fname)
    return raw_images, fnames


def preprocess(raw_images, predictor):
    # Preprocessing
    inputs = []
    for raw_image in raw_images:
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs.append({"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
    return inputs

def extract(args, predictor):

    img_list = os.listdir(args.image_dir)
    for ix in tqdm.tqdm(range(0, len(img_list), args.batch_size)):
        raw_images, fnames = get_images(img_list, ix, args.batch_size)

        with torch.no_grad():

            inputs = preprocess(raw_images, predictor)
            images = predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = predictor.model.proposal_generator(images, features, None)

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in predictor.model.roi_heads.in_features]
            box_features = predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1

            # Predict classes and boxes for each proposal.
            # pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
            pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
            rcnn_outputs = FastRCNNOutputs(
                predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                predictor.model.roi_heads.smooth_l1_beta,
            )

            probs_list = rcnn_outputs.predict_probs()
            boxes_list = rcnn_outputs.predict_boxes()

            attr_prob = pred_attr_logits[..., :-1].softmax(-1)
            max_attr_prob, max_attr_label = attr_prob.max(-1)
            # Fixed-number NMS
            instances_list, ids_list = [], []
            for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
                for nms_thresh in np.arange(args.nms_threshold, 1.0, 0.1):
                    instances, ids = fast_rcnn_inference_single_image(
                        boxes, probs, image_size,
                        score_thresh=args.conf_threshold, nms_thresh=nms_thresh, topk_per_image=args.num_obj
                    )
                    if len(ids) == args.num_obj:
                        break

                instances_list.append(instances)
                ids_list.append(ids)
            #         instances_list, ids_list = rcnn_outputs.inference(
            #             score_thresh=0.2, nms_thresh=0.6, topk_per_image=36
            #         )

            # Post processing for features
            # features_list = feature_pooled.split(
            #     rcnn_outputs.num_preds_per_image)  # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
            # roi_features_list = []
            # for ids, features in zip(ids_list, features_list):
            #     roi_features_list.append(features[ids].detach())

            max_attr_prob = max_attr_prob.split(rcnn_outputs.num_preds_per_image)
            max_attr_label = max_attr_label.split(rcnn_outputs.num_preds_per_image)
            box_features = box_features.split(rcnn_outputs.num_preds_per_image, dim=0)

            for ix, ids in enumerate(ids_list):
                # roi_features_list.append(features_list[ix][ids].detach())
                instances_list[ix].attr_scores = max_attr_prob[ix][ids].detach()
                instances_list[ix].attr_classes = max_attr_label[ix][ids].detach()
                # Pooled features
                instances_list[ix].roi_features = box_features[ix][ids].mean(dim=[2, 3]).detach()
                # Non-pooled features: 14x14x2048
                # instances_list[ix].roi_features = box_features[ix][ids].detach()

            # Post processing for bounding boxes (rescale to raw_image)
            raw_instances_list = []
            for instances, input_per_image, image_size in zip(instances_list, inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                raw_instances = detector_postprocess(instances, height, width)
                raw_instances_list.append(raw_instances)

        for ix, fname in enumerate(fnames):
            with open(os.path.join(args.out_path, fname.split('.')[0]+'.pkl'), 'wb') as f:
                ins_numpy = convert_to_numpy(raw_instances_list[ix].__dict__)
                pickle.dump(ins_numpy, f, protocol=pickle.HIGHEST_PROTOCOL)
        # return raw_instances_list, roi_features_list
        # return raw_instances_list

def convert_to_numpy(ins):
    dump_dict = {'_image_size': ins['_image_size']}
    for k, v in ins['_fields'].items():
        if k == 'pred_boxes':
            dump_dict[k] = v.tensor.cpu().numpy()
        else:
            dump_dict[k] = v.cpu().numpy()
    return dump_dict


def build_predictor(args):
    cfg = get_cfg()
    # add_detectron2_diff_config(cfg)
    add_detectron2_diff_config_att(cfg)
    cfg.merge_from_file(args.model_config)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.WEIGHTS = args.model_weight
    predictor = DefaultPredictor(cfg)

    return predictor

def get_vg_class_att():
    # Load VG Classes and Attributess
    data_path = 'demo/data/genome/1600-400-20'
    vg_classes = []
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())

    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())

    return vg_classes, vg_attrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default='./data/images', help='The directory where the images stored.')
    parser.add_argument('--out_path', type=str, default='./out', help='The path to save the features.')
    parser.add_argument('--model_config', type=str,
                        default="./configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml",
                        help='configuration of the model, the options are inside the path configs/VG-Detection.')
    parser.add_argument('--model_weight', type=str, default="./model/faster_rcnn_from_caffe_attr.pkl",
                        help='path for model weights, check the pretrained weights part in the README.')

    parser.add_argument('--batch_size', type=int, default=2, help='Num of batch to process images.')
    parser.add_argument('--num_obj', type=int, default=36, help='Num of objects to extract per image.')
    parser.add_argument('--conf_threshold', type=float, default=0.2,
                        help='Only return detections with a confidence score exceeding this threshold.')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='The threshold to use for box non-maximum suppression. Value in [0, 1].')

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    vg_classes, vg_attrs = get_vg_class_att()
    MetadataCatalog.get("vg").thing_classes = vg_classes
    MetadataCatalog.get("vg").attr_classes = vg_attrs

    predictor = build_predictor(args)

    torch.cuda.empty_cache()
    extract(args, predictor)
    # instances_list = extract(args, predictor)

    # print("Extracted features for %d images." % len(instances_list))
    # for instances, im in zip(instances_list, ims):
    #     # print(instances)
    #     pred = instances.to('cpu')
    #     v = Visualizer(im[:, :, :], MetadataCatalog.get("vg"), scale=1.5)
    #     v = v.draw_instance_predictions(pred)
    #     showarray(v.get_image()[:, :, ::-1])