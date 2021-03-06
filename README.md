# Bottom-up Top-down features extracted from Faster RCNN trained on Visual Genome using Detectron2

Disclaimer: 99% of job is done by [airsplay](https://github.com/airsplay/py-bottom-up-attention/) and 
[adrelino](https://github.com/adrelino/py-bottom-up-attention-extracted).

In this repo, you can extract the features from Faster RCNN trained on 
Visual Genome provided originally by [UpDown](https://github.com/peteanderson80/bottom-up-attention).
This model is used in many papers, mostly because it is trained to predict **not only bounding box 
but also its attributes.** 

In this repo, we extend the [adrelino](https://github.com/adrelino/py-bottom-up-attention-extracted) 's repo 
to FasterRCNN with attributes. 
 
## Requirements
```
detectron2
torch >= 1.4.0
opencv-python
tqdm
```

## Virtual Environment
It is highly recommended that you create a virtual environment. I have used miniconda3 with python3.7.7.
Create the environment from the env.yml file:

``conda env create --name <env_name> -f env.yml ``

You will not have detectron inside env.yml. To obtain the same detectron2 I have, please run:
 
```python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'```

If that fails as well, Google is your best friend. 

## Pretrained Weights
Pretrained models thankfully converted from Caffe to Pytorch by [airsplay](https://github.com/airsplay/py-bottom-up-attention/)

[with attribute](https://nlp1.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl),

[without attribute](https://nlp1.cs.unc.edu/models/faster_rcnn_from_caffe.pkl)

[original with attribute](http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl)

More info can be found [here](https://github.com/airsplay/py-bottom-up-attention#feature-extraction-scripts-for-ms-coco).

## Usage
You can use the command line and its options:
````python extract.py --image_dir ... ````

More information is coming...