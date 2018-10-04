"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np
import random

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from maskRCNN.pycocotools.coco import COCO
from maskRCNN.pycocotools.cocoeval import COCOeval
from maskRCNN.pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from maskRCNN.config import Config
import maskRCNN.utils as utils
import maskRCNN.model_815 as modellib

import torch
import torch.nn as nn

import pdb


# Root directory of the project
ROOT_DIR = os.getcwd()

DATA_DIR = '/vision/u/bingbin/EPIC_KITCHENS_2018/object_detection_images/'
ANNOT_DIR = '/vision/u/bingbin/EPIC_KITCHENS_2018/annotations'

# Path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")
COCO_MODEL_PATH = '/vision/u/bingbin/mask_rcnn_coco.pth'
EPIC_MODEL_PATH = '/vision/u/cy3/exp/rcnn/batch/coco20180813T2223/mask_rcnn_coco_0023.pth'
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class EPICConfig(Config):
    """Configuration for training on EPIC KITHCHENS
    Derives from the base Config class and overrides values specific
    to the EPIC dataset.
    """
    # Give the configuration a recognizable name
    NAME = "epic"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16
    IMAGES_PER_GPU = 32
    IMAGES_PER_GPU = 64

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 352  # EPIC has 352 nouns

class COCOConfig(Config):
    """Configuration for training on EPIC KITHCHENS
    Derives from the base Config class and overrides values specific
    to the EPIC dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class EPICDataset(utils.Dataset):
    """
    EPIC_train_object_labels (each item is a str):
      noun_class; noun; participant_id; video_id; frame;  bounding_boxes (top, left, h, w)
      20;         bag;  P01;            P01_01;   056581; [(76, 1260, 462, 186)]
    
    COCO:
      area;    bbox;              category_id; id;  image_id; iscrowd; segmentation
      2765.15; [200, 200, 78, 71] 58;          156; 558840;   0;       [[...floats...]]
    """
    def load_coco(self, dataset_dir, subset, annot_json, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, test)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """

        image_dir = os.path.join(dataset_dir, subset)

        coco = COCO(annot_json)


        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds()) # TODO: prepare EPIC w/ category_id

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id]))) # TODO: prepare EPIC w/ image_id
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']), # prepare EPIC w/ file name
                width= 1920, # coco.imgs[i]["width"], # TODO: check if EPIC has the same size Ans: No
                height= 1080, # coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(EPICDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1: # TODO: perhaps can't be commented
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            rand_image_id = random.randint(1, len(self.image_info)-1)
            return self.load_mask(rand_image_id)
            #return super(EPICDataset, self).load_mask(image_id) # TODO: commented by BB

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(EPICDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


# def EPICDataLoader(DataLoader):
#   def __init__(self, ):
#     f

############################################################
#  COCO Evaluation
############################################################

#def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
def build_coco_results(dataset, image_ids, rois, class_ids, scores):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            #mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(int(class_id), "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                # Commented out by BB: EPIC doesn't have seg info
                # "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids




    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    #pdb.set_trace()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)

        #pdb.set_trace()

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on re-formatted EPIC KITCHEN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on EPIC KITCHEN")
    parser.add_argument('--dataset', default=DATA_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the EPIC dataset')
    parser.add_argument('--eval_model',
                        metavar="/path/to/coco/",
                        help='path to model for evaluation')
    parser.add_argument('--model_type', required=True,
                        metavar="/path/to/weights.pth",
                        help="Dataset (e.g. 'charades' / 'epic') to train on, which will use diff configurations.")
    parser.add_argument('--pretrained_path', type=str, default='', required=False,
                        help='Path to pretrained weights .pth file **on EPIC**. Empty for training from scratch.')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=5,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--init_with_coco', type=int, required=True, help='whether to init weights from coco pretrained model')
    args = parser.parse_args()

    assert(not (args.init_with_coco and args.pretrained_path)),\
       print('args.pretrained_path should be a pretrained model on EPIC. Conflicting with init_with_coco.')

    print("Command: ", args.command)
    print("Model type: ", args.model_type)
    print("Pretrained weights:", args.pretrained_path)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print('Use COCO to init early layers for EPIC:', args.init_with_coco)



    # Configurations
    if args.command == "train":
      if args.model_type.lower() == 'epic':
        config = EPICConfig()
      elif args.model_type.lower() == 'charades':
        raise NotImplementedError("CharadesConfig not implemented.")
        config = CharadesConfig()
      else:
        raise ValueError('model_type shoudl be "epic" or "charades"')
    else:
        class InferenceConfig(EPICConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()

    config.display()

    # Create model
    if args.command == "train":
        if args.model_type.lower() == 'coco' or args.init_with_coco:
          coco_config = COCOConfig()
          model = modellib.MaskRCNN(config=coco_config,
                                    model_dir=args.logs)
        else:
          model = modellib.MaskRCNN(config=config,
                                    model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model_type.lower() == "coco" or args.init_with_coco:
        model_path = COCO_MODEL_PATH
    elif args.model_type.lower() == "epic":
        model_path = args.eval_model # EPIC_MODEL_PATH
        model.mask.conv5 = nn.Conv2d(256, config.NUM_CLASSES, kernel_size=1, stride=1)
        model.classifier.linear_class = nn.Linear(1024, config.NUM_CLASSES)
        model.classifier.linear_bbox = nn.Linear(1024, config.NUM_CLASSES * 4)
    elif args.model_type.lower() == "last":
        # TODO: untested (BB)
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = config.IMAGENET_MODEL_PATH
    else:
        # strat from pre-trained weights in given path
        model_path = args.pretrained_path

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)


    if args.init_with_coco:
      # retrain the classifier layer
      model.mask.conv5 = nn.Conv2d(256, config.NUM_CLASSES, kernel_size=1, stride=1)
      model.classifier.linear_class = nn.Linear(1024, config.NUM_CLASSES)
      model.classifier.linear_bbox = nn.Linear(1024, config.NUM_CLASSES * 4)


    if config.GPU_COUNT:
        model = model.cuda()

    #train_json = os.path.join(ANNOT_DIR, 'coco_train_object_labels_exists.json')
    # train on full
    CC_ANNOT_DIR = '/vision/u/cy3/data/EPIC/annotations/'
    train_json = os.path.join(CC_ANNOT_DIR, 'coco_train_object_labels_819c.json')

    val_json = train_json

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = EPICDataset()
        dataset_train.load_coco(args.dataset, "train", train_json)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = EPICDataset()
        dataset_val.load_coco(args.dataset, "train", val_json)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        # print("Training network classifiers")
        # model.train_model(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=40,
        #             layers='classifiers')

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        exit()
        # BB: only finetune the heads (see above)
        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train_model(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=120,
        #             layers='4+')

        # # Training - Stage 3
        # # Fine tune all layers
        # print("Fine tune all layers")
        # model.train_model(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=160,
        #             layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = EPICDataset()
        coco = dataset_val.load_coco(args.dataset, "val", val_json, return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
        #evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
