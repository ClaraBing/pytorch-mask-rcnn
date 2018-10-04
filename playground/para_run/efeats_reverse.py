import csv
import os
import json
from datetime import datetime
import pdb
from tqdm import tqdm

from maskRCNN.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from maskRCNN import visualize_815 as visualize
import torch
from maskRCNN import coco as cocomask
from maskRCNN import model_feats as modellib
import torch.nn as nn
import torch.utils.data as data
import pickle


#annot_base_dir = '/vision/u/bingbin/EPIC_KITCHENS_2018/annotations/'
annot_base_dir = '/vision/u/cy3/data/EPIC/annotations/'
obj_base_dir = '/vision/u/bingbin/EPIC_KITCHENS_2018/object_detection_images/'
epic_frame_format = '{:010d}.jpg'
EPIC_MODEL_PATH = '/vision2/u/cy3/exp/rcnn/0819/coco20180820T0146/mask_rcnn_coco_0034.pth'
EPIC_ANNO_PATH = '/vision2/u/cy3/data/EPIC/annotations/EPIC_train_action_labels.csv'

EPIC_RGB_DIR = '/vision2/u/bingbin/EPIC_KITCHENS_2018/frames_rgb_flow/rgb'

class InferenceConfig(cocomask.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 352

    # added by BB
    dataset = 'epic'
    visualize = True





def get_full_path(subset, pid, vid, frame):
    return os.path.join(obj_base_dir, subset, pid, vid, epic_frame_format.format(int(frame)))


def noun_categories():
    fcsv = '/vision/u/cy3/data/EPIC/annotations/EPIC_noun_classes.csv'
    data = [line for line in csv.reader(open(fcsv, 'r'))]
    header = data[0]
    data = data[1:]
    cats = []
    for line in data:
        cats.append({
            'id': int(line[0])+1,
            'name': line[1],
            'supercategory': line[1]
        })
    return cats


def parse_list(bboxes_str):
    bboxes = bboxes_str.split('),')
    ret = []
    for bbox in bboxes:
        bbox = bbox.replace('(', '').replace(')', '')
        bbox = bbox.replace('[', '').replace(']', '')
        bbox = bbox.replace(' ', '')
        bbox = [int(each) for each in bbox.split(',') if each]
        if bbox:
            ret += bbox,
    return ret


def to_orn_format(fin, fout, subset, model, start_from=0):
    with open(fin, 'r') as f:
        data = [line for line in csv.reader(f)]
    header = data[0]
    data = data[1:]
    # id,participant_id,video_id,narration,start_timestamp,stop_timestamp,start_frame,stop_frame,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
    data = data[start_from:]

    data = data[::-1]

    for [uid, pid, video_id, _, start_timestamp, stop_timestamp, start_frame, stop_frame, _, verb_class, _, noun_class, _, _] in tqdm(data):
        orn_name = video_id + '_' + start_timestamp + '_' + stop_timestamp
        save_filename = os.path.join(fout, subset, orn_name + '.pkl')
        if os.path.isfile(save_filename):
            continue

        frame_dir = os.path.join(EPIC_RGB_DIR, subset, pid, video_id)
        sampled_frames = []
        frame_id = int(start_frame)
        end_frame = int(stop_frame)
        act_list = []
        score_list = []
        #pdb.set_trace()
        while frame_id <= end_frame:

            frame_name = os.path.join(frame_dir, 'frame_' + epic_frame_format.format(frame_id))
            I = io.imread(frame_name)
            sampled_frames.append(I)
            results = model.detect([I])[0]
            from collections import defaultdict
            dboxes = defaultdict(list)
            dscores = defaultdict(list)

            for cid, bbox, score in zip(results['class_ids'], results['rois'], results['scores']):
                dboxes[cid].append(bbox)
                dscores[cid].append(score)
            num_classes = 353
            llist = []
            slist = []
            for i in range(num_classes):
                if i in dboxes:
                    llist.append(dboxes[i])
                    slist.append(dscores[i])
                else:
                    llist.append([])
                    slist.append([])
            act_list.append(llist)
            score_list.append(slist)
            frame_id += 2
        output_dict = {}
        output_dict['segms'] = act_list
        output_dict['boxes'] = act_list
        pickle.dump(output_dict, open(save_filename, 'wb'))
        pickle.dump(score_list, open(os.path.join(fout, 'scores', orn_name + '_scores.pkl'), 'wb'))
        with open('efeats.log', 'w') as handle:
            handle.write('uid {} filename {} saved!'.format(uid, save_filename))


if __name__ == '__main__':
    config = InferenceConfig()
    config.display()

    # Create model object.
    model = modellib.MaskRCNN(model_dir=EPIC_MODEL_PATH, config=config)
    if config.dataset == 'epic':
        # model finetuned on EPIC
        model.mask.conv5 = nn.Conv2d(256, config.NUM_CLASSES, kernel_size=1, stride=1)
        model.classifier.linear_class = nn.Linear(1024, config.NUM_CLASSES)
        model.classifier.linear_bbox = nn.Linear(1024, config.NUM_CLASSES * 4)

    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    if config.dataset == 'coco':
        print("Loading weights from", COCO_MODEL_PATH)
        model.load_state_dict(torch.load(COCO_MODEL_PATH))
    elif config.dataset == 'epic':
        print("Loading weights from", EPIC_MODEL_PATH)
        model.load_state_dict(torch.load(EPIC_MODEL_PATH))
    else:
        raise NotImplementedError('Currently only support dataset COCO or EPIC; got "{}"'.format(config.dataset))


    fin = os.path.join(annot_base_dir, 'EPIC_train_action_labels.csv')
    fout = '/vision2/u/cy3/data/EPIC/bboxes/raw'
    to_orn_format(fin, fout, 'train', model)