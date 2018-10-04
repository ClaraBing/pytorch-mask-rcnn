import csv
import os
import json
from datetime import datetime
import pdb
from tqdm import tqdm

import numpy as np
import skimage.io as io
import pickle


#annot_base_dir = '/vision/u/bingbin/EPIC_KITCHENS_2018/annotations/'
annot_base_dir = '/vision/u/cy3/data/EPIC/annotations/'
obj_base_dir = '/vision/u/bingbin/EPIC_KITCHENS_2018/object_detection_images/'
epic_frame_format = '{:010d}.jpg'
EPIC_MODEL_PATH = '/vision2/u/cy3/exp/rcnn/0819/coco20180820T0146/mask_rcnn_coco_0034.pth'
EPIC_ANNO_PATH = '/vision2/u/cy3/data/EPIC/annotations/EPIC_train_action_labels.csv'

EPIC_RGB_DIR = '/vision2/u/bingbin/EPIC_KITCHENS_2018/frames_rgb_flow/rgb'


def to_orn_format(fin, fout, subset, start_from=0):
    with open(fin, 'r') as f:
        data = [line for line in csv.reader(f)]
    header = data[0]
    data = data[1:]
    # id,participant_id,video_id,narration,start_timestamp,stop_timestamp,start_frame,stop_frame,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
    data = data[start_from:]
    n_frames = 0
    for [uid, pid, video_id, _, start_timestamp, stop_timestamp, start_frame, stop_frame, _, verb_class, _, noun_class, _, _] in data:
        n_frames += int(stop_frame) - int(start_frame)
        #if uid == '39':
        #    print('frames in first 40 clips is : {}'.format(n_frames))
        #    break
        #else: 
        #    continue
    print('taotal is {}'.format(n_frames))

if __name__ == '__main__':
    '''
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

    '''
    fin = os.path.join(annot_base_dir, 'EPIC_train_action_labels.csv')
    fout = '/vision2/u/cy3/data/EPIC/bboxes/raw'
    to_orn_format(fin, fout, 'train')
