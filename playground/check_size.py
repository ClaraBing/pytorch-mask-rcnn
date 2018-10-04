import os
import csv
import json
from datetime import datetime
from PIL import Image
from tqdm import tqdm


annot_base_dir = '/vision/u/bingbin/EPIC_KITCHENS_2018/annotations/'
obj_base_dir = '/vision/u/bingbin/EPIC_KITCHENS_2018/object_detection_images/'
epic_frame_format = '{:010d}.jpg'

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

fin = os.path.join(annot_base_dir, 'EPIC_train_object_labels.csv')
fout = os.path.join(annot_base_dir, 'coco_train_object_labels_exists.json')

with open(fin, 'r') as handle:
    data = [line for line in csv.reader(handle)]
header = data[0]
data = data[1:]  # ['20', 'bag', 'P01', 'P01_01', '056581', '[(76, 1260, 462, 186)]']
print('data: type:{} / len:{}'.format(type(data), len(data)))

x = []
for [noun_cls, noun, pid, vid, frame, bboxes] in tqdm(data):
    full_path = get_full_path('train', pid, vid, frame)
    if not os.path.exists(full_path):
      continue
    im = Image.open(full_path)
    if im.size != (1920, 1080):
        print(full_path)
        x.append((full_path, im.size))
import pickle
with open('shape.pkl', 'wb') as f:
    pickle.dump(x, f)

    # width, height = im.size
    # this is numpy array shape
    # if im.shape != (1080, 1920, 3):
    #     print(full_path)