import os
import csv
import json
from datetime import datetime
import pdb
from tqdm import tqdm

#annot_base_dir = '/vision/u/bingbin/EPIC_KITCHENS_2018/annotations/'
annot_base_dir = '/vision/u/cy3/data/EPIC/annotations/'
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

def to_COCO_format(fin, fout, subset):
  with open(fin, 'r') as handle:
    data = [line for line in csv.reader(handle)]
  header = data[0]
  data = data[1:] # ['20', 'bag', 'P01', 'P01_01', '056581', '[(76, 1260, 462, 186)]']
  print('data: type:{} / len:{}'.format(type(data), len(data)))


  now = str(datetime.now())

  annotations = []
  images = []
  licenses = []
  uid = 0
  num_imgs = 0
  img_dict = {}
  do_not_exist = 0
  for [noun_cls, noun, pid, vid, frame, bboxes] in tqdm(data):
    full_path = get_full_path(subset, pid, vid, frame)
    # print(full_path)
    if not os.path.exists(full_path):
      do_not_exist += 1
      continue

    bboxes = parse_list(bboxes)
    if not bboxes:
      continue

    for bbox in bboxes:
      #xmin, ymin = bbox[:2]
      ymin, xmin = bbox[:2]
      #xmax = xmin + bbox[2]
      xmax = xmin + bbox[3]
      #ymax = ymin + bbox[3]
      ymax = ymin + bbox[2]
      seg = [xmin,ymin, xmax,ymin, xmax,ymax, xmin,ymax]

      # CC: swap (1,2) swap (3,4) to match COCO format
      tmp = [bbox[1], bbox[0], bbox[3], bbox[2]]
      bbox = tmp

      area = bbox[2] * bbox[3]
      if area < 1:
        continue

      unique_img_str = os.path.join(pid, vid, epic_frame_format.format(int(frame)))
      if not unique_img_str in img_dict:
        img_dict[unique_img_str] = num_imgs
        img_id = num_imgs
        num_imgs += 1
      else:
        img_id = img_dict[unique_img_str]


      annotations.append({
        'area': bbox[2] * bbox[3],
        'bbox': bbox,
        'category_id': int(noun_cls)+1,
        'id': uid,
        #'image_id': int(frame),
        'image_id': img_id,
        'iscrowd': 0,
        'segmentation': [seg],
        })
      images.append({
        #'id': int(frame),
        'id': img_id,
        'width': 1920, # TODO: are EPIC images uni size?
        'height': 1080,
        'file_name': full_path,
        'license': 'license placeholder',
        'flickr_url': 'flickr_url placeholder',
        'coco_url': 'coco_url placeholder',
        'date_captured': now
      })
      licenses.append({
        'id': uid+1,
        'name': 'name placeholder',
        'url': 'url placeholder'
      })
      uid += 1
  print('#bbox:', uid)
  print('#images:', num_imgs)
  print('#images not found:', do_not_exist)

  info = {
    'year': 2018,
    'version': 'v1',
    'description': 'placeholder for COCO info',
    'contributor': 'BB (not really)',
    'url': '<placeholder url>',
    'data_created': now
  }

  categories = noun_categories()

  with open(fout, 'w') as handle:
    json.dump({
      'info':info,
      'images':images,
      'licenses':licenses,
      'annotations':annotations,
      'categories':categories
      }, handle)

if __name__ == '__main__':
  fin = os.path.join(annot_base_dir, 'EPIC_train_object_labels.csv')
  fout = os.path.join(annot_base_dir, 'coco_train_object_labels_819c.json')
  to_COCO_format(fin, fout, 'train')
