'''
Kesar TN
University of Central Florida
kesar@Knights.ucf.edu

Example usage:
    python create_tf.py --logtostderr --image_dir="scaled2500_1to2/images/" --annotations_file="scaled2500_1to2/scaled2500_1to2.json" --output_dir="tfrecords/"
    python create_tf.py --logtostderr --image_dir="scaled2500_25to35day/images/" --annotations_file="scaled2500_25to35day/scaled2500_25to35day.json" --output_dir="tfrecords/"

'''
import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image

import tensorflow as tf
from dataset import *
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

dataset = AtrDataset('scaled2500_1to2')

flags = tf.app.flags
tf.flags.DEFINE_string('image_dir', '',
                       'Data image directory.')
tf.flags.DEFINE_string('annotations_file', '',
                       'Data annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      total_targets):

  filename = image['filename']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  image_array = Image.open(full_path)
  image_array = np.asarray(image_array)                    
  image_height = image_array.shape[0]
  image_width = image_array.shape[1]

  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []; xmax = []; ymin = []; ymax = []
  category_names = []
  category_ids = []
  area = []
  num_annotations_skipped = 0

  for object_annotations in annotations_list:
    for target in object_annotations['targets']:
      total_targets += 1
      cx = target['center'][0]
      cy = target['center'][1]

      x = target['ul'][0]
      y = target['ul'][1]

      width = (cx - x) * 2
      height = (cy - y) * 2

      if width <= 0 or height <= 0:
        num_annotations_skipped += 1
        continue
      if x + width > image_width or y + height > image_height:
        num_annotations_skipped += 1
        continue
      xmin.append(float(x) / image_width)
      xmax.append(float(x + width) / image_width)
      ymin.append(float(y) / image_height)
      ymax.append(float(y + height) / image_height)
      category_id = int(dataset._category_lookup(target['category']))
      category_ids.append(category_id)
      category_names.append(target['category'].encode('utf8'))
      area.append(target['bbox_area'])

  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped, total_targets

def _create_tf_record_from_coco_annotations(
    annotations_file, image_dir, output_path, num_shards):
  with contextlib2.ExitStack() as tf_record_close_stack, tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    groundtruth_data = json.load(fid)
    images = dataset._image_names_dict()
    category_index = label_map_util.create_category_index(dataset.get_categories())

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      tf.logging.info(
          'Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['sample_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    total_targets = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    tf.logging.info('%d images are missing annotations.', missing_annotation_count)

    total_num_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      _, tf_example, num_annotations_skipped, total_targets = create_tf_example(
          image, annotations_list, image_dir, category_index, total_targets)
      total_num_annotations_skipped += num_annotations_skipped
      shard_idx = idx % num_shards
      output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    
    print('Total Number of targets encoded: ', total_targets)
    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)


def main(_):
  assert FLAGS.image_dir, '`image_dir` missing.'
  assert FLAGS.annotations_file, '`annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  output_path = os.path.join(FLAGS.output_dir, 'train/train.record')

  _create_tf_record_from_coco_annotations(
      FLAGS.annotations_file,
      FLAGS.image_dir,
      output_path,
      num_shards=100)

if __name__ == '__main__':

    tf.app.run()