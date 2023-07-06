from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os.path
import sys
import build_data
from six.moves import range
import tensorflow as tf
from typing import Sequence, Tuple, Any
FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_string('coco_root', None, 'coco dataset root folder.')

tf.compat.v1.app.flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.')

tf.compat.v1.app.flags.DEFINE_boolean('treat_crowd_as_ignore', True,
                     'Whether to apply ignore labels to crowd pixels in '
                     'panoptic label.')
_NUM_SHARDS = 4
_DATA_FORMAT_MAP = {
    'image': 'jpg',
    'label': 'png',
}
_FOLDERS_MAP = {
    'train': {
        'image': 'train',
        'label': 'annotations',
    },
    'val': {
        'image': 'val',
        'label': 'annotations',
    }}

def _get_images(coco_root: str, dataset_split: str) -> Sequence[str]:
  pattern = '*.%s' % _DATA_FORMAT_MAP['image']
  search_files = os.path.join(
      coco_root, _FOLDERS_MAP[dataset_split]['image'], pattern)
  filenames = tf.io.gfile.glob(search_files)
  return sorted(filenames)

def _convert_dataset(coco_root: str, dataset_split: str,
                     output_dir: str) -> None:
     image_files = _get_images(coco_root, dataset_split)
     num_images = len(image_files)
     num_per_shard = int(math.ceil(len(image_files) / _NUM_SHARDS))
     image_reader = build_data.ImageReader('jpeg', channels=3)
     label_reader = build_data.ImageReader('png', channels=1)
     for shard_id in range(_NUM_SHARDS):
      output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
      with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, num_images)
        for i in range(start_idx, end_idx):
          sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
          sys.stdout.flush()
        image_filename = image_files[i]
        image_data = tf.compat.v1.gfile.GFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        name=os.path.splitext(os.path.basename(image_files[i]))[0]
        seg_filename = os.path.join(coco_root, 
          _FOLDERS_MAP[dataset_split]['label'],'panoptic_%s' % dataset_split,
          name+'_label_ground-truth_coco-panoptic'+ '.' + FLAGS.label_format)
        seg_data = tf.compat.v1.gfile.GFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, image_files[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
      sys.stdout.write('\n')
      sys.stdout.flush()        


        

def main(unused_argv: Sequence[str]) -> None:
  tf.io.gfile.makedirs(FLAGS.output_dir)

  for dataset_split in ('train', 'val'):
    tf.compat.v1.logging.info('Starts processing dataset split %s.', dataset_split)
    _convert_dataset(FLAGS.coco_root, dataset_split, FLAGS.output_dir)


if __name__ == '__main__':
  tf.compat.v1.app.run()
