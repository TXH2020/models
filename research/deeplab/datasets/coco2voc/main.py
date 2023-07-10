import json

from coco2voc import *
from PIL import Image
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Convert coco format data to voc format for deeplab training')
    parser.add_argument('--dataset', dest='dataset',
                        help='directory training dataset',
                        type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    name_bits = 7
    root = args.dataset
    target_folder = os.path.join(root, 'voc')
    instance_target_path = os.path.join(target_folder, 'SegmentationObject')
    class_target_path = os.path.join(target_folder, 'SegmentationClass')
    id_target_path = os.path.join(target_folder, 'SegmentationId')
    image_sets_path = os.path.join(target_folder, 'ImageSets')
    segmentation_path = os.path.join(image_sets_path, 'Segmentation')
    train_image_id_list_path = os.path.join(segmentation_path, f'train.txt')
    val_image_id_list_path = os.path.join(segmentation_path, f'val.txt')
    trainval_image_id_list_path = os.path.join(segmentation_path, f'trainval.txt')
    if not os.path.exists(target_folder):
        os.system(f'mkdir {target_folder}')
        os.system(f'mkdir {image_sets_path}')
        os.system(f'mkdir {segmentation_path}')
    else:
        print(f'Removing {instance_target_path}')
        print(f'Removing {class_target_path}')
        print(f'Removing {id_target_path}')
        print(f"Removing {train_image_id_list_path}")
        print(f"Removing {val_image_id_list_path}")
        print(f"Removing {trainval_image_id_list_path}")
        os.system(f'rm -rf {instance_target_path} {class_target_path} {id_target_path} {train_image_id_list_path} {val_image_id_list_path} {trainval_image_id_list_path}')
        print('Done.\n')
    os.system(f'mkdir {instance_target_path} {class_target_path} {id_target_path}')
    annotations_file = os.path.join(root, 'annotations', 'instances_train.json')
    coco2voc1(annotations_file, target_folder, mode='train', name_bits=name_bits, n=None, compress=True)
    annotations_file = os.path.join(root, 'annotations', 'instances_val.json')
    coco2voc2(annotations_file, target_folder, mode='val', name_bits=name_bits, n=None, compress=True)
