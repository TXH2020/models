# coding=utf-8
# Copyright 2023 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides data from segmentation datasets.

Currently, we support the following datasets:

1. Cityscapes dataset (https://www.cityscapes-dataset.com).

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.


2. KITTI-STEP (http://www.cvlibs.net/datasets/kitti/).

The KITTI-STEP enriches the KITTI-MOTS data with additional `stuff'
anntotations.

3. MOTChallenge-STEP (https://motchallenge.net/).

The MOTChallenge-STEP enriches the MOTSChallenge data with additional `stuff'
annotations.

4. MSCOCO panoptic segmentation (http://cocodataset.org/#panoptic-2018).

Panoptic segmentation annotations for MSCOCO dataset. Note that we convert the
provided MSCOCO panoptic segmentation format to the following one:
panoptic label = semantic label * 256 + instance id.

5. Cityscapes-DVPS (https://github.com/joe-siyuan-qiao/ViP-DeepLab)

The Cityscapes-DVPS dataset augments Cityscapes-VPS
(https://github.com/mcahny/vps) with depth annotations.

6. SemKITTI-DVPS (https://github.com/joe-siyuan-qiao/ViP-DeepLab)

The SemKITTI-DVPS dataset converts 3D point annotations of SemanticKITTI
(http://www.semantic-kitti.org) into 2D image labels.

7. WOD-PVPS
(https://waymo.com/open/data/perception/#2d-video-panoptic-segmentation)

The Waymo Open Dataset: Panoramic Video Panoptic Segmentation contains high
quality panoramic video annotations with time and cross-camera consistency.
The Waymo Open Dataset (WOD): Panoramic Video Panoptic Segmentation (PVPS)
contains high quality panoramic video annotations with time and cross-camera
consistency.

8. ADE20K panoptic segmentation
(https://groups.csail.mit.edu/vision/datasets/ADE20K/)

Panoptic segmentation annotations for ADE20K dataset. Note that we convert the
provided ADE20K panoptic segmentation format to the following one:
panoptic label = semantic label * 1000 + instance id.

We can use the dataset in the following settings:
- In the multicamera setting, an example contains all camera data within a
frame (i.e., the instance correspondence between cameras are guaranteed).
- In the non-multicamera setting, we could still use all the camera frames, but
treat them as individual frames. Datasets variants with a subset of cameras
could be done by discarding the rest of the cameras data when creating the
dataset.

The following variants are provided:
- WOD_PVPS_IMAGE_PANOPTIC_SEG: WOD-PVPS dataset as a single frame panoptic
segmentation dataset.
- WOD_PVPS_IMAGE_PANOPTIC_SEG_MULTICAM: WOD-PVPS dataset as a single frame
panoptic segmentation dataset in the multicamera setting.
- WOD_PVPS_DEPTH_VIDEO_PANOPTIC_SEG: WOD-PVPS dataset as a depth-aware video
panoptic segmentation dataset. The users could ignore the provided depth
groundtruth and use the dataset for the task of video panoptic segmentation.
- WOD_PVPS_DEPTH_VIDEO_PANOPTIC_SEG_MULTICAM: WOD-PVPS dataset as as a
depth-aware video panoptic segmentation dataset in the multicamera setting. The
users could ignore the provided depth groundtruth and use the dataset for the
task of panoramic video panoptic segmentation.


References:

- Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus
  Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele, "The
  Cityscapes Dataset for Semantic Urban Scene Understanding." In CVPR, 2016.

- Andreas Geiger and Philip Lenz and Raquel Urtasun, "Are we ready for
  Autonomous Driving? The KITTI Vision Benchmark Suite." In CVPR, 2012.

- Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr
  Dollar, "Panoptic Segmentation." In CVPR, 2019.

- Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B.
  Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C.
  Lawrence Zitnick, "Microsoft COCO: common objects in context." In ECCV, 2014.

- Anton Milan, Laura Leal-Taixe, Ian Reid, Stefan Roth, and Konrad Schindler,
  "Mot16: A benchmark for multi-object tracking." arXiv:1603.00831, 2016.

- Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin
  Balachandar Gnana Sekar, Andreas Geiger, and Bastian Leibe. "MOTS:
  Multi-object tracking and segmentation." In CVPR, 2019

- Mark Weber, Jun Xie, Maxwell Collins, Yukun Zhu, Paul Voigtlaender, Hartwig
  Adam, Bradley Green, Andreas Geiger, Bastian Leibe, Daniel Cremers, Aljosa
  Osep, Laura Leal-Taixe, and Liang-Chieh Chen, "STEP: Segmenting and Tracking
  Every Pixel." arXiv: 2102.11859, 2021.

- Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. "Video panoptic
  segmentation." In CVPR, 2020.

- Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill
  Stachniss, and Jurgen Gall. "Semantickitti: A dataset for semantic scene
  understanding of lidar sequences." In ICCV, 2019.

- Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
  "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
  Segmentation." In CVPR, 2021.

- Jieru Mei, Alex Zihao Zhu, Xinchen Yan, Hang Yan, Siyuan Qiao, Yukun Zhu,
  Liang-Chieh Chen, Henrik Kretzschmar, and Dragomir Anguelov. "Waymo Open
  Dataset: Panoramic Video Panoptic Segmentation." arXiv: 2206.07704, 2022.

- Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso,
  and Antonio Torralba. "Scene Parsing Through ADE20K Dataset." In CVPR, 2017.

"""

import collections

# Dataset names.

_COCO_PANOPTIC = 'coco_panoptic'


# Colormap names.
COCO_COLORMAP = 'coco'


# Named tuple to describe dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor', [
        'dataset_name',  # Dataset name.
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',   # Number of semantic classes.
        'ignore_label',  # Ignore label value used for semantic segmentation.

        # Fields below are used for panoptic segmentation and will be None for
        # Semantic segmentation datasets.
        # Label divisor only used in panoptic segmentation annotation to infer
        # semantic label and instance id.
        'panoptic_label_divisor',
        # A tuple of classes that contains instance annotations. For example,
        # 'person' class has instance annotations while 'sky' does not.
        'class_has_instances_list',
        # A flag indicating whether the dataset is a video dataset that contains
        # sequence IDs and frame IDs.
        'is_video_dataset',
        # A string specifying the colormap that should be used for
        # visualization. E.g. 'cityscapes'.
        'colormap',
        # A flag indicating whether the dataset contains depth annotation.
        'is_depth_dataset',
        # The ignore label for depth annotations.
        'ignore_depth',
        # A list of camera names, only for multicamera setup.
        'camera_names',
    ]
)


def _build_dataset_info(**kwargs):
  """Builds dataset information with default values."""
  default = {
      'camera_names': None,
  }
  default.update(kwargs)
  return DatasetDescriptor(**default)


COCO_PANOPTIC_INFORMATION = _build_dataset_info(
    dataset_name=_COCO_PANOPTIC,
    splits_to_sizes={'train': 118,
                     'val': 5,
                     'test': 30},
    num_classes=2,
    ignore_label=255,
    panoptic_label_divisor=256,
    class_has_instances_list=tuple(range(1, 3)),
    is_video_dataset=False,
    colormap=COCO_COLORMAP,
    is_depth_dataset=False,
    ignore_depth=None,
)



MAP_NAME_TO_DATASET_INFO = {
    _COCO_PANOPTIC: COCO_PANOPTIC_INFORMATION 
}

MAP_NAMES = list(MAP_NAME_TO_DATASET_INFO.keys())

