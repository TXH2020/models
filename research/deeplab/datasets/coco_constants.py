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

"""File containing the meta info of COCO dataset.
"""

import copy
from typing import Sequence, Mapping, Any

_COCO_META = [
    {
        'color': [0, 113, 188],
        'isthing': 0,
        'id': 1,
        'name': 'horizontal_surface'
    },
    {
        'color': [216, 82, 24],
        'isthing': 0,
        'id': 2,
        'name': 'vertical_surface'
    },
    
]


def get_coco_meta() -> Sequence[Any]:
  return copy.deepcopy(_COCO_META)


def get_id_mapping() -> Mapping[int, int]:
  """Creates a dictionary mapping the original category_id into continuous ones.

  Specifically, in coco annotations, category_id ranges from 1 to 200. Since not
  every id between 1 to 200 is used, we map them to contiguous ids (1 to 133),
  which saves memory and computation to some degree.

  Returns:
    A dictionary mapping original category id to contiguous category ids.
  """
  id_mapping = {}
  for i in range(len(_COCO_META)):
    id_mapping[_COCO_META[i]['id']] = i + 1
  return id_mapping


def get_id_mapping_inverse() -> Sequence[int]:
  """Creates a tuple mapping the continuous ids back to original ones.

  Specifically, in coco annotations, category_id ranges from 1 to 200. Since not
  every id between 1 to 200 is used, we map them to contiguous ids (1 to 133)
  via the function get_id_mapping, which saves memory and computation to some
  degree. This function supports remapping back from the contiguous ids to the
  original ones, which is required for COCO official evaluation.

  Returns:
    A dictionary mapping contiguous category ids to original COCO category id.
  """
  id_mapping_inverse = (0,) + tuple([ori_cat['id'] for ori_cat in _COCO_META])
  return id_mapping_inverse


def get_coco_reduced_meta() -> Sequence[Any]:
  coco_reduced_meta = get_coco_meta()
  id_mapping = get_id_mapping()
  for i in range(len(coco_reduced_meta)):
    coco_reduced_meta[i].update({'id': id_mapping[coco_reduced_meta[i]['id']]})
  return coco_reduced_meta
