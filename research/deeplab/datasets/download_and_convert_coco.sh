#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
#
# Script to download and preprocess the coco dataset.
#
# Usage:
#   bash ./download_and_convert_coco.sh
#
# The folder structure is assumed to be:
# .(COCO_ROOT)
#+-- train2017
#|   |
#|   +-- *.jpg
#|
#|-- val2017
#|   |
#|   +-- *.jpg
#|
#|-- test2017
#|   |
#|   +-- *.jpg
#|
#+-- annotations
#     |
#     +-- panoptic_{train|val}2017.json
#     +-- panoptic_{train|val}2017

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./coco"

# Root path for ADE20K dataset.
COCO_ROOT="${WORK_DIR}/surface_dataset"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

echo "Converting coco dataset..."
python ./build_coco_data.py  \
  --coco_root="${COCO_ROOT}" \
  --output_dir="${OUTPUT_DIR}"
