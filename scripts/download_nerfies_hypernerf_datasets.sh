#!/bin/bash
#
# File   : download_nerfies_hypernerf_dataset.sh
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

set -e -x

DRIVE_NAME=$1

# Download Nerfies dataset.
echo 'Downloading Nerfies dataset...'
mkdir -p datasets/nerfies
pushd datasets/nerfies
wget https://github.com/google/nerfies/releases/download/0.1/nerfies-vrig-dataset-v0.1.zip
unzip nerfies-vrig-dataset-v0.1.zip
rm nerfies-vrig-dataset-v0.1.zip
# Download additional Nerfies data.
echo 'Injecting additional keypoint annotation and meta data to Nerfies dataset...'
rclone copy --drive-shared-with-me --progress "${DRIVE_NAME}:/dycheck-release/datasets/nerfies" .
popd

# Download HyperNeRF dataset.
echo 'Downloading HyperNeRF dataset... Note that "vrig_" prefix will be striped.'
mkdir -p datasets/hypernerf
pushd datasets/hypernerf
wget https://github.com/google/hypernerf/releases/download/v0.1/vrig_3dprinter.zip
unzip vrig_3dprinter.zip
rm vrig_3dprinter.zip
mv vrig_3dprinter 3dprinter
wget https://github.com/google/hypernerf/releases/download/v0.1/vrig_chicken.zip
unzip vrig_chicken.zip
rm vrig_chicken.zip
mv vrig_chicken chicken
wget https://github.com/google/hypernerf/releases/download/v0.1/vrig_peel-banana.zip
unzip vrig_peel-banana.zip
rm vrig_peel-banana.zip
mv vrig_peel-banana peel-banana
# Download additional HyperNeRF data.
echo 'Injecting additional keypoint annotation and meta data to HyperNeRF dataset...'
rclone copy --drive-shared-with-me --progress "${DRIVE_NAME}:/dycheck-release/datasets/hypernerf" .
popd
