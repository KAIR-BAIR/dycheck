#!/bin/bash
#
# File   : download_record3d_raw_data.sh
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

# Download Record3D raw data.
echo 'Downloading Record3D raw data...'
mkdir -p datasets/record3d
pushd datasets/record3d
rclone copy --drive-shared-with-me --progress "${DRIVE_NAME}:/dycheck-release/datasets/record3d" .
popd
