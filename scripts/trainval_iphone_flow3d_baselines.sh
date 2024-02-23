#!/bin/bash
#
# File   : trainval_iphone_flow3d_baselines.sh
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
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

# shellcheck disable=2124
SEQUENCES=${@:1}
MODELS=("ambient" "tnerf")

for SEQUENCE in $SEQUENCES; do
	DATASET="${SEQUENCE%/*}"
	SEQUENCE="${SEQUENCE#*/}"
	# shellcheck disable=2048
	for MODEL in ${MODELS[*]}; do
		# python tools/launch.py \
		#     --gin_configs "configs/${DATASET}/${MODEL}/randbkgd_depth_dist.gin" \
		#     --gin_bindings "Config.engine_cls=@Trainer" \
		#     --gin_bindings 'SEQUENCE="'"${SEQUENCE}"'"' \
		#     --gin_bindings 'iPhoneParser.factor=1' \
		#     --gin_bindings 'iPhoneParser.depth_name="flow3d_preprocessed/aligned_depth_anything"' \
		#     --gin_bindings 'iPhoneParser.covisible_name="flow3d_preprocessed/covisible"'
		python tools/launch.py \
			--gin_configs "configs/${DATASET}/${MODEL}/randbkgd_depth_dist.gin" \
			--gin_bindings "Config.engine_cls=@Trainer" \
			--gin_bindings 'SEQUENCE="'"${SEQUENCE}"'"' \
			--gin_bindings 'iPhoneParser.factor=1' \
			--gin_bindings 'iPhoneParser.depth_name="flow3d_preprocessed/aligned_depth_anything_colmap"' \
			--gin_bindings 'iPhoneParser.covisible_name="flow3d_preprocessed/covisible"' \
			--gin_bindings 'iPhoneParser.use_refined_camera=True'
		sleep 5s
	done
done
