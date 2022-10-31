#!/bin/bash
#
# File   : val_nerfies_hypernerf_video.sh
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
MODELS=("ambient" "dense" "tnerf")

for SEQUENCE in $SEQUENCES; do
    DATASET="${SEQUENCE%/*}"
    SEQUENCE="${SEQUENCE#*/}"
    # shellcheck disable=2048
    for MODEL in ${MODELS[*]}; do
        python tools/launch.py \
            --gin_configs "configs/${DATASET}/${MODEL}/intl.gin" \
            --gin_bindings "Config.engine_cls=@Evaluator" \
            --gin_bindings "Tasks.task_classes=(@Video,)" \
            --gin_bindings 'SEQUENCE="'"${SEQUENCE}"'"'
        sleep 5s
        python tools/launch.py \
            --gin_configs "configs/${DATASET}/${MODEL}/mono.gin" \
            --gin_bindings "Config.engine_cls=@Evaluator" \
            --gin_bindings "Tasks.task_classes=(@Video,)" \
            --gin_bindings 'SEQUENCE="'"${SEQUENCE}"'"'
        sleep 5s
    done
done
