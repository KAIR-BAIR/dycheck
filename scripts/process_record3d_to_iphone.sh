#!/bin/bash
#
# File   : process_record3d_to_iphone.sh
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

# shellcheck disable=2124
SEQUENCES=${@:1}

for SEQUENCE in $SEQUENCES; do
    START="0"
    END="None"
    ROTATE_MODE="clockwise_0"
    case "${SEQUENCE}" in
        "apple"*)
            FRGD_PROMPTS="('a person','an apple')"
            ;;
        "backpack"*)
            FRGD_PROMPTS="('a person','a backpack')"
            END=180
            ;;
        "block"*)
            FRGD_PROMPTS="('a person','a set of blocks')"
            END=447
            ;;
        "creeper"*)
            FRGD_PROMPTS="('a creeper in the wind',)"
            END=360
            ;;
        "handwavy"*)
            FRGD_PROMPTS="('a person','a waving hand')"
            ;;
        "haru-sit"*)
            FRGD_PROMPTS="('a dog',)"
            START=400
            END=600
            ROTATE_MODE="clockwise_90"
            ;;
        "mochi-high-five"*)
            FRGD_PROMPTS="('a cat',)"
            END=180
            ;;
        "paper-windmill"*)
            FRGD_PROMPTS="('a person',)"
            END=277
            ;;
        "pillow"*)
            FRGD_PROMPTS="('a person','a pillow')"
            END=330
            ;;
        "space-out"*)
            FRGD_PROMPTS="('a person',)"
            ;;
        "spin"*)
            FRGD_PROMPTS="('a person',)"
            END=426
            ;;
        "sriracha-tree"*)
            FRGD_PROMPTS="('a cat',)"
            ;;
        "teddy"*)
            FRGD_PROMPTS="('a person','a teddy')"
            ;;
        "wheel"*)
            FRGD_PROMPTS="('a person','a steering wheel')"
            ;;
        *)
            echo "Unknown sequence: ${SEQUENCE}."
            exit 1
            ;;
    esac

    # shellcheck disable=2086
    python tools/process_record3d_to_iphone.py \
        --gin_configs "configs/iphone/process_record3d_to_iphone.gin" \
        --gin_bindings 'SEQUENCE="'"${SEQUENCE}"'"' \
        --gin_bindings "FRGD_PROMPTS=${FRGD_PROMPTS}" \
        --gin_bindings 'Record3DProcessor.rotate_mode="'"${ROTATE_MODE}"'"' \
        --gin_bindings "Record3DProcessor.start=${START}" \
        --gin_bindings "Record3DProcessor.end=${END}"
done
