#!/bin/bash
#
# File   : setup.sh
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

conda create -n dycheck python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate dycheck
echo "layout anaconda dycheck" > .envrc
direnv allow

# tensorflow-graphics dependency.
sudo apt install libopenexr-dev ffmpeg -y

pip install -r requirements.txt

# Setup jupyter tools for annotation and visualization.
jupyter labextension install jupyterlab-plotly ipyevents

# Setup git hooks.
cp .dev/git-hooks/pre-commit .git/hooks/pre-commit
