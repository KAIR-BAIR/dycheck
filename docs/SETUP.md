# Setup

The code was tested on Ubuntu 20.04/22.04 and CUDA 11.5/11.7, with Python 3.8.

## 1. Clone this repo recursively.

Our repo uses [RAFT](https://github.com/princeton-vl/RAFT/) and [DPT](https://github.com/isl-org/DPT) as out-of-the-box data processors for optical flow and
monocular depth estimation, respectively.
They are included as git submodules.

Note that for RAFT, we use our forked repo instead of the original one, where the only [difference](https://github.com/hangg7/raft/commit/d5057a70e090f042c8fb492a15327a766073dbf8) we made is that RAFT models are exposed as python modules for easy access.

```bash
# Make sure you include the `--recursive` arg.
git clone https://github.com/hangg7/dycheck --recursive
```

## 2. Create a new `conda` environment.

This step assumes you have already installed [anaconda](https://www.anaconda.com/). If you have not, please follow its instruction.

```bash
# Create a seperate python environment.
conda create -n dycheck python=3.8 -y

```

## 3. Install necessary dependencies.

Our [default requirements](../requirements.txt) are tailored to CUDA 11.5 and 11.7. If you use other CUDA versions, you might want to edit the requirement file by:

1. Change the `jax` and `jaxlib` versions ([instructions](https://github.com/google/jax#installation)).
2. Change the `torch` and `torchvision` versions ([instructions](https://pytorch.org/)).

```bash
# System level `tensorflow-graphics` dependency.
sudo apt install libopenexr-dev ffmpeg -y
# Main project dependencies.
pip install -r requirements.txt
```

Note that it will install the repo as an editable local package, where you can make any changes you want and they will be propagated into the environment automatically.

## 4. [Optional] Prepare for `jupyterlab` to process data.

For users who want to use our annotation tools to create and process their capture, please follow this step. You need to setup up jupyter environment and jupyter kernel.

```bash
# Setup labextensions for annotation and visualization.
jupyter labextension install jupyterlab-plotly ipyevents
# Make sure that you use the `python` executable from the project environment.
python -m ipykernel install --user --name=dycheck
```

## 5. [Optional] Prepare for `rclone` to pull datasets and checkpoints from google drive.

For users who want to use our provided scripts/instructions to download datasets and checkpoints, please follow this step. This step assumes you have already installed [rclone](https://rclone.org/install/), if you have not, please follow its instruction.

```bash
# Set up `rclone` with your google drive. You only need to run this once.
rclone config
```

Then click on our [released folder](https://drive.google.com/drive/folders/1bjGw6NMxJAanoAyIBC1xvJJDUGM61LvU?usp=sharing) to open it in your google drive. You can verify by running the following command to make sure that it is accessible through terminal:

```bash
# `<DRIVE_NAME>` is the name of your google drive profile for `rclone`.
rclone lsd --drive-shared-with-me "<DRIVE_NAME>:/dybench_release"
```
