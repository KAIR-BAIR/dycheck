# Monocular Dynamic View Synthesis: A Reality Check

### [Paper](https://arxiv.org/abs/2210.13445) | [Project Page](https://hangg7.com/dycheck) | [Video](https://www.youtube.com/watch?v=WwESsNivJP8&t=21s) | [Data](https://drive.google.com/drive/folders/1ZYQQh0qkvpoGXFIcK_j4suon1Wt6MXdZ?usp=sharing)

This repo contains training, evaluation, and visualization code for the reality check that we descripted in our paper on the recent advance in Dynamic View Synthesis (DVS) from monocular video.

Please refer to our project page for more visualizations and qualitative results.

**Monocular Dynamic View Synthesis: A Reality Check** <br> [Hang Gao](https://people.eecs.berkeley.edu/~hangg/),
[Ruilong Li](https://www.liruilong.cn/),
[Shubham Tulsiani](https://shubhtuls.github.io/),
[Bryan Russell](https://bryanrussell.org/),
[Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/).
In NeurIPS, 2022.

<!-- **Table of Contents** -->
<!-- <br> -->

<!-- 1. [Learned Perceptual Image Patch Similarity (LPIPS) metric](#1-learned-perceptual-image-patch-similarity-lpips-metric)<br> -->
<!--    a. [Basic Usage](#a-basic-usage) If you just want to run the metric through command line, this is all you need.<br> -->
<!--    b. ["Perceptual Loss" usage](#b-backpropping-through-the-metric)<br> -->
<!--    c. [About the metric](#c-about-the-metric)<br> -->
<!-- 2. [Berkeley-Adobe Perceptual Patch Similarity (BAPPS) dataset](#2-berkeley-adobe-perceptual-patch-similarity-bapps-dataset)<br> -->
<!--    a. [Download](#a-downloading-the-dataset)<br> -->
<!--    b. [Evaluation](#b-evaluating-a-perceptual-similarity-metric-on-a-dataset)<br> -->
<!--    c. [About the dataset](#c-about-the-dataset)<br> -->
<!--    d. [Train the metric using the dataset](#d-using-the-dataset-to-train-the-metric)<br> -->

## Setup

Please refer to [SETUP.md](docs/SETUP.md) for instructions on setting up a work environment.

By default, our code runs on 4 NVIDIA RTX A5000 GPUs (24 GB memory). Please try decreasing the chunk size if you have fewer resources. You can do so by the following syntax for all of demo/evaluation/training code:

```bash
# Append rendering chunk at the end of your command. Set it to something
# smaller than the default 8192 in case of OOM.
... --gin_bindings="get_prender_image.chunk=<CHUNK_SIZE>"
```

## Quick start

Here is a demo to get you started: re-rendering a paper-windmill video from a pre-trained T-NeRF model. Under the `demo` folder, we include [a minimal dataset](./demo/datasets/iphone/paper-windmill) and [a model checkpoint](demo/work_dirs/iphone/paper-windmill/tnerf/randbkgd_depth_dist/checkpoints) for this purpose.

```bash
# Launch a demo task.
python demo/launch.py --task <TASK>
```

You should be able to get the results below by specifying each task:

<table width="100%">
    <tr align=center>
        <td width="24%"><img src="https://drive.google.com/uc?export=view&id=1zsP4GcuoFK-xwR3oVwNs2pvHtvVnCmhh" width="100%"></td>
        <td width="24%"><img src="https://drive.google.com/uc?export=view&id=1D02bPteah5m2o5YLgyBsnHIhAxIYrhza" width="100%"></td>
        <td width="24%"><img src="https://drive.google.com/uc?export=view&id=1L7tO3dqU6ibKPZyBaKVusFWqAYcUPB_b" width="100%"></td>
        <td width="24%"><img src="https://drive.google.com/uc?export=view&id=1w9OD53Os7tn2EcNa7tmIXwN9ZI_tXvgV" width="100%"></td>
    </tr>
    <tr align=center>
        <td width="24%"><b>Training video</b></td>
        <td width="24%"><b>Novel-view</b><br>fix time; move camera<br><code>&#60;TASK&#62;="novel_view"</code></td>
        <td width="24%"><b>Stabilized-view</b><br>fix camera; replay time<br><code>&#60;TASK&#62;="stabilized_view"</code></td>
        <td width="24%"><b>Bullet-time</b><br>move camera; replay time<br><code>&#60;TASK&#62;="bullet_time"</code></td>
    </tr>
</table>

<details>
<summary>Click to see additional details.</summary>

- The minimal dataset only contain the camera and meta information without the actual video frames.
- The model is our baseline T-NeRF with additional regularizations ([config](configs/iphone/tnerf/randbkgd_depth_dist.gin)) which we find competitive comparing to SOTA methods.
- It takes roughtly 3 minutes to render for novel-view task and 12 minutes for the others at 0.3 FPS.

</details>

## Datasets

Please refer to [DATASETS.md](docs/DATASETS.md) for instructions on downloading processed datasets used in our paper, including:

1. Additional co-visibility masks and keypoint annotations for Nerfies-HyperNeRF dataset.
2. Our accompanying iPhone dataset with more diverse and complex real-life motions.

## Benchmark

Please refer to [BENCHMARK.md](docs/BENCHMARK.md) for our main results and instructions on reproducibility, including:

1. How to evaluate our released checkpoints.
2. How to train from scratch using our configurations.

## Highlights

### (1) Compute Effective Multiview Factors (EMFs) for your own captures.

In our paper, we show a monocular video contains effective multi-view cues when the camera moves much faster than the scene.
We propose to quantify this phenomenon by Effective Multiview Factors (EMFs) which measure the relative camera-scene motion.
Using these metrics, we find that existing datasets operate under multi-view regime, even though technically the camera only observes from one viewpoint at each time step.

To increase the transparency on the experimental protocol, we recommend future works to report their EMFs on their new sequences.

There are two EMFs: Angular EMF ($\omega$) and Full EMF ($\Omega$). The first one is easy to compute but assumes there's a single look-at point of the sequence. The second one is generally applicable but uses optical flow and monocular depth prediction for 3D scene flow estimation and is thus usually noisy. We recommend trying out the Angular EMF first whenever possible.

To use our repo as a module in python:

```python
from dycheck import processors

# 1. Angular EMF (omega): camera angular speed.
# We recommend trying it out first whenever possible.
angular_emf = processors.compute_angular_emf(
    orientations,   # (N, 3, 3) np.ndarray for world-to-camera transforms in the OpenCV format.
    positions,      # (N, 3) np.ndarray for camera positions in world space.
    fps,            # Video FPS.
    lookat=lookat,  # Optional camera lookat point. If None, will be computed by triangulating camera optical axes.
)

# 2. Full EMF (Omega): relative camera-scene motion ratio.
full_emf = processors.compute_full_emf(
    rgbs,           # (N, H, W, 3) np.ndarray video frames in either uint8 or float32.
    cameras,        # A sequence of N camera objects.
    bkgd_points,    # (P, 3) np.ndarray for background points.
)
```

Please see the [camera definition](https://github.com/hangg7/dybench/blob/release/dycheck/geometry/camera.py#L244-L282) in our repo, which follows the one in Nerfies.

To use our repo as a script that takes a video as input, given that the dataset is preprocessed in Nerfies' data format:

```bash
python tools/process_emf.py \
    --gin_configs 'configs/<DATASET>/process_emf.gin' \
    --gin_bindings 'SEQUENCE="<SEQUENCE>"'
```

### (2) Compute Co-visibility mask for better image metrics.

Coming soon.

### (3) Capture and preprocess your own dataset.

Please refer to [docs/RECORD3D_CAPTURE.md](docs/SETUP.md) for instructions on creating your own dataset.

## Citation

If you find this repository useful for your research, please use the following:

```txt
@inproceedings{gao2022dynamic,
    title={Dynamic Novel-View Synthesis: A Reality Check},
    author={Gao, Hang and Li, Ruilong and Tulsiani, Shubham and Russell, Bryan and Kanazawa, Angjoo},
    booktitle={NeurIPS},
    year={2022},
}
```

## Acknowledgement

TODO
