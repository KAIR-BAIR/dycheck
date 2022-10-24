# Datasets

Please find the processed datasets used in our paper at [here](https://drive.google.com/drive/u/0/folders/1ZYQQh0qkvpoGXFIcK_j4suon1Wt6MXdZ), including:

1. Additional co-visibility masks and keypoint annotations for Nerfies-HyperNeRF dataset.
2. Our accompanying iPhone dataset with more diverse and complex real-life motions.

All datasets are assumed to be organized as `<DYCHECK_ROOT>/datasets/<DATASET>/<SEQUENCE>`.

## (0) Additional data

We follow and extend the data [format](https://github.com/google/nerfies#datasets) adopted by Nerfies for better training and evaluation.

In total, we have the following (additional data are marked with `**`):

```
<DYCHECK_ROOT>/datasets/<DATASET>/<SEQUENCE>/
    ├── camera/
    │   └── <ITEM_ID>.json
    ├── **covisible/**
    │   └── <SCALE>x
    │       └── <SPLIT>
    │           └── <ITEM_ID>.png
    ├── **depth/**
    │   └── <SCALE>x
    │       └── <ITEM_ID>.npy
    ├── **keypoint/**
    │   └── <SCALE>x
    │       └── <SPLIT>
    │           ├── <ITEM_ID>.json
    │           └── skeleton.json
    ├── rgb/
    │   └── <SCALE>x
    │       └── <ITEM_ID>.png
    ├── **splits/**
    │   └── <SPLIT>.json
    ├── dataset.json
    ├── **emf**.json
    ├── **extra**.json
    ├── metadata.json
    ├── points.npy
    └── scene.json
```

<details>
<summary>Click to see per-item explainations.</summary>

### `covisible/`

- Contains pre-computed co-visibility masks for masked image metrics.
- Only available if there are multi-camera validation frames.
- For each pixel in the test image `<ITEM_ID>.png` given by the `<SPLIT>`, store the binary value of whether it is co-visible during training. `1` means co-visible and `0` the otherwise.
- As specified in our paper, we deem a test pixel not co-visible if it is seen in less than `max(5, len(train_imgs) // 10)` training images.

### `depth/`

- Contains metric Lidar depth from iPhone sensor in millimeter.
- Only available for the training frames in our iPhone dataset.

### `keypoint/`

- Contains our manual keypoint annotations for correspondence evaluation.
- Only available for 10 uniformly sampled training frames.
- Each keypoint frame `<ITEM_ID>.json` contains a `(J, 3)` list for `J` keypoints, where each row contain the `(x, y, v)` information for the absolute keypoint position `(x, y)` and its visibility in this frame `v` in binary.
- `skeleton.json` stores a serialization of the [skeleton definition](https://github.com/hangg7/dybench/blob/main/dybench/utils/visuals/kps/skeleton.py#L56-L62).

### `splits/`

- Contains data split dictionary for toggling between the teleporting setting of existing methods and the actual monocular (non-teleporting) setting.
- iPhone dataset contains simple `train/val` splits, while Nerfies-HyperNeRF contains `train_/val_<SETTING>` where `<SETTING>` is one of `["mono", "intl", "common"]`.
- They are [defined](https://github.com/hangg7/dybench/blob/main/dybench/datasets/nerfies.py#L409-L455), use Nerfies-HyperNeRF as an example, and will be automatically generated the first time you run the dataloader (by running any of our evaluation/training scripts).
- Each `<SPLIT>.json` contains a dictionary of the following format:

```
{
    // The name of each frame; equivalent to <ITEM_ID>.
    "frame_names": ["left1_000000", ...],
    // The camera ID for indexing the physical camera.
    "camera_ids": [0, ...],
    // The time ID for indexing the time step.
    "time_ids": [0, ...]
}
```

### `emf.json`

- Contins pre-computed effective multi-view factors of the captured training video in the following format:

```
{
    // The training video split name.
    "train_intl": {
        // The Full EMF.
        "Omega": 7.377937316894531,
        // The Angular EMF in deg/sec.
        "omega": 212.57649834023107
    },
    ...
}
```

### `extra.json`

- Contins extra information for evaluation and visualizations in the following format:

```
{
    // Scene bounding box; computed from the original sequence in the
    // normalized coordinate.
    "bbox": [
        [
            -0.24016861617565155,
            -0.35341497925615145,
            -0.6217879056930542
        ],
        [
            0.2697989344596863,
            0.09984857437828297,
            0.42420864422383164
        ]
    ],
    // Video downsampling factor for training and evaluation.
    "factor": 4,
    // Video frame rate.
    "fps": 15,
    // The scene look-at point in the normalized coordinate.
    "lookat": [
        0.0018036571564152837,
        -0.07246068120002747,
        0.05924934893846512
    ],
    // The scene up vector in the normalized coordinate.
    "up": [
        -0.07553146034479141,
        -0.9961089491844177,
        0.045409008860588074
    ]
}
```

- They are [defined](https://github.com/hangg7/dybench/blob/main/dybench/datasets/nerfies.py#L367-L407), use Nerfies-HyperNeRF as an example, and will be automatically generated the first time you run the dataloader (by running any of our evaluation/training scripts).

</details>

## (1) Nerfies-HyperNeRF dataset

Due to their similar capture protocol and data format, we combine the datasets released from Nerfies and HyperNeRF and denote it "Nerfies-HyperNeRF" dataset.

Please refer to [Nerfies's](https://github.com/google/nerfies/releases) and [HyperNeRF's](https://github.com/google/hypernerf/releases) release pages for their download instructions. Together, there should be 7 sequences in total.

We prepared a simple download script for you to set up data from scratch:

```bash
# Download Nerfies and HyperNeRF datasets and inject our additional data.
bash scripts/download_nerfies_hypernerf_datasets.sh <DRIVE_NAME>
```

<details>
<summary>Click to see additional details.</summary>

- The original training sequences on Nerfies-HyperNeRF are generated with severe camera motion due to camera teleportation.
- We focus on the non-teleporting setting by only using the left camera for training in each capture.
- We stripe the `vrig-` prefix from the HyperNeRF sequence names since we only focus on the captures with validation camera rig. This is also taken care of in our download script above.

</details>

## (2) iPhone dataset

In our work, we propose a new dataset captured with iPhone captured in the wild, featuring diverse and complex motions comparing to Nerfies-HyperNeRF dataset.

This dataset contains 7 multi-camera captures and 7 single-camera captures. These captures are made without camera teleportation as we slowly moves the camera around a moving scene. We intentionally include a variety of subjects: general objects, quadruped (our pets), and humans. For visualizations on our captures, please refer to our project page.

We use a third party iOS app ([Record3D](https://record3d.app/)) for depth sensing.

We prepared a simple download script for you to set up data from scratch:

```bash
# Download iPhone dataset.
bash scripts/download_iphone.sh <DRIVE_NAME>
```

<details>
<summary>Click to see additional details.</summary>

- The whole dataset is around 28 GB in total.
- To get started, you might want to just download one sequence first. Make sure you download each sequence into `datasets/iphone/<SEQUENCE>` directory.

</details>

## (3) Process your own captures following our procedure

Please refer to [RECORD3D_CAPTURE.md](RECORD3D_CAPTURE.md) for our documentation.
