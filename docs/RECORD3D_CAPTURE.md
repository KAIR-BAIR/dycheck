# Record3D capture and preprocessing

In this page, we document our capture and data preprocessing procedure for future works to create their own captures, either with a single camera or a multi-camera rig.

## (0) Prerequisite: app and devices

We capture all of our data using iPhone 14 Pros for their Lidar support. We download the Record3D application from the iOS App Store, which takes $1 to allow exporting depth data.

For our single-camera captures, we use only one device. For our multi-camera captures, we use three devices in total for most cases.
For the two testing cameras, we use a tripod camera mount for each, such that we can set them up with wide baseline before capturing and fix them still after.

## (1) Capture exportation

Now that assume you have done capturing, it's time to export to your server. We export both rgb frames, cameras and raw point cloud PLY files during this step.

Our released scripts assume that you have capture structure like following:

```bash
.
├── RGBD_Video  # Export from the training camera.
│   ├── <TIME_STEP_TRAIN>.mp4
│   └── metadata.json
├── sound.m4a
├── Test  # Optional, only for multi-camera captures.
│   ├── 0  # Export from the testing 0 camera.
│   │   ├── RGBD_Video
│   │   │   ├── <TIME_STEP_TEST_0>.mp4
│   │   │   └── metadata.json
│   │   └── sound.m4a
│   └── 1  # Export from the testing 1 camera.
│       ├── RGBD_Video
│       │   ├── <TIME_STEP_TEST_1>.mp4
│       │   └── metadata.json
│       └── sound.m4a
└── Zipped_PLY  # Contain depth information from the training camera.
    └── <TIME_STEP_TRAIN>.zip
```

Note that we only extract depth for the training camera, as we will not use it for evaluation.

## (2) [if multi-camera] Temporal synchronization

1. First make sure to combine AV channels for audio-based temporal synchronization.

```bash
python tools/combine_record3d_av.py \
   --gin_configs configs/iphone/combine_record3d_av.gin \
   --gin_bindings 'combine_record3d_av.Config.data_root="/shared/hangg/datasets/iphone/record3d"' \
   --gin_bindings 'SEQUENCE="<SEQUENCE>"'
```

2. In Adobe Premiere Pro, create a multi-camera source sequence from your capture and open it in "timeline". Here's a [tutorial](https://www.youtube.com/watch?v=57Aek730DNU).
3. Showing the frame number instead of the second unit at the editting console. Here's an [instruction](https://creativecow.net/forums/thread/how-to-change-timeline-display-to-minsecframesae/#:~:text=Go%20to%20Project%3EProject%20Settings,Format%20from%20Frames%20to%20Timecode).
4. Checking and recording the offset between each testing and training video. Document it into the record3d capture directory.

```bash
# <OFFSET> is an integer inferred from Adobe Promiere Pro audio-based synchronization.
# Repeat for all of your testing cameras.
echo <OFFSET> > <TEST_VIDEO_ID>/RGBD_Video/timesync_offset.txt
```

## (3) Setting preprocessing parameters

Now you are ready for preprocessing the Record3D capture into the format of iPhone dataset.
We provide [our processing script](scripts/process_record3d_to_iphone.sh) for your reference.

Note that you will have to specify the start and end frame (possibly trim the video), the foreground prompts for segmentation (to extract background points for Nerfies' background loss), rotation mode (possibly change the image orientation), and depth filtering parameters, which you can usually simply leave as-is.

```bash
# Edit `scripts/process_record3d_to_iphone.sh` with your new sequence.
```

## (4) [if multi-camera] Annotating bad testing frames.

To ensure good view coverage in training sequence with respect to the testing sequences, we intentionally move the training camera in front of the testing cameras at certain frames. At these time steps, the testing frames being occluded is not valid for evaluation, and thus need to be filtered out.

We provide an annotation tool in `tools/annotate_record3d_bad_frames.ipynb` that allows you to do so. Make sure that you set up the jupyter environment correctly by following the [instruction](docs/SETUP.md).

## (5) Preprocessing your Record3D capture.

Check that your capture directory should look something like this.

```bash
.
├── RGBD_Video
│   ├── <TIME_STEP_TRAIN>.mp4
│   └── metadata.json
├── sound.m4a
├── Test
│   ├── 0
│   │   ├── RGBD_Video
│   │   │   ├── <TIME_STEP_TEST_0>.mp4
│   │   │   ├── metadata.json
│   │   │   ├── timesync_bad_frames.txt     # If multi-camera.
│   │   │   └── timesync_offset.txt         # If multi-camera.
│   │   └── sound.m4a
│   └── 1
│       ├── RGBD_Video
│       │   ├── <TIME_STEP_TEST_1>.mp4
│       │   ├── metadata.json
│       │   ├── timesync_bad_frames.txt     # If multi-camera.
│       │   └── timesync_offset.txt         # If multi-camera.
│       └── sound.m4a
└── Zipped_PLY
    └── <TIME_STEP_TRAIN>.zip
```

Run the following command to start your processing.

```bash
bash scripts/process_record3d_to_iphone.sh <SEQUENCE>
```

## (5) [if multi-camera] Processing the co-visibility masks for evaluation.

```bash
python tools/process_covisible.py \
    --gin_configs 'configs/iphone/process_covisible.gin' \
    --gin_configs 'SEQUENCE="<SEQUENCE>"'
```

## (6) (Optional) Annotating keypoints in the training video for evaluation

Please checkout our annotation tool in `tools/annotate_keypoints.ipynb`.
