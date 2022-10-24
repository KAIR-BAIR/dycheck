# Benchmark

Please find our best performing checkpoints used for reality checks at [here](https://drive.google.com/drive/u/0/folders/1ZYQQh0qkvpoGXFIcK_j4suon1Wt6MXdZ), including:

1. Models trained without camera teleportation on Nerfies-HyperNeRF dataset.
2. Models trained with camera teleportation on Nerfies-HyperNeRF dataset to ensure that we match the results from offcial implementations.
3. Models trained without camera teleportation on iPhone dataset, with all the enhancements that we find helpful.

We consider the following models in our benchmark:

1. Time-conditioned NeRF (T-NeRF);
2. [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields/), Li et al., CVPR 2021;
3. [Nerfies](https://github.com/google/nerfies), Park et al., ICCV 2021;
4. [HyperNeRF](https://github.com/google/hypernerf), Park et al., SIGGRAPH Asia 2021;

This repo contains implementations for T-NeRF, Nerfies and HyperNeRF. We plan to release our NSFF's checkpoints, training and evaluation code, please stay tuned.

All chekcpoints are assumed to be organized as `<DYCHECK_ROOT>/work_dirs/<DATASET>/<SEQUENCE>/<MODEL>/<SETTING>`.

While we have reported exhaustive per-sequence breakdowns in the appendix of
our paper, we include some main results below to be self-contained. We also
provide the evaluation and training commands to reproduce our results.

## (0) Prerequisite

### (A) Checkpoints

Due to their sheer volumes (in total we released 84 models), we recommend our users to directly download them through `rclone`.

We prepared a simple download script for you to set up data from scratch:

```bash
# Download all of our checkpoints.
bash scripts/download_all_checkpoints.sh <DRIVE_NAME>
```

You can also just download a single model to get started:

```bash
# Download a checkpoint of your choice. `<MODEL>` and `<SETTING>` are discussed
# next.
bash scripts/download_all_checkpoints.sh <DRIVE_NAME> \
    <DATASET> <SEQUENCE> <MODEL> <SETTING>
```

### (B) Configs

We use [`gin-config`](https://github.com/google/gin-config) to organize and configurate our experments. Please refer to our [configs](../configs) for details. The configs are organized as `<DYCHECK_ROOT>/configs/<DATASET>/<MODEL>/<SETTING>`.

- Model names (with their `<MODEL>` aliases):
  - T-NeRF (`tnerf`);
  - Nerfies (`dense`);
  - HyperNeRF (`ambient`);
- Setting names (with their `<SETTING>` aliases):
  - For Nerfies-HyperNeRF dataset, we have two settings:
    - Training with teleporting camera (`intl`);
    - Training with non-teleporting cameras (`mono`);
  - For iPhone dataset, we have three settings, all with non-teleporting camera:
    - The original model out-of-the-box (`base`);
    - With random background augmentation (`randbkgd`);
    - With random background augmentation, and depth supervision (`randbkgd_depth`);
    - With random background augmentation, depth supervision, and sparsity regularization (`randbkgd_depth_dist`);

### (C) Training time statistics

All models take 4 NVIDIA RTX A5000 GPUs to train by our default configs. T-NeRF takes about 12 hours to finish, while Nerfies and HyperNeRF take about 24 hours.

## (1) Nerfies-HyperNeRF dataset

Please see our paper for full results. Here, we report a part of masked image metrics (mPSNR, mLPIPS) and correspondence accuracy (PCK-T). All results are trained under non-teleporting setting.

| mPSNR     | mean  | broom | curls | tail  | toby-sit | 3dprinter | chicken | peel-banana |
| --------- | :---: | :---: | :---: | :---: | :------: | :-------: | :-----: | :---------: |
| T-NeRF    | 21.56 | 20.04 | 21.86 | 22.56 |  18.53   |   19.77   |  25.54  |    22.64    |
| NSFF      | 19.53 | 20.36 | 18.74 | 21.94 |  18.66   |   16.89   |  21.47  |    18.68    |
| Nerfies   | 20.85 | 19.34 | 23.28 | 21.46 |  18.45   |   19.67   |  23.78  |    19.97    |
| HyperNeRF | 21.13 | 19.04 | 23.13 | 21.54 |  18.40   |   19.58   |  24.90  |    21.34    |

| mLPIPS    | mean  | broom | curls | tail  | toby-sit | 3dprinter | chicken | peel-banana |
| --------- | :---: | :---: | :---: | :---: | :------: | :-------: | :-----: | :---------: |
| T-NeRF    | 0.297 | 0.59  | 0.284 | 0.305 |  0.421   |   0.203   |  0.131  |    0.142    |
| NSFF      | 0.471 | 0.776 | 0.378 | 0.522 |   0.6    |   0.443   |  0.29   |    0.293    |
| Nerfies   |  0.2  | 0.294 | 0.22  | 0.213 |  0.249   |   0.148   |  0.114  |    0.161    |
| HyperNeRF | 0.192 | 0.279 | 0.22  | 0.218 |  0.242   |   0.147   |  0.101  |    0.135    |

| PCK-T     | mean  | broom | curls | tail  | toby-sit | 3dprinter | chicken | peel-banana |
| --------- | :---: | :---: | :---: | :---: | :------: | :-------: | :-----: | :---------: |
| T-NeRF    |   -   |   -   |   -   |   -   |    -     |     -     |    -    |      -      |
| NSFF      | 0.422 | 0.119 | 0.212 | 0.323 |  0.666   |   0.797   |  0.604  |    0.233    |
| Nerfies   | 0.756 | 0.460 | 0.782 | 0.645 |  0.914   |   0.998   |  0.978  |    0.514    |
| HyperNeRF | 0.764 | 0.471 | 0.838 | 0.623 |  0.883   |   0.994   |  1.000  |    0.540    |

To reproduce our results, please follow the commands below. To evaluate our released checkpoints, make sure that you have download them first.

```bash
# <DATASET> in [nerfies, hypernerf]
# <MODEL> in [tnerf, dense, ambient]
# <SEQUENCE> in [broom, curls, tail, toby-sit] if <DATASET> == nerfies
# <SEQUENCE> in [3dprinter, chicken, peel-banana] if <DATASET> == hypernerf

# Evaluate a trained model with all the metrics.
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/launch.py \
    --gin_configs "configs/<DATASET>/<MODEL>/mono.gin" \
    --gin_bindings "Config.engine_cls=@Evaluation" \
    --gin_bindings 'SEQUENCE="'"<SEQUENCE>"'"'

# Train model from scratch.
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/launch.py \
    --gin_configs "configs/<DATASET>/<MODEL>/mono.gin" \
    --gin_bindings "Config.engine_cls=@Trainer" \
    --gin_bindings 'SEQUENCE="'"<SEQUENCE>"'"'
```

For example, to evaluate the HyperNeRF model trained on `peel-banana`:

```bash
# Download the checkpoint. Skip it if you have downloaded all checkpoints.
bash scripts/download_single_checkpoint.sh <DRIVE_NAME> \
    hypernerf peel-banana ambient mono

# Evaluate the checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/launch.py \
    --gin_configs "configs/hypernerf/ambient/mono.gin" \
    --gin_bindings "Config.engine_cls=@Evaluation" \
    --gin_bindings 'SEQUENCE="'"peel-banana"'"'
```

<details>
    <summary>Click to see results on the original teleporting setting (which match the results from Nerfies and HyperNeRF official repos).</summary>

Please see our paper for full results. Here, we report PSNR and LPIPS.

| PSNR      | mean  | broom | curls | tail  | toby-sit | 3dprinter | chicken | peel-banana |
| --------- | :---: | :---: | :---: | :---: | :------: | :-------: | :-----: | :---------: |
| T-NeRF    | 22.11 | 20.51 | 22.62 | 23.09 |  18.53   |   19.99   |  26.25  |    23.78    |
| NSFF      | 21.71 | 21.33 | 18.50 | 23.42 |  21.27   |   20.24   |  24.44  |    22.76    |
| Nerfies   | 21.71 | 19.70 | 24.04 | 21.79 |  18.48   |   20.30   |  26.54  |    21.11    |
| HyperNeRF | 22.09 | 19.36 | 24.59 | 22.16 |  18.41   |   20.12   |  27.74  |    22.25    |

| LPIPS     | mean  | broom | curls | tail  | toby-sit | 3dprinter | chicken | peel-banana |
| --------- | :---: | :---: | :---: | :---: | :------: | :-------: | :-----: | :---------: |
| T-NeRF    | 0.321 | 0.602 | 0.327 | 0.349 |  0.477   |   0.195   |  0.130  |    0.165    |
| NSFF      | 0.495 | 0.747 | 0.448 | 0.606 |  0.637   |   0.449   |  0.268  |    0.315    |
| Nerfies   | 0.217 | 0.296 | 0.245 | 0.236 |  0.375   |   0.115   |  0.079  |    0.174    |
| HyperNeRF | 0.209 | 0.314 | 0.247 | 0.231 |  0.339   |   0.110   |  0.077  |    0.144    |

```bash
# <DATASET> in [nerfies, hypernerf]
# <MODEL> in [tnerf, dense, ambient]
# <SEQUENCE> in [broom, curls, tail, toby-sit] if <DATASET> == nerfies
# <SEQUENCE> in [3dprinter, chicken, peel-banana] if <DATASET> == hypernerf

# Train model from scratch.
python tools/launch.py \
    --gin_configs "configs/<DATASET>/<MODEL>/intl.gin" \
    --gin_bindings "Config.engine_cls=@Trainer" \
    --gin_bindings 'SEQUENCE="'"<SEQUENCE>"'"'

# Evaluate a trained model with all the metrics.
python tools/launch.py \
    --gin_configs "configs/<DATASET>/<MODEL>/intl.gin" \
    --gin_bindings "Config.engine_cls=@Evaluation" \
    --gin_bindings 'SEQUENCE="'"<SEQUENCE>"'"'
```

For example, to evaluate the HyperNeRF model trained on `peel-banana`:

```bash
# Download the checkpoint. Skip it if you have downloaded all checkpoints.
bash scripts/download_single_checkpoint.sh <DRIVE_NAME> \
    hypernerf peel-banana ambient intl

# Evaluate the checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/launch.py \
    --gin_configs "configs/hypernerf/ambient/intl.gin" \
    --gin_bindings "Config.engine_cls=@Evaluation" \
    --gin_bindings 'SEQUENCE="'"peel-banana"'"'
```

</details>

## iPhone dataset

Please see our paper for full results. Here, we again report a part of masked image metrics (mPSNR, mLPIPS) and correspondence accuracy (PCK-T). All results are trained under non-teleporting setting.

We report the performance with the best configuration that we found on iPhone dataset (`randbkgd_depth_dist`).

| mPSNR     | mean  | broom | curls | tail  | toby-sit | 3dprinter | chicken | peel-banana |
| --------- | :---: | :---: | :---: | :---: | :------: | :-------: | :-----: | :---------: |
| T-NeRF    | 21.56 | 20.04 | 21.86 | 22.56 |  18.53   |   19.77   |  25.54  |    22.64    |
| NSFF      | 19.53 | 20.36 | 18.74 | 21.94 |  18.66   |   16.89   |  21.47  |    18.68    |
| Nerfies   | 20.85 | 19.34 | 23.28 | 21.46 |  18.45   |   19.67   |  23.78  |    19.97    |
| HyperNeRF | 21.13 | 19.04 | 23.13 | 21.54 |  18.40   |   19.58   |  24.90  |    21.34    |

| mLPIPS    | mean  | apple | block | paper-windmill | space-out | spin  | teddy | wheel |
| --------- | :---: | :---: | :---: | :------------: | :-------: | :---: | :---: | :---: |
| T-NeRF    | 0.379 | 0.508 | 0.346 |     0.258      |   0.377   | 0.443 | 0.429 | 0.292 |
| NSFF      | 0.396 | 0.414 | 0.438 |     0.348      |   0.341   | 0.371 | 0.527 | 0.331 |
| Nerfies   | 0.339 | 0.478 | 0.389 |     0.211      |   0.303   | 0.309 | 0.372 | 0.310 |
| HyperNeRF | 0.332 | 0.478 | 0.331 |     0.209      |   0.320   | 0.325 | 0.350 | 0.310 |

| PCK-T     | mean  | apple | block | paper-windmill | space-out | spin  | teddy | wheel |
| --------- | :---: | :---: | :---: | :------------: | :-------: | :---: | :---: | :---: |
| T-NeRF    |   -   |   -   |   -   |       -        |     -     |   -   |   -   |   -   |
| NSFF      | 0.256 | 0.132 | 0.180 |     0.163      |   0.598   | 0.083 | 0.291 | 0.346 |
| Nerfies   | 0.453 | 0.599 | 0.274 |     0.113      |   0.812   | 0.177 | 0.801 | 0.394 |
| HyperNeRF | 0.400 | 0.318 | 0.216 |     0.107      |   0.859   | 0.115 | 0.775 | 0.408 |

```bash
# <MODEL> in [tnerf, dense, ambient]
# <SEQUENCE> in [apple, block, paper-windmill, space-out, spin, teddy, wheel]

# Evaluate a trained model with all the metrics.
python tools/launch.py \
    --gin_configs "configs/iphone/<MODEL>/randbkgd_depth_dist.gin" \
    --gin_bindings "Config.engine_cls=@Evaluation" \
    --gin_bindings 'SEQUENCE="'"<SEQUENCE>"'"'

# Train model from scratch.
python tools/launch.py \
    --gin_configs "configs/iphone/<MODEL>/randbkgd_depth_dist.gin" \
    --gin_bindings "Config.engine_cls=@Trainer" \
    --gin_bindings 'SEQUENCE="'"<SEQUENCE>"'"'
```

For example, to evaluate the HyperNeRF model trained on `teddy`:

```bash
# Download the checkpoint. Skip it if you have downloaded all checkpoints.
bash scripts/download_single_checkpoint.sh <DRIVE_NAME> \
    iphone teddy ambient randbkgd_depth_dist

# Evaluate the checkpoint.
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/launch.py \
    --gin_configs "configs/iphone/ambient/randbkgd_depth_dist.gin" \
    --gin_bindings "Config.engine_cls=@Evaluation" \
    --gin_bindings 'SEQUENCE="'"teddy"'"'
```
