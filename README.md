This is the repo for reproducing the depth results in our "Self-Supervised Camera Self-Calibration from Video" paper.

## Install
The same as the original packnet-sfm repo.

## Usage
### Data
#### Download

For KITTI data, follow the original packnet-sfm repo. 
The EUROC data can be downloaded from [here](https://www.dropbox.com/sh/112wtkmxapesws5/AADawGf7QaCvlTiL8klpH7Uya?dl=0). It contains the the `cam0` data for euroc sequences. It also contains the Basalt calibrated results for UCM, EUCM, DS camera models.

#### Mount

Please mount the data under `/data/datasets/`. You can add to the Makefile
```bash
-v /your/data/dir/:/data/datasets/
```
The script will then read `/data/datasets/euroc_cam` for example.

### Training

The repo supports learning UCM, EUCM, DS camera models on datasets KITTI and EUROC.
```bash
# UCM on tss
make docker-run COMMAND="python3 scripts/train.py configs/ucm_tss.yaml"
# UCM on KITTI
make docker-run COMMAND="python3 scripts/train.py configs/ucm_kitti.yaml"
# UCM on EUROC
make docker-run COMMAND="python3 scripts/train.py configs/ucm_euroc.yaml"
```
#### Intialization

The initialization for the camera model can be set in the config file under `init_intrinsic` field (currently only UCM model is supported). If it's not set, the default value is $(sigmoid(-1) * 1000)$ for ${fx, fy, cx, cy}$ and $sigmoid(-1)$ for alpha.

#### Augmentation

There are 3 augmentations implemented: center_crop, random_hflip (random horizontal flipping), random_vflip (random vertical flipping). For center_crop, specific shape can be set in the config file. For random_hflip and random_hflip, they can be turned on and off in the config file, the default is on.

### Inference

use `scripts/infer.py` to do the inference with checkpoint.
```bash
# Format
make docker-run COMMAND="python3 scripts/infer.py --checkpoint /path/to/ckpt/file --input /path/to/images --output /path/to/saving/folder"

# Example inference on EUROC (gray scale images)
make docker-run COMMAND="python3 scripts/infer.py --checkpoint /data/datasets/pretrained_ckpts/euroc_141.ckpt --input /data/datasets/euroc/MH_01_easy/mav0/cam0/data/ --output /data/datasets/results/euroc --gray"
```

### Logging

Enable Wandb if you can (the instructions are in the original Packnet-sfm repo). It provides better visualization and parameter logging. Otherwise, the terminal will also print out predicted intrinsics every 100 steps as well as depth evaluation results every epoch.

## Known issues

1. nan error when alpha is very close to 1. This is because we did not implement the valid mask mentioned in the [double sphere paper](https://arxiv.org/abs/1807.08957).The difficulty is inside the [PyTorch design](https://github.com/pytorch/pytorch/issues/15506). So even if we apply the mask to filter out nan values in the forward pass, the nan is still there in the backward step. However, this issue does not appear in most of our experiments. The method works without a problem on datasets like KITTI, EUROC. It is only a problem when the alpha value in the UCM, EUCM, DS models are reaching 0.8 or above like the omnicam dataset. One way to get around it is to constrain alpha between [0, 0.5] when all pixels are valid.
2. The training procedure may be subject to some hyper-parameter or random seed tuning. The qualitative depth results on EUROC dataset are not the best even with fixed Basalt-calibrated intrinsics.