import argparse
import torch

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.models.model_checkpoint import ModelCheckpoint
from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_train_file
from packnet_sfm.utils.load import set_debug, filter_args_create
from packnet_sfm.utils.horovod import hvd_init, rank
from packnet_sfm.loggers import WandbLogger

def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM inspect chheckpoint script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args

def read_ckpt(file):
    config, ckpt = parse_train_file(file)
    # print('config')
    # print(config)
    print('ckpt')
    # print(ckpt.keys())
    # print(ckpt['state_dict'].keys())
    I = ckpt['state_dict']['model.depth_net.intrinsic_decoder.intrinsic_vector']
    print(I)
    fx, fy, cx, cy = torch.sigmoid(I[0:4]) * 1000
    alpha = torch.sigmoid(I[4]) * 1/2
    print(fx,fy,cx,cy,alpha)

    # config.datasets.augmentation.center_crop = False
    # config.datasets.augmentation.random_hflip = True
    # config.datasets.augmentation.random_vflip = True
    # # config.wandb.dry_run = False
    # config.wandb.name = 'fine_tunning_euroc_173'
    # config.arch.max_epochs = 100
    # config.gpu.idx = 2

if __name__ == '__main__':
    args = parse_args()
    read_ckpt(args.file)