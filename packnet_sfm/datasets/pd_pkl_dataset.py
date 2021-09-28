import os
import numpy as np
import pickle

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class PDPKLDataset(Dataset):
    def __init__(self, data_dir, split='pd_batch_{:3}', data_transform=None, 
                forward_context=0, back_context=0, **kwargs):
        self.data_dir = data_dir
        self.split = split
        self.forward_context = forward_context
        self.back_context = back_context
        self.with_context = (back_context != 0 or forward_context != 0)
        self.files = []
        for fn in os.listdir(self.data_dir):
            f_path = os.path.join(self.data_dir, fn)
            if self._has_context(fn):
                self.files.append(f_path)

        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _has_context(self, fn):
        # print('fn: {}'.format(fn))
        f_idx = int(fn[-7:-4])
        f_back_idx = f_idx - self.back_context
        fn_back = 'pd_batch_{}.pkl'.format(str(f_back_idx).zfill(3))
        f_forward_idx = f_idx + self.forward_context
        fn_forward = 'pd_batch_{}.pkl'.format(str(f_forward_idx).zfill(3))
        # print('fn_forward: {}'.format(fn_forward))
        # print('fn_back: {}'.format(fn_back))
        if self._path_exists(fn_back) and self._path_exists(fn_forward):
            return True
        else:
            return False

    def _path_exists(self, fn):
        return os.path.exists(os.path.join(self.data_dir, fn))

    def _read_rgb_PIL(self, idx):
        # print(f'idx: {idx}')
        f_path = os.path.join(self.data_dir, 'pd_batch_{}.pkl'.format(str(idx).zfill(3)))
        data_dict = pickle.load(open(f_path, 'rb'))
        return transforms.ToPILImage()(data_dict['rgb'][0][0])

    def __getitem__(self, idx):
        f_path = self.files[idx]
        data_dict = pickle.load(open(f_path, 'rb'))

        fn = f_path.split('/')[-1]
        f_idx = int(fn[-7:-4])

        depth_tensor = data_dict['depth'][0][0]
        D,H,W = depth_tensor.shape

        sample = {
            'idx': f_idx,
            'filename': fn,
            'rgb': self._read_rgb_PIL(idx),
            # 'rgb_context': [self._read_rgb_PIL(f_idx + delta_idx) for delta_idx in [-self.back_context, self.forward_context]],
            'intrinsics': data_dict['intrinsics'][0][0],
            'intrinsic_type': 'PD',
            'depth': data_dict['depth'][0][0].view(H,W,D).numpy(),
            # 'extrinsics': data_dict['extrinsics'][0][0],
            'pose': data_dict['pose'][0][0]
        }

        if self.with_context:
            sample['rgb_context'] = [self._read_rgb_PIL(f_idx + delta_idx) for delta_idx in [-self.back_context, self.forward_context]]

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

if __name__ == "__main__":
    data_dir = '/data/datasets/pd_pinhole'
    pd_dataset = PDPKLDataset(data_dir=data_dir,forward_context=1, back_context=1)
    print(len(pd_dataset))
    print(pd_dataset[0].keys())
    print(pd_dataset[0]['rgb'])
    print(pd_dataset[0]['rgb_context'])
    print('intrinsics')
    # print(pd_dataset[0]['intrinsics'])
    print(pd_dataset[0]['intrinsics'].shape)
    print(pd_dataset[0]['intrinsics'])
    print('depth')
    # print(pd_dataset[0]['depth'])
    print(pd_dataset[0]['depth'].shape)
    print(type(pd_dataset[0]['depth']))
    # print('extrinsics')
    # print(pd_dataset[0]['extrinsics'])
    # print(pd_dataset[0]['extrinsics'].shape)
    print('pose')
    # print(pd_dataset[0]['pose'])
    print(pd_dataset[0]['pose'].shape)

    # for i in range(len(pd_dataset)):
    #     print(i, pd_dataset[i]['idx'])
