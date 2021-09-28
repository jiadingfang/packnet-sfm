import os
import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt

from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.camera_ucm import UCMCamera
from packnet_sfm.geometry.camera_eucm import EUCMCamera
from packnet_sfm.geometry.camera_ds import DSCamera

euroc_depth_dir = '/data/datasets/results/euroc_MH_01_depth_npz'

# intrinsic_hist = np.load('intrinsic_history.npy')
evolve_intrinsics = np.load('evolving_intrinsics.npy')

# for f_name in os.listdir(euroc_depth_dir):
for i in range(300):
# for i in range(len(evolve_intrinsics)):
    print(i)
    # print(idx)
    depth_path = '/data/datasets/results/omnicam_384x384_inv_depth_npz/{}.npz'.format(str(i).zfill(9))
    # depth_path = '/data/datasets/results/euroc_MH_01_depth_npz/1403636579763555584.npz'
    # depth_path = os.path.join(euroc_depth_dir, f_name)
    # print('depth path')
    # print(depth_path)
    np_depth = np.load(depth_path)['depth']
    image_shape = np_depth.shape
    # np_depth = np.where(np_depth < 80.0, np_depth, 80.0)
    tensor_depth = torch.tensor(np_depth).unsqueeze(0).unsqueeze(1)
    tensor_depth = 1. / tensor_depth.clamp(min=1e-6) # from inv depth to depth
    # print('depth')
    # print(np_depth.shape)
    # print(np.max(np_depth))
    # print(np.min(np_depth))
    # print(np.median(np_depth))
    # print(np.sum(np.abs(np_depth) <= 80))

    # plt.hist(np_depth, bins=100)
    # plt.savefig('np_depth_hist_before_filter.png')

    # I_ucm = torch.tensor([112.11, 90.04, 514.95, 521.48, 0.18])
    I_ucm = torch.tensor([112.5976,  93.0813,  199.6491,  195.1849,  0.5])
    I_ucm = I_ucm.unsqueeze(0)
    ucm_cam = UCMCamera(I=I_ucm)

    # eucm_I = torch.tensor([235.64381137951174, 245.38803860055288, 186.44431894063212, 132.64829510142745, 0.5966287792627975, 1.1122253956511319], dtype=torch.double)
    # eucm_I = torch.tensor([237.78, 247.71, 186.66, 129.09, 0.598, 1.075]) # learned
    # print(evolve_intrinsics[i])
    # eucm_I = torch.tensor(evolve_intrinsics[i])
    # print(eucm_I)
    # eucm_I = eucm_I.unsqueeze(0)
    # eucm_cam = EUCMCamera(I=eucm_I)

    # reconstruct
    tensor_points = ucm_cam.reconstruct(tensor_depth, 'c').squeeze(0)
    # tensor_points = eucm_cam.reconstruct(tensor_depth, 'c').squeeze(0)
    np_points = np.array(tensor_points)

    # # import mask
    # mask = cv2.imread('./omnicam_mask.png', cv2.IMREAD_GRAYSCALE)
    # mask = (mask/255.).astype(np.uint8)
    # resized_mask = cv2.resize(mask, image_shape)

    # # apply mask
    # points_list = []
    # for i in range(image_shape[0]):
    #     for j in range(image_shape[1]):
    #         if resized_mask[i,j] == 1:
    #            points_list.append(np_points[:, i, j])

    # filtered_points = np.array(points_list)
    # print(filtered_points.shape)

    # save
    save_dir = '/data/datasets/results/omnicam_384x384_pc_1'
    # save_dir = '/data/datasets/results/euroc_MH_01_pointclouds'
    # save_dir = '/data/datasets/results/euroc_MH_01_pointclouds_evolve_sigmoid_0'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, '{}.npy'.format(str(i).zfill(9)))
    # save_path = os.path.join(save_dir, '1403636579763555584.npy')
    # save_path = os.path.join(save_dir, 'epoch_{}.npy'.format(str(i).zfill(3)))
    # save_path = os.path.join(save_dir, '{}.npy'.format(f_name[:-4]))
    np.save(save_path, np_points)

    # print('points')
    # np_points_x = np_points[0,:,:]
    # np_points_y = np_points[1,:,:]
    # np_points_z = np_points[2,:,:]
    # print(np_points_x.shape)
    # print(np_points[:,:5,:5])
    # print(np.max(np_points_z))
    # print(np.median(np_points_z))
    # print(np.min(np_points_z))
    # print(np.sort(np.unique(np_points_z)))
    # print(np.sum(np.abs(np_points_z) <= 80))

    # print(np.max(np_points_x))
    # print(np.median(np_points_x))
    # print(np.min(np_points_x))

    # print(np.max(np_points_y))
    # print(np.median(np_points_y))
    # print(np.min(np_points_y))

    # plt.hist(np_points_x, bins=100)
    # plt.savefig('pointcloud_x_hist.png')