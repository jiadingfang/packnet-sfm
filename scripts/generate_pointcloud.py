import os
import numpy as np
import torch
import cv2

import matplotlib.pyplot as plt

from packnet_sfm.geometry.camera_ucm import UCMCamera

for idx in range(300):
    # print('idx')
    print(idx)
    depth_path = '/data/datasets/results/omnicam_1024x1024_npz/{}.npz'.format(str(idx).zfill(9))
    np_depth = np.load(depth_path)['depth']
    image_shape = np_depth.shape
    # np_depth = np.where(np_depth < 80.0, np_depth, 80.0)
    tensor_depth = torch.tensor(np_depth).unsqueeze(0).unsqueeze(1)
    # print('depth')
    # print(np_depth.shape)
    # print(np.max(np_depth))
    # print(np.min(np_depth))
    # print(np.median(np_depth))
    # print(np.sum(np.abs(np_depth) <= 80))

    # plt.hist(np_depth, bins=100)
    # plt.savefig('np_depth_hist_before_filter.png')

    I_ucm = torch.tensor([112.11, 90.04, 514.95, 521.48, 0.018])
    I_ucm = I_ucm.unsqueeze(0)
    ucm_camera = UCMCamera(I=I_ucm)

    # reconstruct
    tensor_points = ucm_camera.reconstruct(tensor_depth, 'c').squeeze(0)
    np_points = np.array(tensor_points)

    # import mask
    mask = cv2.imread('./omnicam_mask.png', cv2.IMREAD_GRAYSCALE)
    mask = (mask/255.).astype(np.uint8)
    resized_mask = cv2.resize(mask, image_shape)

    # apply mask
    points_list = []
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if resized_mask[i,j] == 1:
               points_list.append(np_points[:, i, j])

    filtered_points = np.array(points_list)
    print(filtered_points.shape)

    # save
    save_dir = '/data/datasets/results/omnicam_1024x1024_pointclouds'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, '{}.npy'.format(str(idx).zfill(9)))
    np.save(save_path, filtered_points)

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