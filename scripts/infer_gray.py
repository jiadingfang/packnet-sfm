# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch

from glob import glob
from cv2 import imread, imwrite, resize

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.utils.rectify import to_perspective, warp_img

from packnet_sfm.geometry.camera_ucm import UCMCamera
from packnet_sfm.geometry.camera_eucm import EUCMCamera
from packnet_sfm.geometry.camera_ds import DSCamera


def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) and is_image(args.output)) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args


@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, mask, pt_depth_dir, half, save):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file : str
        Image file
    output_file : str
        Output file, or folder where the output will be saved
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    print('image shape')
    print(image_shape)
    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Prepare RGB image
    gray = image.detach().cpu().numpy()
    rgb_np = np.tile(gray, (1,3,1,1))
    image = torch.tensor(rgb_np) # artificailly make gray image 3 channels

    # resize mask
    resized_mask = resize(mask, image_shape)
    resized_mask = resized_mask.astype(int).astype(float)
    # resized_mask = mask
    resized_mask = torch.tensor(resized_mask)
    resized_mask = resized_mask.unsqueeze(0).unsqueeze(1)
    # print(resized_mask.shape)
    
    # Send image to GPU if available
    if torch.cuda.is_available():
        # image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        image = image.to('cuda:{}'.format(0), dtype=dtype)
        resized_mask = resized_mask.to('cuda:{}'.format(0), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]
    # print(pred_inv_depth.shape)

    # get depth from inv_depth
    pred_depth=inv2depth(pred_inv_depth)
    # print(pred_depth)
    # pred_depth = pred_depth.detach().cpu().numpy()

    # deptht rescale according to pointcloud depth
    # if pt_depth_dir is not None:
    #     pt_depth_path = os.path.join(pt_depth_dir, 'depth_0' + output_file.split('/')[-1].split('.')[0] + '.npy')
    #     pt_depth = np.load(pt_depth_path)
    #     ratio = np.median(pt_depth[pt_depth>0]) / np.median(pred_depth[pred_depth>0])
    #     print('ratio: {}'.format(ratio))
    #     pred_depth *= ratio

    # apply mask
    # pred_inv_depth = pred_inv_depth * resized_mask[:,0,:,:]
    # pred_depth = pred_depth * resized_mask[:,0,:,:].numpy()

    # generate recitified images
    # ucm_I = torch.tensor([235.4,  245.1,  186.5,  132.6,  0.650]) # gt
    ucm_I = torch.tensor([236.1,  246.9,  178.3,  146.7,  0.635]) # learned
    ucm_I = ucm_I.unsqueeze(0)

    # eucm_I = torch.tensor([235.64381137951174, 245.38803860055288, 186.44431894063212, 132.64829510142745, 0.5966287792627975, 1.1122253956511319], dtype=torch.double)
    eucm_I = torch.tensor([237.78, 247.71, 186.66, 129.09, 0.598, 1.075]) # learned
    eucm_I = eucm_I.unsqueeze(0)

    # ds_I = torch.tensor([183.9, 191.5, 186.7, 132.8, -0.208, 0.560]) # gt
    ds_I = torch.tensor([187.2, 195.9, 188.6, 138.9, -0.227, 0.569]) # learned
    ds_I = ds_I.unsqueeze(0)

    ucm_cam = UCMCamera(I=ucm_I)
    eucm_cam = EUCMCamera(I=eucm_I)
    ds_cam = DSCamera(I=ds_I)

    rgb_np = rgb_np.squeeze(0)
    rgb_np = np.moveaxis(rgb_np, 0, -1)
    # image_np = resize(rgb_np, (384,256))
    rectified_image = to_perspective(ds_cam, rgb_np, img_size=image_shape, f=0.8)

    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        # write_depth(filename, depth=pred_inv_depth)
        write_depth(filename, depth=pred_depth)
    else:
        # Prepare RGB image
        # gray = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # rgb = np.tile(gray, (1,1,3)) # artificailly make gray image 3 channels
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        rectified_rgb = rectified_image * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        # image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # image = np.concatenate([rgb, viz_pred_inv_depth, rectified_rgb], 0) # vertical
        image = np.concatenate([rgb, viz_pred_inv_depth, rectified_rgb], 1) #hohrizontal
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        imwrite(output_file, image[:, :, ::-1])


def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(0), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()

    if os.path.isdir(args.input):
        # If input file is a folder, search for image files
        files = []
        for ext in ['png', 'jpg']:
            files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
        files.sort()
        print0('Found {} files'.format(len(files)))
    else:
        # Otherwise, use it as is
        files = [args.input]

    # load omnicam mask
    # mask = np.load('omnicam_mask.npy')
    mask = np.load('mask_resized.npy')

    # load point cloud depth
    # pt_depth_dir = '/data/datasets/pt_depth_npy_1024x1024'
    pt_depth_dir = None
    
    # Process each file
    for fn in files[rank()::world_size()]:
        infer_and_save_depth(
            fn, args.output, model_wrapper, image_shape, mask, pt_depth_dir, args.half, args.save)


if __name__ == '__main__':
    args = parse_args()
    main(args)
