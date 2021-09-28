# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os
import torch
from torchvision.utils import save_image
from PIL import Image

from glob import glob
from cv2 import imwrite, resize
import matplotlib.pyplot as plt

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.geometry.pose_utils import pose_vec2mat
from packnet_sfm.geometry.camera_utils import view_synthesis
from packnet_sfm.geometry.pose import Pose

from packnet_sfm.geometry.camera import Camera
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

def warp_ref_image(inv_depth, ref_image, I, ref_I, pose):
    """
    Warps a reference image to produce a reconstruction of the original one.

    Parameters
    ----------
    inv_depths : torch.Tensor [B,1,H,W]
        Inverse depth map of the original image
    ref_image : torch.Tensor [B,3,H,W]
        Reference RGB image
    I : torch.Tensor [B,3,3]
        Original camera intrinsics
    ref_I : torch.Tensor [B,3,3]
        Reference camera intrinsics
    pose : Pose
        Original -> Reference camera transformation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image (reconstructing the original one)
    """
    B, _, H, W = ref_image.shape
    device = ref_image.get_device()
    # Generate cameras for all scales
    _, _, DH, DW = inv_depth.shape
    scale_factor = DW / float(W)
    eucm_cam_src = EUCMCamera(I=I).to(device)
    eucm_cam_ref = EUCMCamera(I=ref_I, Tcw=pose).to(device)
    # View synthesis
    depth = inv2depth(inv_depth)
    ref_warped = view_synthesis(ref_image, depth, eucm_cam_ref, eucm_cam_src, padding_mode='zeros')
    # Return warped reference image
    return ref_warped


@torch.no_grad()
def infer_pose(f_ref, f_src, model_wrapper, image_shape, mask):
    """
    Produce the pose and the warped ref image from a ref and a src image

    Parameters
    ===========
    f_ref : str
        reference image file
    f_src : str
        source image file
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    mask : binary
        mask for the image
    """

    # Load ref and src image
    ref_image = load_image(f_ref)
    src_image = load_image(f_src)
    # Resize and to tensor
    ref_image = resize_image(ref_image, image_shape)
    ref_image = to_tensor(ref_image).unsqueeze(0)
    src_image = resize_image(src_image, image_shape)
    src_image = to_tensor(src_image).unsqueeze(0)

    ref_image = ref_image.detach().cpu().numpy() # artificailly make gray image rgb
    ref_image = np.tile(ref_image, (1,3,1,1))
    ref_image = torch.tensor(ref_image)

    src_image = src_image.detach().cpu().numpy() # artificailly make gray image rgb
    src_image = np.tile(src_image, (1,3,1,1))
    src_image = torch.tensor(src_image)

    save_image(ref_image, 'scripts/ref_img.png')
    save_image(src_image, 'scripts/src_img.png')

    # print('ref shape')
    # print(ref_image.shape)
    # print('src image')
    # print(src_image.shape)

    # Send image to GPU if available
    if torch.cuda.is_available():
        ref_image = ref_image.to('cuda:{}'.format(0), dtype=torch.float)
        src_image = src_image.to('cuda:{}'.format(0), dtype=torch.float)

    # infer depth
    pred_inv_depth = model_wrapper.depth(ref_image)[0]
    print('inv depth shape')
    print(pred_inv_depth.shape)
    viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
    print(viz_pred_inv_depth.shape)
    print(type(viz_pred_inv_depth))
    imwrite('scripts/pred_inv_depth.png', viz_pred_inv_depth[:, :, ::-1])

    # infer pose
    # pred_inv_depth = model_wrapper.depth(image)[0]
    src_images = [src_image]
    pose_vec = model_wrapper.pose(ref_image, src_images)[0]
    print('pose vec')
    print(pose_vec.shape)

    pose_mat_3x4 = pose_vec2mat(pose_vec)
    pose_mat_3x4 = pose_mat_3x4.detach().cpu().numpy()
    print('pose_mat')
    print(pose_mat_3x4)
    print(pose_mat_3x4.shape)

    pose_mat_4x4 = np.zeros((1,4,4))
    pose_mat_4x4[:,:3,:] = pose_mat_3x4
    pose_mat_4x4[:,3,3] = 1
    pose_mat_4x4 = torch.tensor(pose_mat_4x4, dtype=torch.float)
    print('pose mat 4x4')
    print(pose_mat_4x4)

    pose = Pose(pose_mat_4x4)
    print(pose)

    eucm_I = torch.tensor([235.64381137951174, 245.38803860055288, 186.44431894063212, 132.64829510142745, 0.5966287792627975, 1.1122253956511319])
    # eucm_I = torch.tensor([237.78, 247.71, 186.66, 129.09, 0.598, 1.075]) # learned
    eucm_I = eucm_I.unsqueeze(0)
    # pred_inv_depth = pred_inv_depth.to(torch.double)
    # ref_image = ref_image.to(torch.double)
    warped_ref_image = warp_ref_image(pred_inv_depth, ref_image, eucm_I, eucm_I, pose)
    print('warped image')
    print(warped_ref_image.shape)
    save_image(warped_ref_image, 'scripts/warped_ref_image.png')

    photo_loss = torch.abs(src_image - warped_ref_image)
    print('photo_loss')
    print(photo_loss.shape)
    photo_loss = photo_loss[0,0,:,:] # since the 3 channels are duplicate
    print(photo_loss.shape)
    photo_loss = photo_loss.detach().cpu().numpy()
    cm = plt.get_cmap('viridis')
    colored_photo_loss = cm(photo_loss)
    colored_photo_loss = Image.fromarray((colored_photo_loss[:, :, :3] * 255).astype(np.uint8))
    colored_photo_loss.save('scripts/photo_loss.png')


    # save_image(photo_loss, 'scripts/photo_loss.png')




@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, mask, half, save):
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
    mask : binary image
        mask for the image
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

    # Load image
    image = load_image(input_file)
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # resize mask
    # resized_mask = resize(mask, image_shape)
    # resized_mask = resized_mask.astype(int).astype(float)
    # # resized_mask = mask
    # resized_mask = torch.tensor(resized_mask)
    # resized_mask = resized_mask.unsqueeze(0).unsqueeze(1)
    # print(resized_mask.shape)
    
    # Send image to GPU if available
    if torch.cuda.is_available():
        # image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        image = image.to('cuda:{}'.format(0), dtype=dtype)
        # resized_mask = resized_mask.to('cuda:{}'.format(0), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]
    # print(pred_inv_depth.shape)

    # apply mask
    # pred_inv_depth = pred_inv_depth * resized_mask[:,0,:,:]

    # # get depth from inv_depth
    # pred_depth=inv2depth(pred_inv_depth)

    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        # write_depth(filename, depth=inv2depth(pred_inv_depth))
        write_depth(filename, depth=pred_inv_depth)
    else:
        # Prepare RGB image
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        image = np.concatenate([rgb, viz_pred_inv_depth], 0)
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
    
    # # Process each file
    # for fn in files[rank()::world_size()]:
    #     infer_and_save_depth(
    #         fn, args.output, model_wrapper, image_shape, mask, args.half, args.save)

    # infer pose and warped image
    # for i in range(len(files) - 1):
    for i in range(1):
        f_ref = files[i]
        print('ref')
        print(f_ref)
        f_src = files[i+1]
        print('src')
        print(f_src)
        infer_pose(f_ref, f_src, model_wrapper, image_shape, mask)


if __name__ == '__main__':
    args = parse_args()
    main(args)
