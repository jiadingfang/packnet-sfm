# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image

from packnet_sfm.utils.misc import filter_dict

########################################################################################################################

def resize_image(image, shape, interpolation=Image.ANTIALIAS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2)

def resize_sample_image_and_intrinsics(sample, shape,
                                       image_interpolation=Image.ANTIALIAS):
    """
    Resizes the image and intrinsics of a sample

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample['rgb'].size
    (out_h, out_w) = shape
    # Scale intrinsics
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        intrinsics = np.copy(sample[key])
        intrinsics[0] *= out_w / orig_w
        intrinsics[1] *= out_h / orig_h
        sample[key] = intrinsics
    # Scale images
    for key in filter_dict(sample, [
        'rgb', 'rgb_original',
    ]):
        sample[key] = image_transform(sample[key])
    # Scale context images
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original',
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]
    # Return resized sample
    return sample

def resize_sample(sample, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and intrinsics
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)
    # Resize depth maps
    for key in filter_dict(sample, [
        'depth',
    ]):
        sample[key] = resize_depth(sample[key], shape)
    # Resize depth contexts
    for key in filter_dict(sample, [
        'depth_context',
    ]):
        sample[key] = [resize_depth(k, shape) for k in sample[key]]
    # Return resized sample
    return sample

########################################################################################################################

def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    # Return converted sample
    return sample

########################################################################################################################

def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    # Duplicate single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    # Duplicate lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    # Return duplicated sample
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.

    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    prob : float
        Jittering probability

    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare transformation
        color_augmentation = transforms.ColorJitter()
        brightness, contrast, saturation, hue = parameters
        augment_image = color_augmentation.get_params(
            brightness=[max(0, 1 - brightness), 1 + brightness],
            contrast=[max(0, 1 - contrast), 1 + contrast],
            saturation=[max(0, 1 - saturation), 1 + saturation],
            hue=[-hue, hue])
        # Jitter single items
        for key in filter_dict(sample, [
            'rgb'
        ]):
            sample[key] = augment_image(sample[key])
        # Jitter lists
        for key in filter_dict(sample, [
            'rgb_context'
        ]):
            sample[key] = [augment_image(k) for k in sample[key]]
    # Return jittered (?) sample
    return sample

########################################################################################################################

def random_flip_sample(sample, prob=0.5):
    hflipper = transforms.RandomHorizontalFlip(p=1) # make a determined hflip
    vflipper = transforms.RandomVerticalFlip(p=1) # make a determined vflip

    p1, p2 = torch.rand(1), torch.rand(1) # make two guesses
    if p1 < prob and p2 < prob: 
        flipper = transforms.Compose([])
    elif p1 >= prob and p2 < prob:
        flipper = transforms.Compose([hflipper])
    elif p1 < prob and p2 >= prob:
        flipper = transforms.Compose([vflipper])
    else:
        flipper = transforms.Compose([hflipper, vflipper])

    # Flip single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample[key] = flipper(sample[key])
    # Flip lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample[key] = [flipper(k) for k in sample[key]]
    # Return flipped sample
    return sample

    ########################################################################################################################

def center_crop_sample(sample, crop_shape):
    cropper = transforms.CenterCrop(size=crop_shape)

    # Crop single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample[key] = cropper(sample[key])
    # Crop lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample[key] = [cropper(k) for k in sample[key]]
    # Return cropped sample
    return sample
