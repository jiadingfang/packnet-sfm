# Copyright 2020 Toyota Research Institute.  All rights reserved.
import math
import cv2
import numpy as np
import random
from PIL import Image
from typing import Tuple, List, Optional

import torch
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

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

def random_hflip_sample(sample, prob=0.5):
    # get image size
    w, h = sample['rgb'].size

    hflipper = transforms.RandomHorizontalFlip(p=1) # make a determined hflip
    
    p = torch.rand(1) # make a guess
    if p < prob:
        flipper = transforms.Compose([])
        sample['intrinsic_transform'] = intrinsic_transform()
    else:
        flipper = transforms.Compose([hflipper])
        sample['intrinsic_transform'] = intrinsic_transform(mode='horizontal_flipping', w=w, h=h)

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

def random_vflip_sample(sample, prob=0.5):
    # get image size
    w, h = sample['rgb'].size

    vflipper = transforms.RandomVerticalFlip(p=1) # make a determined vflip

    p = torch.rand(1) # make a guess
    if p < prob:
        flipper = transforms.Compose([])
        sample['intrinsic_transform'] = intrinsic_transform()
    else:
        flipper = transforms.Compose([vflipper])
        sample['intrinsic_transform'] = intrinsic_transform(mode='vertical_flipping', w=w, h=h)

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

########################################################################################################################

def init_intrinsic_transform_sample(sample):
    sample['intrinsic_transform'] = intrinsic_transform()
    return sample

def random_crop_sample(sample, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = img.size
        area = height * width
        original_ratio = width / height

        log_ratio = torch.log(torch.tensor(ratio))
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = original_ratio * torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        dw = int(round(math.sqrt(target_area * aspect_ratio)))
        dh = int(round(math.sqrt(target_area / aspect_ratio)))

        if not (0 < dw <= width and 0 < dh <= height):
            i = 0
            j = 0
            dh = height
            dw = width
            return i,j,dh,dw
        else:
            i = torch.randint(0, height - dh + 1, size=(1, )).item()
            j = torch.randint(0, width - dw + 1, size=(1, )).item()
            return i,j,dh,dw

    img = sample['rgb']
    width, height = img.size
    scale = (0.5, 1.0)
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    i,j,dh,dw = get_params(img, scale, ratio)
    resizer = transforms.Resize((height, width))
    # cropper = transforms.Compose([transforms.RandomCrop(size=(h, w)),
    #                                 transforms.Resize((height, width))])
    # cropper = transforms.Compose([transforms.CenterCrop(size=(h, w)),
    #                                 transforms.Resize((height, width))])

    # Crop single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        img = sample[key]
        crop = F.crop(img,i,j,dh,dw)
        sample[key] = resizer(crop)
        # sample[key] = transforms.RandomCrop(size=(h, w))(sample[key])
        # img = sample[key]
        # img.save('rgb_test.jpg')
        # sample[key] = transforms.Resize((height, width))(sample[key])
    # Crop lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        crops = [F.crop(k,i,j,dh,dw) for k in sample[key]]
        sample[key] = [resizer(crop) for crop in crops]

        # add intrinsic transforms
        l = j
        t = i
        r = j + dw
        b = i + dh
        sample['intrinsic_transform'] = intrinsic_transform(mode='cropping',w=width,h=height,l=l,t=t,r=r,b=b)

    # Return cropped sample
    return sample


def intrinsic_transform(camera='UCM', mode='identity', **kwargs):
    if not camera == 'UCM':
        raise ValueError('currently instrinsic changes are only implemented foorr UCM')
    else:
        if mode == 'identity':
            return torch.cat((torch.eye(5), torch.zeros(5,1)), 1)

        elif mode == 'cropping':
            w = kwargs['w']
            h = kwargs['h']
            l = kwargs['l']
            t = kwargs['t']
            r = kwargs['r']
            b = kwargs['b']

            # print('w')
            # print(w)
            # print('h')
            # print(h)
            # print('l')
            # print(l)
            # print('t')
            # print(t)
            # print('r')
            # print(r)
            # print('b')
            # print(b)

            transform = torch.zeros(5,6)
            transform[0,0] = w / (r - l)
            transform[1,1] = h / (b - t)
            transform[2,2] = w / (r - l)
            transform[2,5] = -l * w / (r - l)
            transform[3,3] = h / (b - t)
            transform[3,5] = -t * h / (b - t)
            transform[4,4] = 1

            # print(transform)

            return transform
            
        elif mode == 'horizontal_flipping':
            w = kwargs['w']
            h = kwargs['h']
            
            transform = torch.zeros(5,6)
            transform[0,0] = -1
            transform[1,1] = 1
            transform[2,2] = -1
            transform[2,5] = w
            transform[3,3] = 1
            transform[4,4] = 1

            return transform

        elif mode == 'vertical_flipping':
            w = kwargs['w']
            h = kwargs['h']

            transform = torch.zeros(5,6)
            transform[0,0] = 1
            transform[1,1] = -1
            transform[2,2] = 1
            transform[3,3] = -1
            transform[3,5] = h
            transform[4,4] = 1

            return transform

        else:
            raise ValueError('Currently only mode cropping, horizontal_flipping, vertical_flipping are implemented')
