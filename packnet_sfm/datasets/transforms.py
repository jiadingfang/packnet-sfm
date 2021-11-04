# Copyright 2020 Toyota Research Institute.  All rights reserved.

from functools import partial
from packnet_sfm.datasets.augmentations import resize_image, resize_sample, \
    duplicate_sample, colorjitter_sample, to_tensor_sample, random_hflip_sample, \
    random_vflip_sample, center_crop_sample, random_crop_sample, init_intrinsic_transform_sample

########################################################################################################################

def train_transforms(sample, image_shape, jittering, random_hflip, random_vflip, center_crop, random_crop):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    random_hflip : Boolean
        If using random hfilp or not
    random_vflip : Boolean
        If using random vfilp or not
    center_crop : Boolean
        If using cneter crop or not

    Returns
    -------
    sample : dict
        Augmented sample
    """

    # if center_crop:
    #     crop_shape = (448, 704)
    #     sample = center_crop_sample(sample, crop_shape) # center crop for euroc dataset
    sample = init_intrinsic_transform_sample(sample)
    if len(image_shape) > 0:
        sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if len(jittering) > 0:
        sample = colorjitter_sample(sample, jittering)

    if random_hflip + random_vflip + random_crop > 1:
        raise ValueError('only one data augmentation method allowed currently')
    if random_hflip:
        sample = random_hflip_sample(sample) # add random hflipper as data augmentation
    if random_vflip:
        sample = random_vflip_sample(sample) # add random vflipper as data augmentation
    if random_crop:
        sample = random_crop_sample(sample) # use random crop as data augmentation

    sample = to_tensor_sample(sample)
    return sample

def validation_transforms(sample, image_shape):
    """
    Validation data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    sample = init_intrinsic_transform_sample(sample)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)

    # image_shape = (192, 320)
    # sample = resize_sample(sample, image_shape)

    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape):
    """
    Test data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """
    sample = init_intrinsic_transform_sample(sample)
    if len(image_shape) > 0:
        sample['rgb'] = resize_image(sample['rgb'], image_shape)

    # image_shape = (192, 320)
    # sample = resize_sample(sample, image_shape)

    sample = to_tensor_sample(sample)
    return sample

def get_transforms(mode, image_shape, jittering, random_hflip, random_vflip, center_crop, random_crop,  **kwargs):
    """
    Get data augmentation transformations for each split

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the data augmentation transformations
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    random_hflip : Boolean
        If using random hfilp or not
    random_vflip : Boolean
        If using random vfilp or not
    center_crop : Boolean
        If using cneter crop or not
    random_crop: Boolean
        If using random crop or not

    Returns
    -------
        XXX_transform: Partial function
            Data augmentation transformation for that mode
    """
    if mode == 'train':
        return partial(train_transforms,
                       image_shape=image_shape,
                       jittering=jittering,
                       random_hflip=random_hflip,
                       random_vflip=random_vflip,
                       center_crop=center_crop,
                       random_crop=random_crop)
    elif mode == 'validation':
        return partial(validation_transforms,
                       image_shape=image_shape)
    elif mode == 'test':
        return partial(test_transforms,
                       image_shape=image_shape)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

########################################################################################################################

