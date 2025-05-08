'''
Module for setting up the augmentation chain

In general augmentation was implemented according to:
https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
'''
import sys
import numpy as np
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = '1'
import albumentations as A


def get_train_augment_geom01(p1=1.0, p2=0.5):
    '''
    p: is the probability that this transformation is applied
    A.RandomRotate90: Randomly rotate the input by 90 degrees zero or more times.
    A.Flip: Flips image
    '''
    try:
        # for older albumentations version
        train_transform = A.Compose([
            A.RandomRotate90(p=p1),
            A.Flip(p=p2),]
            )
    except:
        # for newer albumentations version
        train_transform = A.Compose([
            A.RandomRotate90(p=p1),
            A.HorizontalFlip(p=p2),]
            )

    return train_transform


def get_train_augment_col_grey(
        p1=0.5, p2=0.5, b_limit=0.2, c_limit=0.2,
        gamma_min=80, gamma_max=120):
    '''
    p: is the probability that this transformation is applied

    A.RandomBrightnessContrast: Randomly change brightness and
    contrast of the input image.
    A.RandomGamma: randomly applies gamma stretch
    '''
    train_transform = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=b_limit,
            contrast_limit=c_limit, p=p1),
        A.RandomGamma(gamma_limit=(gamma_min, gamma_max), p=p2),
        ])

    return train_transform


def get_train_augment_col_grey_advanced(
        p1=0.5, p2=0.2, p3=0.2, b_limit=0.2, c_limit=0.2,
        gamma_min=80, gamma_max=120,
        j_bright=None, j_contrast=None, j_sat=None, j_hue=None):
    '''
    p: is the probability that this transformation is applied

    A.RandomBrightnessContrast: Randomly change brightness and
    contrast of the input image.
    A.RandomGamma: Applies random gamma correction to the input image.
    A.ColorJitter: Randomly changes the brightness, contrast, saturation, and hue of an image.
    A.Sharpen: Sharpen image using Gaussian blur.
    A.Blur: Apply uniform box blur to the input image using a randomly sized square kernel.
    '''
    if j_bright is None:
        j_bright = (0.8, 1)
    if j_contrast is None:
        j_contrast = (0.8, 1)
    if j_sat is None:
        j_sat = (0.8, 1)
    if j_hue is None:
        j_hue = (-0.5, 0.5)

    train_transform = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=b_limit, contrast_limit=c_limit, p=p1),
        A.RandomGamma(gamma_limit=(gamma_min, gamma_max), p=p2),
        A.ColorJitter(brightness=j_bright, contrast=j_contrast,
                      saturation=j_sat, hue=j_hue, p=p2),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=p3),
        A.Blur(blur_limit=3, p=p3)
        ])

    return train_transform


def convert_to_float(bit_depth):
    '''
    to create float albumentations will divide by bit depth
    '''
    transform = A.Compose([A.ToFloat(max_value=bit_depth)])
    return transform


def revert_to_int(bit_depth):
    '''
    to create float albumentations will divide by bit depth
    '''
    transform = A.Compose([A.FromFloat(max_value=bit_depth)])
    return transform


def augmentation_chain(x_inp, y_inp, augmentations_geom, augmentations_col,
                       augmentations_range, band_lst_col_aug,
                       augmentations_fad=None,
                       sensor_id=None):
    '''
    Set up chain for augmentstion

    Parameters
    ------
    x_inp: numpy array !!! type must be integer
        selected input features to augment e.g. selected with
        x[:, :, band_lst_col_aug]
    augmentations_geom: pre set geometric augmentation functions from
        albumentation (setup in param_uils.get_aug_param())
    augmentations_col: pre set color augmentation functions from
        albumentation (setup in param_uils.get_aug_param())
    augmentations_range: pre set cugmentation to convert from int to float
        since some albumentation augmentations require float inptu
    band_lst_col_aug: list
        List of bands for which to apply augmentation
    augmentations_fad: dict
        pre set augmentation for fourier domain adaptation (per sensor type)
        (not used in this script version)
    sensor_id: int
        is required to select correct FAD augmentation from
        augmentations_fad dict

    # Note: here it is not possible to mask all with y mask.
    #   This is because x window was choosen larger and extends padded
    #   y at edges, in order to avoid edge effects in GLCM calculation.
    #   Therefore, create mask arrays in this function.
    '''
    x = x_inp.copy()
    y = y_inp.copy()

    if augmentations_geom is not None:
        augmented = augmentations_geom(
            image=x, mask=y)
        x, y = augmented['image'], augmented['mask']
        # get nan mask
        maskx = (x == 0).copy()
    else:
        # get nan mask
        maskx = (x == 0).copy()

    if augmentations_col is not None or augmentations_fad is not None:
        # augmentation range should only be used if applied to
        # greyscale imagery with integers !!!
        if augmentations_range is not None:
            # fits all values from 0 - (2**bitrange - 1) to 0 - 1 range

            # !!! NOTE with new albumentations library int are required
            # as inputs here. Otherwise values are just converted to
            # floats but not notmalized to be within 0 - 1
            if x.dtype.name.find('int') == -1:
                # ensure compatibilty with albumentations library
                sys.exit('!!! there is an error with dtypes as input for albumentations')
            x[~maskx] -= 1  # subtract 1 but only for values which are
            # not 0 as Uint does not allow negative values (would wrap it around)
            augmented = augmentations_range[0](image=x)
            x = augmented['image']

            # again ensure correct range
            if x.max() > 1:
                # ensure compatibilty with albumentations library
                sys.exit('!!! there is an error converting ints to floats with albumentations')

        if augmentations_fad is not None:
            # this is for fourier domain adaptation augmentations
            # reference image must have been defined before in the main
            # training file (not implemented in MAIN_train_incl_aug)
            augmented = augmentations_fad[sensor_id](
                image=x[:, :, band_lst_col_aug])
            x[:, :, band_lst_col_aug] = augmented['image']

        if augmentations_col is not None:
            # color augmentation
            augmented = augmentations_col(
                image=x[:, :, band_lst_col_aug])
            x[:, :, band_lst_col_aug] = augmented['image']

        if (np.min(x[:, :, band_lst_col_aug]) < 0
            or np.max(x[:, :, band_lst_col_aug]) > 1):
            # make sure that values are between -0 an 1
            x[:, :, band_lst_col_aug] = normalize_range_bands(
                x[:, :, band_lst_col_aug])

        # move back to integers --> range [0, 2**PARAM['BIT_DEPTH'] - 2]
        if augmentations_range is not None:
            augmented = augmentations_range[1](image=x)
            x = augmented['image']
            x += 1  # shift integers back to orig range

        x[maskx] = 0

    return x, y


def normalize_range_bands(array_inp):
    '''
    Normalize to 0 - 1 range

    Parameters:
        array_inp: numpy array or rio data array
        (e.g. single band with rio_array.sel(band='band_name').values)
        but can also use full array rio_array.data

    Returns:
        numpy array
    '''
    axis = (0, 1)  #(1, 2)

    array_min = np.nanmin(array_inp,axis=axis)
    array_max = np.nanmax(array_inp,axis=axis)
    array_range = array_max - array_min
    out = (array_inp - array_min[np.newaxis, np.newaxis, :]) / array_range[np.newaxis, np.newaxis, :]

    return out

