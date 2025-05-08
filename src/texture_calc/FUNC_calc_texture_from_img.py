'''
Functions to calculate texture for small raster inputs
The input raster is not split-up into smaller subtiles.
In case of memory overflow issues would need to use MAIN_calc_texture.py

This functions are used in:
- cnn_workflow/MAIN_augment_calc_texture.py: to prepare tiles for
    offline augmentation, where training tiles are augmented and GLCM
    are recalculated outside training loop). The functions are then called within
    AugmentDataset() (self.augment_texture())
- cnn_workflow/MAIN_calc_texture_tiles.py: to prepare tiles for
    prediction, where GLCM are directly calculated on non-augmented tiles
    AugmentDataset() (self.augment_texture())
- texture_calc/MAIN_calc_texture_single_file.py: to calculate the texture
    for single tiles this is called from within the dataloader
    (CustomDataset_augment_ScaleMerge) for online augmentation
    (i.e. augmenting and recalculating the GLCMs at every epoch)

Thexture is calculates using using glcm-cupy
(https://github.com/Eve-ning/glcm-cupy)

'''

import os
import numpy as np
import copy

# ===== import specific utils =====
import texture_calc.texture_utils as texture_utils
import utils.geo_utils as geo_utils


def calc_texture(img, PATH_PROC, FILE_PREFIX, PARAM,
                 img_np=None, add_orig_img=False):
    '''

    Notes:
        - img input can either be path to input image file or it can be
            xarray of raster image
    '''
    PARAM['PATH_EXPORT'] = PATH_PROC
    PARAM['FILE_PREFIX'] = FILE_PREFIX
    PARAM['FILE_INP'] = img

    # initialize texture class
    texture = texture_utils.TextureCalc(
        PARAM['EPSG_TARGET'], PARAM['PATH_EXPORT'], PARAM['FILE_PREFIX'],
        padding_const=PARAM['PADDING_CONST'],
        bin_from=PARAM['BIN_FROM'], bin_to=PARAM['BIN_TO'],
        tex_suffix=PARAM['TEX_SUFFIX'],
        img=img)  # image is directly added here !!!!

    if isinstance(PARAM['FILE_INP'], str):
        # read file if FILE_INP is path
        # here use mask nan because is better for cupy-glcm
        # however this isla so checked later
        texture.preproc_img(
            PARAM['PATH_EXPORT'], PARAM['FILE_INP'],
            AOI_coords=None, RESOLUTION_target=0, mask_nan=True)
    else:
        texture.image[texture.img_inp_key] = img

    if img_np is None:
        texture.img_to_nparray(band=None, proc_type='img')
    else:
        texture.img_np[texture.img_inp_key] = img_np

    for i in PARAM['TEX_PARAM']:
        texture.setup_window_param(*i)
        texture.derive_texture_np_GPU()

        if add_orig_img:
            texture.texture_np_add_inp_img()

        texture.texture_to_img_save(AOI_poly=None)

        # delete texture_np, and image image
        texture.del_texture_and_img()

    del texture
    return


def texture_cross_calc(
        img, PATH_PROC, FILE_PREFIX, PARAM, img_np=None):
    """
    Notes:
        - img input can either be path to input image file or it can be
            xarray of raster image
    """

    PARAM['PATH_EXPORT'] = PATH_PROC
    PARAM['FILE_PREFIX'] = FILE_PREFIX
    PARAM['FILE_INP'] = img

    texture = texture_utils.TextureCalc(
        PARAM['EPSG_TARGET'], PARAM['PATH_EXPORT'], PARAM['FILE_PREFIX'],
        padding_const=PARAM['PADDING_CONST'],
        bin_from=PARAM['BIN_FROM'], bin_to=PARAM['BIN_TO'],
        tex_suffix=PARAM['TEX_SUFFIX'],
        img=img)  # image is directly added here !!!!

    if isinstance(PARAM['FILE_INP'], str):
        # read file if FILE_INP is path
        # here use mask nan because is better for cupy-glcm
        # howevert this is also checked later
        texture.preproc_img(
            PARAM['PATH_EXPORT'], PARAM['FILE_INP'],
            AOI_coords=None, RESOLUTION_target=0, mask_nan=True)
    else:
        texture.image[texture.img_inp_key] = img

    if img_np is None:
        texture.img_to_nparray(band=None, proc_type='img')
    else:
        texture.img_np[texture.img_inp_key] = img_np

    for i_key, i_set in PARAM['img_cross_calc'].items():
        names = []
        for i in i_set:
            texture.setup_window_param(*i)
            texture.derive_texture_np_GPU()

            # do not save separate direction texture files here
            texture.texture_to_img_save(AOI_poly=None, save=False)
            # !!! here self.texture_np needs to be kept for cross
            # calculation and is deleted afterwards

        names = texture.texture_np.keys()
        calc_out, bands_calc = geo_utils.concat_arrays_and_calc(
            {x: texture.image[x] for x in names},
            calc=PARAM['cross_calc'], dict_prefix=i_key + '_calc')

        # remove not required images
        texture.del_texture_and_img()

        # save stats
        for i_key, i_val in calc_out.items():
            filename = (PARAM['FILE_PREFIX'] + '_' + i_key)
            geo_utils.save_to_geotiff_options(
                i_val, PARAM['PATH_EXPORT'], filename,
                single=False)

    del texture
    return


def calc_texture_complete(img_file, x_bands, PARAM):
    '''
    img_file: file_names.data_files[idx]

    !!! input must be int uint8 unit16 etc...!!!
    '''
    # get data
    x_img = geo_utils.read_rename_according_long_name(
        img_file, mask_nan=False).sel(band=x_bands)

    x = x_img.values
    x = np.moveaxis(x, 0, -1)

    # filename and path for output
    file_name = os.path.basename(img_file).split('.')[0]
    path_img = os.path.dirname(img_file)

    # calcuate GLCM features
    calc_texture(
        x_img, path_img, file_name, copy.deepcopy(PARAM),
        img_np=x)

    # calcuate texture statistics across GLCM features that were
    # calculated in different directions
    texture_cross_calc(
        x_img, path_img, file_name, copy.deepcopy(PARAM),
        img_np=x)

    return