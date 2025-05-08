"""
Class for augmenting tiles and recalculating GLCM features
outside (before) the training loop

"""

import os
import sys
import copy
import numpy as np
import importlib

import cupy

PATH_BASE_TEX = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../texture_calc/'))

# ===== import specific utils =====
sys.path.insert(0, PATH_BASE_TEX)
import texture_calc.FUNC_calc_texture_from_img as FUNC_calc_texture_from_img
import utils.geo_utils as geo_utils
import cnn_workflow.cnn_workflow.custom_augmentation as custom_augmentation


class AugmentDataset():
    '''
    Class to create and save augmentation files outside the dataloader
    loop. This can be used for offline augmentation where augmentation and
    recalculaiton of GLCM features is done outside (before) the training
    loop
    '''
    def __init__(self, data_files, seg_files,
                 proj_param_file=None, texture_param_file=None,
                 x_bands=None, y_bands=None, pytorch=True,
                 norm=False,
                 augmentations_geom=None, augmentations_col=None,
                 band_lst_col_aug=0, augmentations_val=None,
                 n_augment=1, augmentations_range=None,
                 proc_step=4, aug_vers=1,
                 tex_file_area_prefix_add='',
                 PATH_BASE=None, SCALE_TYPE=None):
        '''
        Dimension required for py torch:
            [n_channel, n_y, n_x]

        x_bands list of band names to input as features
        y_bands list of band name for segmentation (usually only one item)

        '''
        self.data_files = data_files
        self.seg_files = seg_files

        self.x_bands = x_bands
        self.y_bands = y_bands
        if self.y_bands is None:
            self.y_bands = ['class']
        self.n_y_bands = len(self.y_bands)
        self.norm = norm

        self.pytorch = pytorch

        self.transform = False

        if texture_param_file is not None:
            proj_param_module = importlib.import_module(
                proj_param_file)
            texture_param_module = importlib.import_module(
                f'.{texture_param_file}', 'param_settings')
            self.PARAMtex = proj_param_module.get_proj_param(
                proc_step, PATH_BASE, SCALE_TYPE)
            self.PARAMtex.update(
                {'tiling_area': tex_file_area_prefix_add})
            texture_param_module.add_proc_param(self.PARAMtex)

        self.augmentations_range = augmentations_range

        # geometric augmentations (applied to all bands)
        # e.g. rotate etc
        self.augmentations_geom = augmentations_geom

        # color augmentations (applied to self.band_lst_col_aug)
        self.augmentations_col = augmentations_col

        self.augmentations_val = augmentations_val

        # how many times to run augmentation pipeline (to create different
        # augmentations to be included in training)
        self.n_augment = n_augment
        self.aug_vers = aug_vers

        # which bands to use for color augmentation
        # (all bands are used if None)
        if band_lst_col_aug is None:
            self.band_lst_col_aug = slice(None)
        else:
            self.band_lst_col_aug = band_lst_col_aug

    def open_data_img(self, idx):
        '''
        open image and select required bands
        '''
        img = geo_utils.read_rename_according_long_name(
            self.data_files[idx], mask_nan=False)

        return img.sel(band=self.x_bands)


    def open_seg_img(self, idx):
        '''
        open raster with labels (segmentation)
        '''
        img = geo_utils.read_rename_according_long_name(
            self.seg_files[idx], mask_nan=False)

        return img


    def augment_img(self, idx):
        '''
        '''
        # get data
        x_img = self.open_data_img(idx)
        x = x_img.values
        # !!! for albumentations the channels must be at last dimension
        # thus need to use moveaxis
        x = np.moveaxis(x, 0, -1)  # .astype('float32')
        # Note converting to float should not be done as with new albumentaitons
        # changing range to float would not work otherwise

        # get classes
        y_img = self.open_seg_img(idx)
        y = y_img.values.squeeze()
        if self.n_y_bands > 1:
            y = np.moveaxis(y, 0, -1)

        # remove nan (but there shouldn't be any nans)
        x = np.where(np.isnan(x), 0, x)

        path_img = os.path.dirname(self.data_files[idx])
        file_name = os.path.basename(self.data_files[idx]).split('.')[0]
        file_name_seg = os.path.basename(self.seg_files[idx]).split('.')[0]

        x_orig = x.copy()
        y_orig = y.copy()
        # repeat augmentation by n_augment-times
        for i in range(self.n_augment):
            # run augmentation
            x, y = custom_augmentation.augmentation_chain(
                x_orig, y_orig, self.augmentations_geom,
                self.augmentations_col, self.augmentations_range,
                self.band_lst_col_aug)

            # move axis back --> is not needed here. axis is moved when
            # creating image

            prefix = f'_aug{str(self.aug_vers)}-{"{0:02d}".format(i + 1)}'
            # ---- save augmented files
            # - save data
            attrs = {
                'AREA_OR_POINT': 'Area', 'orig_type': str(x.dtype),
                'scale_factor': 1.0, 'add_offset': 0.0, '_FillValue': 0,
                'long_name': x_img.attrs['long_name']}
            x_img_aug = geo_utils.create_multiband_xarray_from_np_array(
                x_img, x, x_img.band.values.tolist(), attrs=attrs)
            out_path = os.path.join(
                path_img, file_name + prefix + '.tif')
            x_img_aug.rio.to_raster(out_path, driver='GTiff')

            # - save classes
            attrs = {
                'AREA_OR_POINT': 'Area', 'orig_type': str(y.dtype),
                'scale_factor': 1.0, 'add_offset': 0.0, '_FillValue': 0,
                'long_name': y_img.attrs['long_name']}
            if self.n_y_bands > 1:
                y_img_aug = geo_utils.create_multiband_xarray_from_np_array(
                    y_img, y, self.y_bands, attrs=attrs)
            else:
                y_img_aug = geo_utils.create_multiband_xarray_from_np_array(
                    y_img, y[:, :, np.newaxis], self.y_bands,
                    attrs=attrs)
            out_path = os.path.join(
                path_img, file_name_seg + prefix + '.tif')
            y_img_aug.rio.to_raster(out_path, driver='GTiff')

        return

    def recalc_texture(self, idx):
        '''
        !!! input must be int uint8 unit16 etc...!!!
        '''
        # get data
        x_img = self.open_data_img(idx)
        x = x_img.values
        x = np.moveaxis(x, 0, -1)

        # filename and path for output
        file_name = os.path.basename(self.data_files[idx]).split('.')[0]
        path_img = os.path.dirname(self.data_files[idx])

        # calcuate GLCM features
        FUNC_calc_texture_from_img.calc_texture(
            x_img, path_img, file_name, copy.deepcopy(self.PARAMtex),
            img_np=x)

        # calcuate texture statistics across GLCM features that were
        # calculated in different directions
        FUNC_calc_texture_from_img.texture_cross_calc(
            x_img, path_img, file_name, copy.deepcopy(self.PARAMtex),
            img_np=x)

        return

    def run_recalc_texture_gpu(self, idx, gpu_num):

        with cupy.cuda.Device(gpu_num):
            self.recalc_texture(idx)
