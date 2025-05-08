"""
Data loading functionalities for PyTorch

Dimension required for PyTorch:
            [n_channels, n_y, n_x]

Possible additional TODOs:
- Define self.take_log and usage of get_img_log() and allow band
  specific selection if log should be taken (same also for exp)

"""

import os
import sys

import tempfile
import subprocess
from glob import glob
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import xarray

import torch

# check if GPU available
if torch.cuda.is_available():
    import cupy

import utils.geo_utils as geo_utils
import utils.xarray_utils as xarray_utils
import utils.plotting_utils as plotting_utils
import utils.stats_utils as stats_utils
import utils.conversion_utils as conversion_utils

class GetFilenames():
    """
    Class to handle filenames for CNN input
    """
    def __init__(self, path_inp, file_prefix,
                 x_bands=None, y_bands=None):
        '''
        x_bands: list of band names to input as features
        y_bands: list of band name for segmentation (usually only one item)

        file_prefix: where want to read in train and validate etc separate
        need to set prefix with phase e.g. BLyakhE_SPOT2018_B01_4bit_validate
        '''
        self.path_inp = path_inp
        self.file_prefix = file_prefix

        self.x_bands = x_bands
        self.y_bands = y_bands

        self.data_files = {}  # file path to data files (e.g. greyscale imagery)
        self.seg_files = {}  # file path to training label files
        self.file_id = {}  # specifies file origin
        # e.g. BLyaE_HEX1979_A01_perc0-2_g0-3_8bit is use to later relate
        # file type specific paraneters

        self.meta = {}  # metadata dictionary

    def get_metadata(self, phase_inp, phase_out='all',
                     meta_suffix='meta_data', file_id=None,
                     aug_query=None):
        '''
        read metadata file if available

        if want to add different regions then need to run this function
        several times with different input folders (change self.path_inp)

        phase_inp: specifies which metadata file to read. This can for
            example correspond to a specific training sub-area
            (e.g. for [train-01, train-02])

        phase_out: is the phase to which the metadata from te sub-area
            belongs (e.g. 'train')

        file_id: is key that specifies input data type
            (e.g. BLyaE_HEX1979_A01_perc0-2_g0-3_8bit)
            It is later to select corresponding file type specific parameters
            (e.g. mean and std for standardisation)

        '''
        if not isinstance(phase_inp, list):
            phase_inp_lst = [phase_inp]
        else:
            phase_inp_lst = phase_inp

        # collect metadata for different phases sub-areas
        # (e.g.  ['train-01', 'train-02'])
        meta_lst = []
        for i_phase in phase_inp_lst:
            # get metadata file path
            file_name = [self.file_prefix + i_phase, meta_suffix]
            file_path = os.path.join(
                self.path_inp, '_'.join(file_name) + '.txt')
            # read metadata
            meta = pd.read_csv(
                file_path, sep='\t', header=0, index_col=0)
            if aug_query is not None:
                meta = meta.query(aug_query)

            # exchange path with actual input path
            meta['file_data_path'] = meta['file_data'].apply(
                lambda x: os.path.join(self.path_inp, os.path.basename(x)))
            if phase_out != 'predict':
                # for prediction no labels are available (file_class_path)
                meta['file_class_path'] = meta['file_class'].apply(
                    lambda x: os.path.join(self.path_inp, os.path.basename(x)))

            # extract prefix from file name (will be used as base to
            # search e.g. for GLCM feature files with ending ..._a0-1-2-3_r05_norm_C01.tif)
            meta['orig_prefix_data'] = meta['file_data'].apply(
                lambda x: os.path.basename(x).split('.')[0])

            # add file_id
            meta['file_id'] = file_id

            # rename to avoid issues with query due to minus sign
            if phase_out != 'predict':
                rename_dict = {'perc-class_' + str(x): 'perc_class_' + str(x)
                            for x in range(1, 8)}
                meta.rename(columns=rename_dict, inplace=True)
            meta_lst.append(meta)

        # to define x_band etc just use last meta item...
        if self.x_bands is None:
            try:
                self.x_bands = str(meta['x_bands'].tolist()[0]).split(':')
            except:
                self.x_bands = meta['bands'].tolist()[0].split(':')[:-1]
        if self.y_bands is None and phase_out != 'predict':
            try:
                self.y_bands = str(meta['y_bands'].tolist()[0]).split(':')
            except:
                self.y_bands = meta['bands'].tolist()[0].split(':')[-1:]

        # add metadata info to phase specific dictionaly item
        if phase_out in self.meta.keys():
            self.meta[phase_out] = pd.concat(
                [self.meta[phase_out], *meta_lst], axis=0)
            # make sure that there are no duplicates
            self.meta[phase_out].drop_duplicates(inplace=True)
        else:
            self.meta[phase_out] = pd.concat(meta_lst, axis=0)

        del meta, meta_lst
        return

    def filter_files(self, query_text, phase):
        '''
        Choose specific files with query
        query_text = 'perc-class_1 >= 10'

        '''
        if ('perc-class_0' not in self.meta[phase].columns
            or np.any(np.isnan(self.meta[phase]['perc-class_0']))) and phase != 'predict':
            count_col = [x for x in self.meta[phase].columns
                 if x.find('count-class') > -1]
            count_sum = self.meta[phase].loc[:, count_col].sum(axis=1)
            self.meta[phase]['perc-class_0'] = (self.meta[phase].loc[:, 'count-class_0']/count_sum)*100

        self.meta[phase] = self.meta[phase].query(query_text)
        return

    def get_file_names_from_meta(
            self, phase_inp, phase='all', meta_suffix='meta_data',
            file_id=None, query_text=None, aug_query=None):
        '''
        phase_inp can be list with sub-areas e.g. [train-01, train-02]
        phase is actual output phase name e.g. train
        and then later for validation it could be
        phase_inp train-03, phase=validate


        if only_augment is True then only augmented image will be used
        in training orignal image will be discarded
        '''

        self.get_metadata(phase_inp, phase, meta_suffix, file_id=file_id,
                          aug_query=aug_query)
        if query_text is not None:
            self.filter_files(query_text, phase)
        self.len_orig_files = self.meta[phase].shape[0]

        self.data_files[phase] = self.meta[phase]['file_data_path'].tolist()
        if phase != 'predict':
            self.seg_files[phase] = self.meta[phase]['file_class_path'].tolist()
        else:
            self.seg_files[phase] = None
        self.file_id[phase] = self.meta[phase]['file_id'].tolist()

        return

    def save_metadata(
            self, bands_out, path_out=None, phase='all',
            meta_suffix='meta_data'):
        '''
        save metadata file
        '''
        self.meta[phase].loc[:, 'bands'] = ':'.join(bands_out)
        file_name = [
            self.file_prefix + phase, meta_suffix + '.txt']
        # file_name.remove('')

        if path_out is None:
            path_out = self.path_inp

        file_path = os.path.join(
            path_out, '_'.join(file_name))

        self.meta[phase].to_csv(
            file_path, sep='\t', lineterminator='\n', header=True)
        return

    def split_train_validate(self, phase='all', split_ratio=0.6,
                             test_patch_num=10, n_augment=1):

        file_arr = np.stack(
            [self.data_files[phase], self.seg_files[phase], self.file_id[phase]]).T
        np.random.shuffle(file_arr)  # is applied in place to file_arr

        file_split = int(np.ceil(file_arr.shape[0]*split_ratio))
        self.data_files['train'], self.seg_files['train'], self.file_id['train'] = \
            file_arr[:file_split, :].T.tolist()
        self.data_files['validate'], self.seg_files['validate'], self.file_id['validate'] = \
            file_arr[file_split:-test_patch_num, :].T.tolist()
        self.data_files['test'], self.seg_files['test'], self.file_id['test'] = \
            file_arr[-test_patch_num:, :].T.tolist()

        if n_augment > 1:
            self.data_files['train'] *= n_augment
            self.seg_files['train'] *= n_augment
            self.file_id['train'] *= n_augment
        return

    def get_file_names_glob(
            self, file_id, phase='all', file_prefix='',
            data_search_suffix="*data.tif", seg_search_suffix="*seg.tif",
            append_to_lst=False):
        '''
        Files can be appended. This can be useful if want to merge files
        with different suffixes (e.g. if from different augmentations:
            *seg_aug1-0*.tif, where aug1 corresponds to augmentation type 1)
        '''
        if not append_to_lst or phase not in self.data_files.keys():
            self.data_files[phase] = []
            self.seg_files[phase] = []
            self.file_id[phase] = []

        self.data_files[phase] += sorted(
            glob(os.path.join(self.path_inp,
                              file_prefix + data_search_suffix)))
        self.seg_files[phase] += sorted(
            glob(os.path.join(self.path_inp,
                              file_prefix + seg_search_suffix)))
        self.file_id[phase] += [file_id]*len(self.data_files[phase])

        # get available bands
        if self.x_bands is None:
            self.x_bands = geo_utils.get_bands(self.data_files[phase][0])

        if self.y_bands is None and len(self.seg_files[phase]) > 0:
            self.y_bands = geo_utils.get_bands(self.seg_files[phase][0])

        return


class CustomDataset_ScaleMerge_base():
    """
    Base class with functions to be inherited by other
    CustomDataset classes
    """
    def __init__(self, data_files, file_suffix_lst, z_stats,
                 path_proc, seg_files=None, path_suffix=None,
                 x_bands=None, y_bands=None, add_bands=None,
                 width_out=None, height_out=None,
                 calc_pca=False, PCA_components=3,
                 standardize_add=True, norm_on_std=True,
                 norm_min_max_band1=None, take_log=False,
                 gpu_no=7, standardize_individual='No',
                 if_std_band1=False, norm_clip=True):
        """
        """
        self.data_files = data_files
        self.seg_files = seg_files  # this is None for prediction data
        # (full set)

        self.z_stats = z_stats  # is module to calculate PCA

        self.x_bands = x_bands
        self.y_bands = y_bands  # this is None for prediction data
        self.add_bands = add_bands  # list with bands to take from each
        # file defined in file_suffix_lst

        self.file_suffix_lst = file_suffix_lst  # file suffix of files
        # to be merge with grey scale imagery

        self.calc_pca = calc_pca  #  None is no calculation
        # 'separate' is PCA calculation for each GLCM file separately
        # (e.g. specific window size)
        # 'all' is over all merged additional files (but not including
        # grey scale imagery)
        self.PCA_components = PCA_components  # how many components

        # standardisation
        self.standardize_add = standardize_add  # if want to standardscale
        # channels from file_suffix_lst (True or False)
        self.standardize_individual = standardize_individual  # if standardize
        # according to current patch. No normalization (if == "add"
        # then only the additional bands are standardized with "all"
        # all bands are standardized)
        self.if_std_band1 = if_std_band1  # if standard scale band 1
        # with values form stats file

        self.norm_min_max_band1 = norm_min_max_band1  # if normalize grey
        # scale band (e.g. [0, 255]). If set to None then no normalzation
        # is done
        if norm_on_std == 999:
            self.norm_on_std = False
            self.norm_min_max = [0.5, 99.5]
            self.norm_clip = False
        else:
            self.norm_on_std = True  # # if normalize standardized
            self.norm_min_max = [norm_on_std, 100 - norm_on_std]
            self.norm_clip = norm_clip  # if clip to 0 and 1 after nomalize

            # data (True or False)

        self.take_log = take_log  # True or False

        # window sizes for tiles (used to crop edges effects if
        # original tile has a larger size)
        # will be none for prediction data
        self.width_out = width_out
        self.height_out = height_out

        self.gpu_no = gpu_no

        if path_suffix is not None:
            self.path_out = os.path.join(path_proc, path_suffix)
            if not os.path.isdir(self.path_out):
                os.mkdir(self.path_out)

        # plot counting for file plot file name
        self.count_plot = 0
        self.count_plot_hist = 0

    def open_data_img(self, idx, chunk='auto'):
        '''
        Open data file according to index in list and select required
        band (greyscale)
        Additional other bands are read in with open_additional_img() as
        called in scale_merge()
        '''
        # do not use mask nan due to augmentation
        img = geo_utils.read_rename_according_long_name(
            self.data_files[idx], mask_nan=False, chunk=chunk)

        return img.sel(band=self.x_bands)

    def open_seg_img(self, idx, chunk='auto'):
        '''
        Open segmentation file according to index in list
        '''
        img = geo_utils.read_rename_according_long_name(
            self.seg_files[idx], mask_nan=False, chunk=chunk)

        return img

    def open_additional_img(self, basename, suffix, band, chunk='auto'):
        '''
        Open and return xarray image with additional bands and select
        required bands
        '''
        file_path = basename + '_' + suffix + '.tif'
        img = geo_utils.read_rename_according_long_name(
            file_path, mask_nan=False, chunk=chunk)

        return img.sel(band=band)

    def extract_xarray_stats(self, file_suffix, band_lst, log_pref):
        '''
        Extracts the required statistical values from xarray dataset
        self.xarr_stats and calculates e.g. the mean across the AOIs
        Note: self.xarr_stats was extractd from
            self.xarr_stats_dict[file_id] and there fore contains the
            correct AOIs already

        Parameters
        ------
        - log_prefix: str
            The prefix used to select the relevant variables in the
            self.xarr_stats xarray dataset.
            Can e.g. be "log" if is using log data
        - file_suffix: str
            Defines DataArray "name" coordinate to be choosen.
            It corresponds to the data filename suffix
            (e.g. 'a0-1-2-3_r01_norm_C01').
        - band_list: list
            A list of band names to extract statistics for.
            (e.g. ['1'] or  ['1', 'ASM', ...])

        Returns
        ------
        - mean_inp: float
            The mean value of the selected bands.
        - std_inp: float
            The standard deviation of the selected bands.
        - std_sym_inp: np.ndarray
            The symmetric standard deviation, computed as the maximum
            absolute value of the minimum and maximum percentage standard
            deviations.
        '''
        key_perc_min = f'{log_pref}std_perc_{self.norm_min_max[0]}'
        key_perc_max = f'{log_pref}std_perc_{self.norm_min_max[1]}'
        mean_inp = self.xarr_stats[log_pref + 'mean'].sel(
            name=file_suffix, drop=True).sel(band=band_lst).mean(dim='AOI')
        std_inp = self.xarr_stats[log_pref + 'std'].sel(
            name=file_suffix, drop=True).sel(band=band_lst).mean(dim='AOI')
        std_perc_min = self.xarr_stats[key_perc_min].sel(
            name=file_suffix, drop=True).sel(band=band_lst).min(dim='AOI')
        std_perc_max = self.xarr_stats[key_perc_max].sel(
            name=file_suffix, drop=True).sel(band=band_lst).max(dim='AOI')
        std_sym_inp = np.max(
            np.abs([std_perc_min, std_perc_max]),
                axis=0)[:, np.newaxis, np.newaxis]

        return mean_inp, std_inp, std_sym_inp

    def get_stats_dict(
            self, path_prefix, stats_file_dict, stats_AOI_dict):
        '''
        Read the pre-calculated statistics from the meta stats file.
        The stats file is selected according to the training data areas
        used.

        Parameters
        ----------
        path_prefix : str
            The prefix of the file paths.
        stats_file_dict : dict
            A dictionary where keys are file IDs and values are dictionaries
            with AOI keys and file paths.
            Example:
            {
                'BLyakhE_HEX1979_A01_perc0-2_g0-3_8bit': {
                    'A01': 'path_to_file',
                    'A02': 'path_to_file'
                }
            }
        stats_AOI_dict : dict
            A dictionary where keys are file IDs and values are dictionaries
            with AOI keys and the list of training splits, which are used to
            select the correct file (replace "REPLACE" in file name e.g.
            with 'train-01').
            Example:
            {
                'BLyakhE_HEX1979_A01_perc0-2_g0-3_8bit': {
                    'A01': ['train-01', 'train-02'],
                    'A02': ['train-01', 'train-02']
                }
            }

        Returns
        -------
        None

        Note:
        - input for stats_AOI_dict is
            PARAM['PHASE_STATS_FILE'][CV_NUM] thus the correct train-XX
            subset was taken accordin to cross-validation number CV_NUM
        - later stats can be choosen with:
            xarr_stats[file_id]['perc_99-5'].sel(name=img_key, drop=True)
        '''
        self.xarr_stats_dict = {}
        for file_id, i_path_dict in stats_file_dict.items():
            # file_id is e.g. 'BLyakhE_HEX1979_A01_perc0-2_g0-3_8bit'
            cv_stats_dict = stats_AOI_dict[file_id]

            stats_xr = {}
            # i_path_dict is e.g. {'A01': 'path_to_file', 'A02': 'path_to file'}
            for i_key, i_path in i_path_dict.items():
                cv_train_lst = cv_stats_dict[i_key]
                for i_cv in cv_train_lst:
                    i_path_cv = i_path.replace('REPLACE', i_cv)
                    stats_xr[f'{i_key}:{i_cv}'] = read_stats_to_xarray(
                        path_prefix, i_path_cv, bands_ordered_lst=None)

            self.xarr_stats_dict[file_id] = xarray_utils.merge_xarray_dict(
                stats_xr, name_new_dim='AOI')
        return

    def scale_merge(self, x_img, basepath, chunk='auto', y_img=None):
        """
        Scale input features according to selected options from
        PARAM['file_merge_param']
        and clear edge effects

        main steps:
            1) From base file name: get all required files
                    e.g. for
                    BLyakhE_HEX1979_B01_train_00_00-00_data.tif
                    BLyakhE_HEX1979_B01_train_00_00-00_seg.tif
                    BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r05_norm_C01.tif
                    BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r10_norm_C01.tif
                    BLyakhE_HEX1979_B01_train_00_00-00_data_r10_calc_std.tif
                    file_suffix_lst = ['a0-1-2-3_r05_norm_C01',
                        'a0-1-2-3_r10_norm_C01', 'r10_calc_std']
            2) scale img
            3) merge img

        """
        # -- 1) clear area-AOI edge effects
        # This is done by selecting merged data according to y
        if self.seg_files is not None:
            x_img = x_img.where(
                ~(y_img.sel(band=self.y_bands[0]) == 0).data.squeeze())

        # -- 2) remove tile-edge effects of GLCM calculation
        if self.width_out is not None or self.height_out is not None:
            # center crop to requested inage size (tiles were creeated
            # with an additional padding)
            x_img_shape = x_img.shape
            x_img, x_crop_id, y_crop_id = geo_utils.center_crop_img(
                x_img, x_img_shape[1], x_img_shape[2],
                self.width_out, self.height_out)

        # ------ 3) Scale greyscale band
        if self.norm_min_max_band1 is not None:
            # option: normalize greyscale with specified min max to [0, 1]
            x_img = stats_utils.normalize_xarray(
                x_img, min_norm=self.norm_min_max_band1[0],
                max_norm=self.norm_min_max_band1[1])
        elif self.if_std_band1:
            # Option: standardise greyscale with stat values from
            # trainng area
            # get training area statistics
            mean_inp_g, std_inp_g, std_sym_inp_g = self.extract_xarray_stats(
                    'raw', ['1'], '')
            # requires background in greyscale image to ne nans not zeros!
            x_img = stats_utils.standardize_norm(
                x_img, mean_inp=mean_inp_g, std_inp=std_inp_g,
                sym_min_max=std_sym_inp_g, if_norm=self.norm_on_std,
                norm_clip=self.norm_clip)

        # ------4) Optional: Read additional input features
        # Additional input features (e.g. GLCM features) are included
        # accordinag to self.file_suffix_lst (defined as PARAM['file_suffix_lst']),
        # containing e.g. ['a0-1-2-3_r05_norm_C01'] (radius 5px moving
        # window in all direction). self.add_bands (prvided by
        # PARAM['merge_bands']) define which bands to take per file_suffix
        # e.g. [['HOMOGENEITY', 'CONTRAST', 'ASM', 'VAR]] ()
        xx_lst = []
        xx_img = None
        for i, i_band in zip(self.file_suffix_lst, self.add_bands):
            # read .tif as defined by the file suffix and extract the
            # required bands
            xx_img = self.open_additional_img(
                basepath, i, i_band, chunk=chunk)
            if self.seg_files is not None:
                # for training also open the corresponding .tif file with
                # the label data
                xx_img = xx_img.where(
                    ~(y_img.sel(band=self.y_bands[0]) == 0).data.squeeze())

            # center crop to not include edge effects when calculating
            # mean and std
            if self.width_out is not None or self.height_out is not None:
                xx_img_shape = xx_img.shape
                xx_img, x_crop_id, y_crop_id = geo_utils.center_crop_img(
                    xx_img, xx_img_shape[1], xx_img_shape[2],
                    self.width_out, self.height_out)

            if self.take_log:
                # take the log of the intput features
                # here use a small shihist of 0.01 to ti shift all values
                # this makes sure that no negative vlaues are present

                # negtive values is nan
                #img_min = self.xarr_stats['min'].sel(
                #    name=i, drop=True).sel(band=i_band).min(dim='AOI')
                # here use fixed shift of 0.01 as GLCMs should not be negative 0
                xx_img = geo_utils.get_img_log(xx_img, img_shift=0.01)
                log_pref = 'log_'
            else:
                log_pref = ''

            # ---- standardize additional GLCM channels
            if self.standardize_add:
                # --- standardize GLCMs according ot stats file ---
                # extract required stats
                mean_inp, std_inp, std_sym_inp = self.extract_xarray_stats(
                    i, i_band, log_pref)

                # standardise and if required also take norm
                xx_img = stats_utils.standardize_norm(
                    xx_img, mean_inp=mean_inp, std_inp=std_inp,
                    sym_min_max=std_sym_inp, if_norm=self.norm_on_std,
                    norm_clip=self.norm_clip)
            elif (self.standardize_individual == 'add'
                  or (self.calc_pca == 'all' or self.calc_pca == 'separate')):
                # --- standardize according to individual values
                # if use PCA then always use individual tile scaling here
                xx_img = stats_utils.standardize_norm(
                    xx_img, if_norm=self.norm_on_std,
                    get_perc_qmin_qmax=self.norm_min_max,
                    norm_clip=self.norm_clip)

            if self.calc_pca == 'separate':
                # -- calculate PCA for separately per GLCM-window/direction
                # combination
                if torch.cuda.is_available() and self.gpu_no != 99:
                    with cupy.cuda.Device(self.gpu_no):
                        xx_img = self.z_stats.get_PCA(
                            xx_img, col_use=None,
                            n_components=self.PCA_components,
                            whiten=False, bands=None)
                else:
                    xx_img = self.z_stats.get_PCA(
                        xx_img, col_use=None,
                        n_components=self.PCA_components,
                        whiten=False, bands=None)

                # if calculated pca and want to stadardise the calculated PCA,
                # hten always tile based standardisation is used.
                # This is because to use the pre-comuted stats from
                # the trainng areas would require the stats to have been
                # calculated for the PCA, of the specific band combination
                # and GLCM windows...
                if self.standardize_individual == 'add' or self.standardize_add:
                    xx_img = stats_utils.standardize_norm(
                        xx_img, if_norm=self.norm_on_std,
                        get_perc_qmin_qmax=self.norm_min_max,
                        norm_clip=self.norm_clip)

            xx_lst.append(xx_img)

        if self.calc_pca == 'all':
            # --- calculate PCA over all additional bands ------
            # (not over greyscale band)
            xx_merge = merge_img_to_analyse(
                x_img, xx_lst, resampling_type='bilinear',
                band_prefix_lst=None, ref_img_add=False,
                nodata_out=np.nan)
            if torch.cuda.is_available() and self.gpu_no != 99:
                with cupy.cuda.Device(self.gpu_no):
                    xx_merge = self.z_stats.get_PCA(
                        xx_merge, col_use=None,
                        n_components=self.PCA_components,
                        whiten=False, bands=None)
            else:
                xx_merge = self.z_stats.get_PCA(
                    xx_merge, col_use=None,
                    n_components=self.PCA_components,
                    whiten=False, bands=None)

            # if calculated pca always use individual std afterwards
            # (see comment above)
            if self.standardize_individual == 'add' or self.standardize_add:
                xx_merge = stats_utils.standardize_norm(
                    xx_merge, if_norm=self.norm_on_std,
                    get_perc_qmin_qmax=self.norm_min_max,
                    norm_clip=self.norm_clip)

            xx_lst = [xx_merge]
            del xx_merge

        x_img = merge_img_to_analyse(
            x_img, xx_lst, resampling_type='bilinear',
            band_prefix_lst=None, ref_img_add=True, nodata_out=np.nan)

        if self.standardize_individual == 'all':
            # --- for all bands use tile-based standardisation ----
            # requires background in greyscale image to be nans not zeros!
            x_img = stats_utils.standardize_norm(
                x_img, if_norm=self.norm_on_std,
                get_perc_qmin_qmax=self.norm_min_max,
                norm_clip=self.norm_clip)

        del xx_lst, xx_img
        return x_img

    def plot_within_loader(self, xarray_data, xarray_seg, scale_stage, idx,
                           weights=None):
        '''
        changed according to
        https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
        title e.g. orig or augmented
        '''
        bands = xarray_data.band.values.tolist()
        n_bands = len(np.array(self.add_bands).ravel())
        if scale_stage.find('orig') > -1:
            self.fig, self.ax = plt.subplots(
                nrows=2, ncols=n_bands + 3, figsize=(5*(n_bands + 3), 10))
            ax_num = 0
        else:
            ax_num = 1

        # orig image
        image, mask = xarray_data, xarray_seg
        plotting_utils.plot_all_bands_mask(
            self.fig, self.ax[ax_num, :], image, mask, bands, scale_stage,
            weights=weights)

        if scale_stage.find('orig') > -1:
            diff_b = n_bands
            for i_del in range(1, diff_b+1):
                plotting_utils.make_patch_spines_invisible(
                    self.ax[0, i_del], 1)
        else:
            file_name = (
                f'check_input_data_{scale_stage}_idx{idx}_count{self.count_plot_hist}')
            self.fig.savefig(
                os.path.join(self.path_out, file_name + '.png'),
                format='png')

            self.count_plot += 1
            plt.close(self.fig)
        return

    def plot_hist_within_loader(self, img, idx, scale_stage):
        '''
        img is xarray
        '''
        file_name_out = (
            f'check_input_data_{scale_stage}_idx{idx}_count{self.count_plot_hist}_hist.pdf')

        plotting_utils.plot_hist_from_img(
            img, self.path_out, file_name_out)

        self.count_plot_hist += 1
        return


class CustomDataset_augment_ScaleMerge(
        Dataset, CustomDataset_ScaleMerge_base):
    '''
    Scale features and merge to single xarray data_array ()
    '''
    def __init__(self, data_files, seg_files,
                 file_id_lst, file_suffix_lst, z_stats,
                 path_proc, scale_type_dict_GLCM, path_local,
                 x_bands=None, y_bands=None, add_bands=None,
                 width_out=None, height_out=None,
                 calc_pca=False, PCA_components=3,
                 standardize_add=True, norm_on_std=True,
                 norm_min_max_band1=None, take_log=False,
                 gpu_no=7, gpu_no_GLCM=6,
                 augmentations_geom=None, augmentations_col=None,
                 band_lst_col_aug=0, augmentations_range=None,
                 save_files_debug=False, set_to_nan_lst=None,
                 dict_relabel=None,
                 standardize_individual='No', if_std_band1=False,
                 dl_phase='', debug_plot=False, norm_clip=True,
                 exclude_greyscale=False, augmentations_fad=None,
                 sensor_dict=None, feature_add=None):
        '''
        1) If required standardize and scale all features to 0 - 1
        2) If required calculate PCA and either derive stats
            (for prediction data) or use stats from predition data to
            scale tiles (!!! PCAs will vary from tile to tile if
            calculate separately)
        3) augment, if required calculate GLCMs, standardise and
            merge all features
            output: into separate data and segmentatio (with labels files)
            ..._data_merge.tif & ..._seg_merge.tif

        proc steps:
        1) From base file name: get all required files
            e.g. for
            BLyakhE_HEX1979_B01_train_00_00-00_data.tif
            BLyakhE_HEX1979_B01_train_00_00-00_seg.tif
        2) augment input image and calculate GLCMs
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r05_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r10_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_r10_calc_std.tif
            file_suffix_lst = ['a0-1-2-3_r05_norm_C01',
                'a0-1-2-3_r10_norm_C01', 'r10_calc_std']
        3) scale features
        4) merge all bands (i.e. merge GLCMs calculated with different
            window size and direction to greyscale imagery)
        '''
        # augmentation is imported here since is not required in all other
        # data loaders
        import cnn_workflow.custom_augmentation as custom_augmentation
        self.custom_augmentation = custom_augmentation

        self.debug_plot = debug_plot
        self.dl_phase = dl_phase
        self.data_files = data_files
        self.file_id_lst = file_id_lst
        self.seg_files = seg_files  # this is None for prediction data
        # (full set)
        self.path_local = path_local  # need to use local path due to temporary folder
        # not able to be deleted due to .nsf files if save it on network server

        self.len_orig_files = len(self.data_files)  # amount files
        self.z_stats = z_stats  # is module to calculate PCA

        self.x_bands = x_bands
        self.y_bands = y_bands  # this is None for prediction data
        self.n_y_bands = len(self.y_bands)
        self.add_bands = add_bands  # list with bands to take from each
        # file defined in file_suffix_lst

        # colors to be used for hist plot
        if len(self.add_bands) > 0:
            self.n_bands_plot = len(self.add_bands[0])
        else:
            self.n_bands_plot = 10  # set to 10 for colorplot otherwise get error!

        self.file_suffix_lst = file_suffix_lst  # file suffix of files
        # to be merge with grey scale imagery

        self.calc_pca = calc_pca  #  None is no calculation
        # 'separate' is PCA calculation for each GLCM file separately
        # (e.g. specific window size)
        # 'all' is over all merged additional files (but not including
        # grey scale imagery)
        self.PCA_components = PCA_components  # how many components

        # standardisation
        self.standardize_add = standardize_add  # if want to standardscale
        # channels from file_suffix_lst (True or False)
        self.standardize_individual = standardize_individual  # if standardize
        # according to current patch. No normalization (if == "add"
        # then only the additional bands are standardized with "all"
        # all bands are standardized)
        self.if_std_band1 = if_std_band1  # if standard scale band 1
        # with values form stats file

        self.norm_min_max_band1 = norm_min_max_band1  # if normalize grey
        # scale band (e.g. [0, 255]). If set to None then no normalzation
        # is done
        if norm_on_std == 999:
            self.norm_on_std = False
            self.norm_min_max = [0.5, 99.5]
            self.norm_clip = False
        else:
            self.norm_on_std = True  # # if normalize standardized
            self.norm_min_max = [norm_on_std, 100 - norm_on_std]
            self.norm_clip = norm_clip

        self.take_log = take_log  # True or False

        # window sizes for tiles (used to crop edges effects if
        # original tile has a larger size)
        # will be none for prediction data
        self.width_out = width_out
        self.height_out = height_out

        self.gpu_no = gpu_no
        self.gpu_no_GLCM = gpu_no_GLCM

        self.path_out = path_proc
        #if path_suffix is not None:
        #    self.path_out = os.path.join(path_proc, path_suffix)
        #    if not os.path.isdir(self.path_out):
        #        os.mkdir(self.path_out)
        self.augmentations_geom = augmentations_geom
        self.augmentations_col = augmentations_col
        self.augmentations_range = augmentations_range
        self.band_lst_col_aug = band_lst_col_aug
        self.augmentations_fad = augmentations_fad

        self.scale_type_dict_GLCM = scale_type_dict_GLCM  # dict with scale
        # type per file used for GLCM calc
        self.save_files_debug = save_files_debug  # saved intermediate files if debug

        self.set_to_nan_lst = set_to_nan_lst
        self.dict_relabel = dict_relabel

        self.n_channels = len(np.array(self.add_bands).ravel()) + len(self.x_bands)
        if exclude_greyscale:
            self.n_channels = self.n_channels - 1
        self.exclude_greyscale = exclude_greyscale

        # plot counting for file plot file name
        self.count_plot = 0
        self.count_plot_hist = 0

        self.sensor_dict = sensor_dict  # to assign which sensor was taken
        # (for sensor specific augmentation e.g. FAD)
        self.feature_add = feature_add  # add sensor id
        if self.feature_add is not None and 'sensor' in self.feature_add:
            self.add_sensor_nr = True
            self.n_channels = self.n_channels + 1
        else:
            self.add_sensor_nr = False

        self.sensor_nr_dict = {'HEX': 0.5, 'SPOT': 1.0}

    def augment_single_int(self, x_img, y_img, sensor_id=None):
        '''
        Augment single tiles
        1) reformat input fetures and labels (albumentaitons
            requires numpy arrays with dimenstion [y, x, bands])
        2) Augment image/label pairs using self.custom_augmentation.augmentation_chain()

        Parameters
        ------
        - x_img: xarray
            tile with input data
        - y_img: xarray
            corresponding tile with training labels
        - sensor_id: sensor key
            key to select correct FAD (fourier domain adaptartion)
            augmentation

        Returns
        -------
        - x, y: numpy array
            augmented feature/label pair
        - prefix: str
            augmentation prefix

        Inputs for albumenations are numpy arrays with dimens
        !!! input type must be integer with zero as fill value!!!
        '''
        # ---- reformat input
        # create numpy data
        x = x_img.values
        # for albumentations the channels must be at last dimension
        # thus use moveaxis to rearrange to [y, x, bands]

        x = np.moveaxis(x, 0, -1)  # .astype('float32')
        # !!! Note: converting to float should not be done as with new
        # version albumentations changing range to float would not
        # work otherwise

        # create numpy class array
        y = y_img.values.squeeze()
        if self.n_y_bands > 1:
            y = np.moveaxis(y, 0, -1)

        # replace nans with zeros in grey scale imagery
        # not needed here as ther shoud not be any nan
        # x = np.where(np.isnan(x), 0, x)

        # do augmentation using self.custom_augmentation.augmentation_chain()
        if (self.augmentations_geom is not None
            or self.augmentations_col is not None
            or self.augmentations_fad is not None):
            x_orig = x.copy()
            y_orig = y.copy()
            x, y = self.custom_augmentation.augmentation_chain(
                x_orig, y_orig, self.augmentations_geom,
                self.augmentations_col, self.augmentations_range,
                self.band_lst_col_aug,
                augmentations_fad=self.augmentations_fad,
                sensor_id=sensor_id)
            prefix = ('_aug' + '{0:02d}'.format(1))
        else:
            prefix = ''

        # move axis back --> not needed axis is moved when creating image

        return x, y, prefix

    def __getitem__(self, idx):
        """
        get inpt for pytorch

        1) Open data and label pairs and read stats
        2) Apply augmentation
        3) Optional: calculate GLCM features
        4) Convert input features to float for scalinf and merging
        5) Optional: scale (standardise or normalise) input features
        6) Merge all input features into one array
        7) create torch arrays with all inputs (input features,
           classes and weighting. If required, classes are relabelled
           or specific classes can be exclded)

        Note:
        - To check augmentation can use self.debug_plot which creates
            and saves additional plots
        """

        # --- 1) read data, segmentation and stats files
        x_img = self.open_data_img(idx, chunk=None)
        y_img = self.open_seg_img(idx, chunk=None)
        file_id = self.file_id_lst[idx]
        sensor_id = self.sensor_dict[file_id]
        # file_id is e.g. 'BLyakhE_SPOT2018_A01_std4_8bit'

        # ensure that array is square
        if x_img.shape[1] != x_img.shape[2]:
            xmin = min(x_img.shape[1], x_img.shape[2])
            x_img = x_img[:, :xmin, :xmin]
            y_img = y_img[:, :xmin, :xmin]

        # ----- get stats for specific file
        # (stats file must have been read before with get_stats_dict.
        # Which stats file is choosen is defined per labelling area and
        # is defined in parameter file e.g. PARAM['PHASE_STATS_FILE'])
        self.xarr_stats = self.xarr_stats_dict[file_id]

        if self.debug_plot:
            # For checking in debug mode: create plots of raw imagery
            # add orig to imshow
            if y_img.shape[0] > 1:
                w_plot = y_img[1, :, :]
            else:
                w_plot = None
            self.plot_within_loader(x_img, y_img[0, :, :], 'orig', idx,
                                    weights=w_plot)
            # add orig to host plot
            self.plot_hist_within_loader(
                    x_img, idx, 'raw_' + self.dl_phase)

        # --- 2) apply augmentation ---
        # For albumentations the input type for augmentation is float
        # Thus the orig type with uint (e.g. 0 to 2**XX, with 0 as Nan)
        # is first transformed to floats. After augmentation ist is pach
        # transformed to uint as glcm_cupy requires uint values as input
        x, y, prefix = self.augment_single_int(x_img, y_img, sensor_id)

        # extract file name (without file extension) to be used as prefix
        # for naming GLCM files
        file_name_data = os.path.basename(
            self.data_files[idx]).split('.')[0]
        file_name_seg = os.path.basename(
            self.seg_files[idx]).split('.')[0]

        # Note: GLCM calculation is done as a subprocess. This is a workaround
        # due to CUDA initialisation issues.
        # Thus the augmented imagery needs to be saved as well as the
        # generated GLCM features, which later will be read and
        # trasformed to torch tensors. Data files will be saved in temp
        # folder which will be deleted subsequently.
        temp_prefix = os.path.join(self.path_local, 'temp_')
        with tempfile.TemporaryDirectory(prefix=temp_prefix) as tempdir:
            # create new file name and path for augmted image
            new_fn_data = file_name_data + prefix
            new_fn_seg = file_name_seg + prefix
            new_fp_data = os.path.join(tempdir, new_fn_data)
            new_fp_seg = os.path.join(tempdir, new_fn_seg)

            if len(self.file_suffix_lst) > 0:
                # ------- 3) Apply GLCM calculation ------
                # (GLCM features are defined in file_suffix_lst)
                # create image from augmented numpy array and save
                x_img, y_img = xy_xarray_to_img_save(
                    x_img, x, y_img, y, tempdir,
                    new_fn_data + '.tif', new_fn_seg + '.tif',
                    y_bands=self.y_bands)

                # run GLCM calculation (done in new process and from disk)
                path_to_module = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '../../texture_calc'))
                cmd_inp = [
                    'python',
                    os.path.join(path_to_module, 'MAIN_calc_texture_single_file.py'),
                    '--PARAM_FILE', 'PARAM06_calc_texture_train',
                    '--GPU_NUM', str(self.gpu_no_GLCM),
                    new_fp_data + '.tif',
                    ':'.join(self.file_suffix_lst),
                    self.scale_type_dict_GLCM[file_id]]
                out = subprocess.run(
                    cmd_inp, check=False, capture_output=True)
                # here use captured output since otherwise error is not
                # captured with subprocess.DEVNULL however if do use
                # stderr=subprocess.STDOUT then also get GLCM outputs...
                if out.returncode != 0:
                    # if error, print it to stderr, and exit
                    print("there was an error :\n")
                    print(out.stderr.decode("utf-8"))
                    sys.exit('Error occured in subprocess')
            else:
                # ---- without GLCM features -----
                # without calculartion of GLCM there is no need to save
                # the data
                x_img, y_img = xy_xarray_to_img_save(
                    x_img, x, y_img, y, None,
                    new_fn_data + '.tif', new_fn_seg + '.tif',
                    y_bands=self.y_bands)

            # --- 4) convert input features to float ----
            # this is done to make sure that there are e.g skipnan can
            # be used in standardisation.
            # However, actually it is not required for all cases
            # (e.g. is use stats from file) but it is safer in this way...
            # Another option would be to read the files directly with mask
            # nan (will then be float). However, this would require
            # transforming the values to int for the GLCM calculation
            x_img = geo_utils.convert_img_to_dtype(
                x_img, dtype_out='float64', nodata_out=np.nan,
                replace_zero_with_nan=True)

            # --- 5) scale according to options and 6) merge all features.
            # x_img here is still just grey scale image.
            # The GLCMs are read from file
            x_img = self.scale_merge(
                x_img, new_fp_data, chunk=None, y_img=y_img)
            # Note: scale_merge also removes edge effect of GLCM calculation
            # (both at tile esges as well as area-AOI edges)

            # ONLY for tiles:
            # cropping is done below with augment (gives same result)
            # clears GLCM artefacts at tile edges
            if self.width_out is not None or self.height_out is not None:
                # crop label tile (y_img) to window size
                # data tile (x_img) is already cropped within scale_merge
                # (cropping is required to avoide edge effects of GLCM
                # calculation)
                img_shape = y_img.shape
                y_img, y_crop_id, y_crop_id = geo_utils.center_crop_img(
                    y_img, img_shape[1], img_shape[2],
                    self.width_out, self.height_out)

            if self.save_files_debug:
                # to debig and check, save merged imagery
                # !!! is saved to tempdir
                # (for plotting of single tiles into export folder
                #  use self.debug_plot see below)
                file_path = os.path.join(
                    tempdir, new_fn_data  + '_merge.tif')
                x_img.rio.to_raster(file_path, driver='GTiff')
                self.band_x_out = x_img.band.values.tolist()

                file_path = os.path.join(
                    tempdir, new_fn_seg + '_merge.tif')
                y_img.rio.to_raster(file_path, driver='GTiff')
                self.band_y_out = y_img.band.values.tolist()

            # get numpy array
            x = x_img.values
            y = y_img.values

            if self.debug_plot:
                # plot all augmented features with imshow
                if y_img.shape[0] > 1:
                    w_plot = y_img[1, :, :]
                else:
                    w_plot = None
                self.plot_within_loader(
                    x_img, y_img[0, :, :],
                    'augmented_' + self.dl_phase, idx, weights=w_plot)
                # add to hist plot
                self.plot_hist_within_loader(
                    x_img, idx, 'augmented_' + self.dl_phase)

            del x_img, y_img, new_fp_data, new_fp_seg

        # ----- create torch arrays with all inputs
        # (input features, classes and weighting. If required, classes
        # are relabelled or specific classes can be exclded)

        # create torch array with weights if provided
        if self.n_y_bands > 1:
            w_out = y[1, :, :]/100
            w_torch = torch.tensor(w_out, dtype=torch.torch.float32)
        else:
            w_torch = torch.tensor([], dtype=torch.torch.float32)

        # extract class labels
        y_out = y[0, :, :]
        # relabel classes is required
        if self.dict_relabel is not None:
            y_out = relabel_arr(y_out, self.dict_relabel)

        # replace nans with zero as pytorch cant cope with nans
        x = np.where(np.isnan(x), 0, x)

        # if required, set specific class labels to NaN-class (0)
        if self.set_to_nan_lst is not None:
            x, y_out = set_to_nan(
                x, y_out, self.set_to_nan_lst, fill_val=0)

        # excluded greyscale if should not be included in training
        if self.exclude_greyscale:
            x = x[1:, :, :]

        # add sensor number if this should be inclued in training
        # (might help to learn differences of imagery?)
        if self.add_sensor_nr:
            x = np.concatenate(
                [x, np.zeros([1] + list(x.shape[1:]))+self.sensor_nr_dict[sensor_id]],
                axis=0)

        # create torch tensors for data and labels
        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = torch.tensor(y_out, dtype=torch.torch.int64)

        return x_torch, y_torch, w_torch

    def __len__(self):
        return len(self.data_files)


class CustomDataset_ScaleMerge(
        Dataset, CustomDataset_ScaleMerge_base):
    '''
    Scale features and merge to single xarray data_array
    This is the same as CustomDataset_augment_ScaleMerge() but without
    augmentation and GLCM recalculation
    '''
    def __init__(self, data_files, seg_files,
                 file_id_lst, file_suffix_lst, z_stats,
                 path_proc,
                 x_bands=None, y_bands=None, add_bands=None,
                 width_out=None, height_out=None,
                 calc_pca=False, PCA_components=3,
                 standardize_add=True, norm_on_std=True,
                 norm_min_max_band1=None, take_log=False,
                 gpu_no=7,
                 save_files_debug=False, set_to_nan_lst=None,
                 dict_relabel=None,
                 standardize_individual='No', if_std_band1=False,
                 dl_phase='', norm_clip=True, exclude_greyscale=False,
                 sensor_dict=None, feature_add=None):
        '''
        1) If required standardize and scale all features to 0 - 1
        2) If required calculate PCA and either derive stats
            (for prediction data) or use stats from predition data to
            scale tiles (!!! PCAs wil vary from tile to tile if
            calculate separately)
        3) Merge all features
            output: into separate data and segmentatio (with labels files)
            ..._data_merge.tif & ..._seg_merge.tif

        proc steps:
        1) From base file name: get all required files
            e.g. for
            BLyakhE_HEX1979_B01_train_00_00-00_data.tif
            BLyakhE_HEX1979_B01_train_00_00-00_seg.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r05_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r10_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_r10_calc_std.tif
            file_suffix_lst = ['a0-1-2-3_r05_norm_C01',
                'a0-1-2-3_r10_norm_C01', 'r10_calc_std']
        2) scale img
        3) merge img
        '''
        self.debug_plot = False
        self.dl_phase = dl_phase
        self.data_files = data_files
        self.file_id_lst = file_id_lst
        self.seg_files = seg_files  # this is None for prediction data
        # (full set)

        self.len_orig_files = len(self.data_files)  # amount files
        self.z_stats = z_stats  # is module to calculate PCA

        self.x_bands = x_bands
        self.y_bands = y_bands  # this is None for prediction data
        self.n_y_bands = len(self.y_bands)
        self.add_bands = add_bands  # list with bands to take from each
        # file defined in file_suffix_lst

        # colors to be used for hist plot
        if len(self.add_bands) > 0:
            self.n_bands_plot = len(self.add_bands[0])
        else:
            self.n_bands_plot = 10  # set to 10 for colorplot otherwise get error!

        self.file_suffix_lst = file_suffix_lst  # file suffix of files
        # to be merge with grey scale imagery

        self.calc_pca = calc_pca  #  None is no calculation
        # 'separate' is PCA calculation for each GLCM file separately
        # (e.g. specific window size)
        # 'all' is over all merged additional files (but not including
        # grey scale imagery)
        self.PCA_components = PCA_components  # how many components

        # standardisation
        self.standardize_add = standardize_add  # if want to standardscale
        # channels from file_suffix_lst (True or False)
        self.standardize_individual = standardize_individual  # if standardize
        # according to current patch. No normalization (if == "add"
        # then only the additional bands are standardized with "all"
        # all bands are standardized)
        self.if_std_band1 = if_std_band1  # if standard scale band 1
        # with values form stats file

        self.norm_min_max_band1 = norm_min_max_band1  # if normalize grey
        # scale band (e.g. [0, 255]). If set to None then no normalzation
        # is done
        if norm_on_std == 999:
            self.norm_on_std = False
            self.norm_min_max = [0.5, 99.5]
            self.norm_clip = False
        else:
            self.norm_on_std = True  # # if normalize standardized
            self.norm_min_max = [norm_on_std, 100 - norm_on_std]
            self.norm_clip = norm_clip

        self.take_log = take_log  # True or False

        # window sizes for tiles (used to crop edges effects if
        # original tile has a larger size)
        # will be none for prediction data
        self.width_out = width_out
        self.height_out = height_out

        self.gpu_no = gpu_no

        self.path_out = path_proc
        #if path_suffix is not None:
        #    self.path_out = os.path.join(path_proc, path_suffix)
        #    if not os.path.isdir(self.path_out):
        #        os.mkdir(self.path_out)

        self.save_files_debug = save_files_debug  # saved intermediate files if debug

        self.set_to_nan_lst = set_to_nan_lst
        self.dict_relabel = dict_relabel

        self.n_channels = len(np.array(self.add_bands).ravel()) + len(self.x_bands)
        if exclude_greyscale:
            self.n_channels = self.n_channels - 1
        self.exclude_greyscale = exclude_greyscale

        # plot counting for file plot file name
        self.count_plot = 0
        self.count_plot_hist = 0

        self.sensor_dict = sensor_dict  # to assign which sensor was taken
        self.feature_add = feature_add  # add sensor id
        if self.feature_add is not None and 'sensor' in self.feature_add:
            self.add_sensor_nr = True
            self.n_channels = self.n_channels + 1
        else:
            self.add_sensor_nr = False

        self.sensor_nr_dict = {'HEX': 0.5, 'SPOT': 1.0}

    def __getitem__(self, idx):
        # get data

        x_img = self.open_data_img(idx, chunk=None)  # open greyscale image
        # (the GLCMs are added within scale_merge())
        y_img = self.open_seg_img(idx, chunk=None)
        file_id = self.file_id_lst[idx]
        sensor_id = self.sensor_dict[file_id]

        basepath = self.data_files[idx].split('.')[0].split('_raw')[0]
        basename_data = os.path.basename(basepath)
        basename_seg = os.path.basename(
            self.seg_files[idx]).split('.')[0]
        # file_id is e.g. 'BLyakhE_SPOT2018_A01_std4_8bit'

        # get stats for specific file
        # (stats file is read before. Which stats file is choosen is
        #  defined per labelling area and is defined in parameter file
        #  e.g. PARAM06_train_model...)
        self.xarr_stats = self.xarr_stats_dict[file_id]

        if self.debug_plot:
            # For checking in debug mode: create plots of raw imagery
            # add orig to imshow
            if y_img.shape[0] > 1:
                w_plot = y_img[1, :, :]
            else:
                w_plot = None
            self.plot_within_loader(x_img, y_img[0, :, :], 'orig', idx,
                                    weights=w_plot)
            self.plot_hist_within_loader(
                    x_img, idx, 'raw_' + self.dl_phase)

        # convert to float due to merge later
        # to make sure that there are no issues when calculate means etc
        # for standardization
        # however actually it is not required for all cases.
        # e.g. is use stats from file but it is safer in this way...
        # however to just calculate the stats could also read the
        # files with mask nan (will then be float) but this isnot
        # possibl ehere since need inisially integer values to
        # calculate the GLCMs
        x_img = geo_utils.convert_img_to_dtype(
            x_img, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True)

        # scale and merging. x_img here is still just grey scale image
        # the GLCMs are red from file
        x_img = self.scale_merge(
            x_img, basepath, chunk=None, y_img=y_img)

        # follwoing is done within scale_merge
        # mask merged data according to y (clears AOI edge effects)
        #x_img = x_img.where(~(y_img.sel(band=self.y_bands[0]) == 0).data.squeeze())

        # ONLY for tiles>
        # cropping is done below with augment (gives same result)
        # clears GLCM artefacts at tile edges
        if self.width_out is not None or self.height_out is not None:
            #img_shape = x_img.shape
            # follwoing is done within scale_merge
            #x_img, x_crop_id, y_crop_id = geo_utils.center_crop_img(
            #    x_img, img_shape[1], img_shape[2],
            #    self.width_out, self.height_out)
            img_shape = y_img.shape
            y_img, y_crop_id, y_crop_id = geo_utils.center_crop_img(
                y_img, img_shape[1], img_shape[2],
                self.width_out, self.height_out)

        if self.save_files_debug:
            # save to file
            path_img = os.path.dirname(self.data_files[idx])
            file_path = os.path.join(
                path_img, basename_data  + '_merge.tif')
            x_img.rio.to_raster(file_path, driver='GTiff')
            self.band_x_out = x_img.band.values.tolist()

            file_path = os.path.join(
                path_img, basename_seg + '_merge.tif')
            y_img.rio.to_raster(file_path, driver='GTiff')
            self.band_y_out = y_img.band.values.tolist()

        if self.debug_plot:
            if y_img.shape[0] > 1:
                w_plot = y_img[1, :, :]
            else:
                w_plot = None
            self.plot_within_loader(
                x_img, y_img[0, :, :], 'scaled_' + self.dl_phase, idx,
                weights=w_plot)
            self.plot_hist_within_loader(
                x_img, idx, 'scaled_' + self.dl_phase)

        x = x_img.values
        y = y_img.values  #.squeeze()
        del x_img, y_img

        if self.n_y_bands > 1:
            w_out = y[1, :, :]/100
            w_torch = torch.tensor(w_out, dtype=torch.torch.float32)
        else:
            w_torch = torch.tensor([], dtype=torch.torch.float32)

        y_out = y[0, :, :]
        if self.dict_relabel is not None:
            y_out = relabel_arr(y_out, self.dict_relabel)

        # is this needed???? yes otherwise it doesnt seem to
        # converge !!
        x = np.where(np.isnan(x), 0, x)

        if self.set_to_nan_lst is not None:
            x, y_out = set_to_nan(
                x, y_out, self.set_to_nan_lst, fill_val=0)

        # excluded greyscale if should not be included in training
        if self.exclude_greyscale:
            x = x[1:, :, :]

        # add sensor number if this should be inclued in training
        # (might help to learn differences of imagery?)
        if self.add_sensor_nr:
            x = np.concatenate(
                [x, np.zeros([1] + list(x.shape[1:]))+self.sensor_nr_dict[sensor_id]],
                axis=0)

        # create torch tensors for data and labels
        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = torch.tensor(y_out, dtype=torch.torch.int64)

        return x_torch, y_torch, w_torch

    def __len__(self):
        return len(self.data_files)


class CustomDataset_ScaleMerge_Test(
        Dataset, CustomDataset_ScaleMerge_base):
    '''
    Scale features and merge to single xarray data_array
    This is the same as CustomDataset_augment_ScaleMerge() but without
    augmentation and GLCM recalculation
    '''
    def __init__(self, data_files, seg_files,
                 file_id_lst, file_suffix_lst, z_stats,
                 path_proc,
                 x_bands=None, y_bands=None, add_bands=None,
                 width_out=None, height_out=None,
                 calc_pca=False, PCA_components=3,
                 standardize_add=True, norm_on_std=True,
                 norm_min_max_band1=None, take_log=False,
                 gpu_no=7,
                 save_files_debug=False, set_to_nan_lst=None,
                 dict_relabel=None,
                 standardize_individual='No', if_std_band1=False,
                 dl_phase='', debug_plot=False, norm_clip=True,
                 exclude_greyscale=False,
                 sensor_dict=None, feature_add=None):
        '''
        1) If required standardize and scale all features to 0 - 1
        2) If required calculate PCA and either derive stats
            (for prediction data) or use stats from predition data to
            scale tiles (!!! PCAs wil vary from tile to tile if
            calculate separately)
        3) Merge all features
            output: into separate data and segmentatio (with labels files)
            ..._data_merge.tif & ..._seg_merge.tif

        proc steps:
        1) From base file name: get all required files
            e.g. for
            BLyakhE_HEX1979_B01_train_00_00-00_data.tif
            BLyakhE_HEX1979_B01_train_00_00-00_seg.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r05_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r10_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_r10_calc_std.tif
            file_suffix_lst = ['a0-1-2-3_r05_norm_C01',
                'a0-1-2-3_r10_norm_C01', 'r10_calc_std']
        2) scale img
        3) merge img
        '''
        self.debug_plot = debug_plot
        self.dl_phase = dl_phase
        self.data_files = data_files
        self.file_id_lst = file_id_lst
        self.seg_files = seg_files  # this is None for prediction data
        # (full set)

        self.len_orig_files = len(self.data_files)  # amount files
        self.z_stats = z_stats  # is module to calculate PCA

        self.x_bands = x_bands
        self.y_bands = y_bands  # this is None for prediction data
        self.n_y_bands = len(self.y_bands)
        self.add_bands = add_bands  # list with bands to take from each
        # file defined in file_suffix_lst

        # colors to be used for hist plot
        if len(self.add_bands) > 0:
            self.n_bands_plot = len(self.add_bands[0])
        else:
            self.n_bands_plot = 10  # set to 10 for colorplot otherwise get error!

        self.file_suffix_lst = file_suffix_lst  # file suffix of files
        # to be merge with grey scale imagery

        self.calc_pca = calc_pca  #  None is no calculation
        # 'separate' is PCA calculation for each GLCM file separately
        # (e.g. specific window size)
        # 'all' is over all merged additional files (but not including
        # grey scale imagery)
        self.PCA_components = PCA_components  # how many components

        # standardisation
        self.standardize_add = standardize_add  # if want to standardscale
        # channels from file_suffix_lst (True or False)
        self.standardize_individual = standardize_individual  # if standardize
        # according to current patch. No normalization (if == "add"
        # then only the additional bands are standardized with "all"
        # all bands are standardized)
        self.if_std_band1 = if_std_band1  # if standard scale band 1
        # with values form stats file

        self.norm_min_max_band1 = norm_min_max_band1  # if normalize grey
        # scale band (e.g. [0, 255]). If set to None then no normalzation
        # is done
        if norm_on_std == 999:
            self.norm_on_std = False
            self.norm_min_max = [0.5, 99.5]
            self.norm_clip = False
        else:
            self.norm_on_std = True  # # if normalize standardized
            self.norm_min_max = [norm_on_std, 100 - norm_on_std]
            self.norm_clip = norm_clip

        self.take_log = take_log  # True or False

        # window sizes for tiles (used to crop edges effects if
        # original tile has a larger size)
        # will be none for prediction data
        self.width_out = width_out
        self.height_out = height_out

        self.gpu_no = gpu_no

        self.path_out = path_proc
        #if path_suffix is not None:
        #    self.path_out = os.path.join(path_proc, path_suffix)
        #    if not os.path.isdir(self.path_out):
        #        os.mkdir(self.path_out)

        self.save_files_debug = save_files_debug  # saved intermediate files if debug

        self.set_to_nan_lst = set_to_nan_lst
        self.dict_relabel = dict_relabel

        self.n_channels = len(np.array(self.add_bands).ravel()) + len(self.x_bands)
        if exclude_greyscale:
            self.n_channels = self.n_channels - 1
        self.exclude_greyscale = exclude_greyscale

        # plot counting for file plot file name
        self.count_plot = 0
        self.count_plot_hist = 0

        self.sensor_dict = sensor_dict  # to assign which sensor was taken
        # (for sensor specific augmentation e.g. FAD)

        self.feature_add = feature_add  # add sensor id
        if self.feature_add is not None and 'sensor' in self.feature_add:
            self.add_sensor_nr = True
            self.n_channels = self.n_channels + 1
        else:
            self.add_sensor_nr = False

        self.sensor_nr_dict = {'HEX': 0.5, 'SPOT': 1.0}

    def __getitem__(self, idx):
        # get data

        x_img = self.open_data_img(idx, chunk=None)  # open greyscale image
        # (the GLCMs are added within scale_merge())
        basepath = self.data_files[idx].split('.')[0].split('_raw')[0]
        basename_data = os.path.basename(basepath)

        y_img = self.open_seg_img(idx, chunk=None)
        file_id = self.file_id_lst[idx]
        basename_seg = os.path.basename(
            self.seg_files[idx]).split('.')[0]
        # file_id is e.g. 'BLyakhE_SPOT2018_A01_std4_8bit'
        sensor_id = self.sensor_dict[file_id]

        # get stats for specific file
        # (stats file is read before. Which stats file is choosen is
        #  defined per labelling area and is defined in parameter file
        #  e.g. PARAM06_train_model...)
        self.xarr_stats = self.xarr_stats_dict[file_id]

        if self.debug_plot:
            self.plot_within_loader(x_img, y_img[0, :, :], 'orig', idx)
            self.plot_hist_within_loader(
                    x_img, idx, 'raw_' + self.dl_phase)

        # convert to float due to merge later
        # to make sure that there are no issues when calculate means etc
        # for standardization
        # however actually it is not required for all cases.
        # e.g. is use stats from file but it is safer in this way...
        # however to just calculate the stats could also read the
        # files with mask nan (will then be float) but this isnot
        # possibl ehere since need inisially integer values to
        # calculate the GLCMs
        x_img = geo_utils.convert_img_to_dtype(
            x_img, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True)

        # scale and merging. x_img here is still just grey scale image
        # the GLCMs are red from file
        x_img = self.scale_merge(x_img, basepath, chunk=None,
                                 y_img=y_img)

        # mask merged data according to y (clears AOI edge effects)
        #x_img = x_img.where(~(y_img.sel(band=self.y_bands[0]) == 0).data.squeeze())

        # ONLY for tiles>
        # cropping is done below with augment (gives same result)
        # clears GLCM artefacts at tile edges
        if self.width_out is not None or self.height_out is not None:
            #img_shape = x_img.shape
            #x_img, x_crop_id, y_crop_id = geo_utils.center_crop_img(
            #    x_img, img_shape[1], img_shape[2],
            #    self.width_out, self.height_out)
            # (x_img was center cropped in scale_merge. Thus is not
            #  required here)
            img_shape = y_img.shape
            y_img, y_crop_id, y_crop_id = geo_utils.center_crop_img(
                y_img, img_shape[1], img_shape[2],
                self.width_out, self.height_out)

        if self.save_files_debug:
            # save to file
            path_img = os.path.dirname(self.data_files[idx])
            file_path = os.path.join(
                path_img, basename_data  + '_merge.tif')
            x_img.rio.to_raster(file_path, driver='GTiff')
            self.band_x_out = x_img.band.values.tolist()

            file_path = os.path.join(
                path_img, basename_seg + '_merge.tif')
            y_img.rio.to_raster(file_path, driver='GTiff')
            self.band_y_out = y_img.band.values.tolist()

        if self.debug_plot:
            self.plot_within_loader(
                x_img, y_img[0, :, :], 'scaled', idx)
            self.plot_hist_within_loader(
                x_img, idx, 'scaled_' + self.dl_phase)

        x = x_img.values
        y = y_img.values
        x_coords = x_img.coords['x'].values
        y_coords = x_img.coords['y'].values
        x_img.close()
        y_img.close()
        del x_img, y_img

        y_out = y[0, :, :]
        if self.dict_relabel is not None:
            y_out = relabel_arr(y_out, self.dict_relabel)

        # required for torch
        x = np.where(np.isnan(x), 0, x)

        if self.set_to_nan_lst is not None:
            x, y_out = set_to_nan(
                x, y_out, self.set_to_nan_lst, fill_val=0)

        if self.exclude_greyscale:
            x = x[1:, :, :]

        if self.add_sensor_nr:
            x = np.concatenate(
                [x, np.zeros([1] + list(x.shape[1:]))+self.sensor_nr_dict[sensor_id]], axis=0)

        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = torch.tensor(y_out, dtype=torch.torch.int64)

        if self.width_out is None or self.height_out is None:
            # Check image size if window size was not set.
            # Use padding in case file is not dividable
            # with 32 use padding
            # check image size
            img_size = x.shape[-2:]
            h_r = img_size[0]%32
            w_r = img_size[1]%32
            if h_r > 0 or w_r > 0:
                # pad input is hw namy pixels are added to
                # (left, right, top, bottom)
                x_torch = torch.functional.F.pad(
                    input=x_torch, pad=(0, 32 - w_r, 0, 32 - h_r),
                    mode='constant', value=0)
                y_torch = torch.functional.F.pad(
                    input=y_torch, pad=(0, 32 - w_r, 0, 32 - h_r),
                    mode='constant', value=0)
                # after the prediction the file additional padding is
                # removed again

                # need also to bad coordinates as input for training
                # from data loader needs alwayas to have same size, otherwise
                # get resizing error
                # (nans will be removed in custom_train run_test)
                x_coords = np.pad(x_coords, (0, 32 - w_r),
                                  'constant', constant_values=(np.nan, np.nan))
                y_coords = np.pad(y_coords, (0, 32 - h_r),
                                  'constant', constant_values=(np.nan, np.nan))

        # in addition use seg file name as output. This is for saving
        # predicted .tif files
        return x_torch, y_torch, self.seg_files[idx], x_coords, y_coords

    def __len__(self):
        return len(self.data_files)


class CustomDataset_ScaleMerge_pred(
        Dataset, CustomDataset_ScaleMerge_base):
    """
    Scale features and merge to single xarray data_array
    This is the same as CustomDataset_augment_ScaleMerge() but without
    augmentation and GLCM recalculation

    features can are loaded and preprocessed via __getitem__
    the following steps are inluded:
      - open data tile, greyscale image(as xarray)
      - within scale_merge:
          - open additional features (GLCMs)
          - if required, standardize and/or normalize image depending
            on selected standardisation type: either use pre-calculated
            training area stats or calculate tile specific
          - clear edge effects
          - merge all features into one xarray
      - again center crop and pad to fit window size if required
      - create torch tensors with data

    """
    def __init__(self, data_files,
                 file_id_lst, file_suffix_lst, z_stats,
                 path_proc,
                 x_bands=None, add_bands=None,
                 width_out=None, height_out=None,
                 calc_pca=False, PCA_components=3,
                 standardize_add=True, norm_on_std=True,
                 norm_min_max_band1=None, take_log=False,
                 gpu_no=7, save_files_debug=False,
                 standardize_individual='No', if_std_band1=False,
                 dl_phase='', debug_plot=False,
                 norm_clip=True,
                 exclude_greyscale=False,
                 aug_col=None,
                 sensor_dict=None, feature_add=None):

        self.debug_plot = debug_plot
        self.dl_phase = dl_phase
        self.data_files = data_files
        self.file_id_lst = file_id_lst
        self.seg_files = None  # this is None for prediction data
        # (full set)

        self.len_orig_files = len(self.data_files)  # amount files
        self.z_stats = z_stats  # is module to calculate PCA

        self.x_bands = x_bands
        if x_bands is None:
            self.x_bands = geo_utils.get_bands(self.data_files[0])
        else:
            self.x_bands = x_bands

        self.add_bands = add_bands  # list with bands to take from each
        # file defined in file_suffix_lst

        # colors to be used for hist plot
        if len(self.add_bands) > 0:
            self.n_bands_plot = len(self.add_bands[0])
        else:
            self.n_bands_plot = 10  # set to 10 for colorplot otherwise get error!

        self.file_suffix_lst = file_suffix_lst  # file suffix of files
        # to be merge with grey scale imagery

        self.calc_pca = calc_pca  #  None is no calculation
        # 'separate' is PCA calculation for each GLCM file separately
        # (e.g. specific window size)
        # 'all' is over all merged additional files (but not including
        # grey scale imagery)
        self.PCA_components = PCA_components  # how many components

        # standardisation
        self.standardize_add = standardize_add  # if want to standardscale
        # channels from file_suffix_lst (True or False)
        self.standardize_individual = standardize_individual  # if standardize
        # according to current patch. No normalization (if == "add"
        # then only the additional bands are standardized with "all"
        # all bands are standardized)
        self.if_std_band1 = if_std_band1  # if standard scale band 1
        # with values form stats file

        self.norm_min_max_band1 = norm_min_max_band1  # if normalize grey
        # scale band (e.g. [0, 255]). If set to None then no normalzation
        # is done
        if norm_on_std == 999:
            self.norm_on_std = False
            self.norm_min_max = [0.5, 99.5]
            self.norm_clip = False
        else:
            self.norm_on_std = True  # # if normalize standardized
            self.norm_min_max = [norm_on_std, 100 - norm_on_std]
            self.norm_clip = norm_clip

        self.take_log = take_log  # True or False

        # window sizes for tiles (used to crop edges effects if
        # original tile has a larger size)
        # will be none for prediction data
        self.width_out = width_out
        self.height_out = height_out

        self.gpu_no = gpu_no

        self.path_out = path_proc

        self.save_files_debug = save_files_debug  # saved intermediate files if debug

        self.n_channels = len(np.array(self.add_bands).ravel()) + len(self.x_bands)
        if exclude_greyscale:
            self.n_channels = self.n_channels - 1
        self.exclude_greyscale = exclude_greyscale

        # plot counting for file plot file name
        self.count_plot = 0
        self.count_plot_hist = 0

        self.aug_col = aug_col  # is not used there but could be
        # implemented e.g. for using FDA

        self.sensor_dict = sensor_dict  # to assign which sensor was taken
        # (for sensor specific augmentation e.g. FAD)
        self.feature_add = feature_add  # add sensor id
        if self.feature_add is not None and 'sensor' in self.feature_add:
            self.add_sensor_nr = True
            self.n_channels = self.n_channels + 1
        else:
            self.add_sensor_nr = False

        self.sensor_nr_dict = {'HEX': 0.5, 'SPOT': 1.0}

    def __getitem__(self, idx):
        """

        """
        # get data
        x_img = self.open_data_img(idx, chunk=None)  # open greyscale image
        # (the GLCMs are added within scale_merge())

        # extract basepath and basename for further processing
        file_id = self.file_id_lst[idx]
        # file_id is e.g. 'BLyakhE_SPOT2018_A01_std4_8bit'
        sensor_id = self.sensor_dict[file_id]

        # extract basepath and basename for further processing
        basepath = self.data_files[idx].split('.')[0].split('_raw')[0]
        basename_data = os.path.basename(basepath)

        # get stats for specific file
        # (stats file is read before. Which stats file is choosen is
        #  defined per labelling area and is defined in parameter file
        #  e.g. PARAM06_train_model...)
        self.xarr_stats = self.xarr_stats_dict[file_id]

        if self.debug_plot:
            self.plot_hist_within_loader(
                    x_img, idx, 'raw_' + self.dl_phase)

        if self.aug_col is not None:
            # this could for example be used for FDA (fourier domain adaptation)
            # to adjust prediction tiles to another site
            # However self.aug_col would need to have been setup before
            # e.g. with
            # ref_img_val = np.moveaxis(ref_img.values, 0, -1)
            # aug_col = A.Compose([A.FDA([ref_img_val], p=1, read_fn=lambda x: x,
            #                             beta_limit=(0.1, 0.1))], p=1)
            x_img_val = np.moveaxis(x_img.values, 0, -1)
            x_img_val = self.aug_col(image=x_img_val)['image']
            x_img_val = np.moveaxis(x_img_val, -1, 0)
            x_img_val[x_img.values==0] = 0

            # to check
            #fig, ax = plt.subplots(
            #   1, 2, sharex=True, sharey=True, figsize=(11.7, 8.3))
            #   ax[0].imshow(x_img.values[0, :, :], clim=[1, 255], cmap='Greys_r')
            #   ax[1].imshow(x_img_val[0, :, :], clim=[1, 255], cmap='Greys_r')
            #fig.savefig(os.path.join(self.path_out, 'FDA_test.pdf'), format='pdf')

            x_img.values = x_img_val

        # convert to float due to merge later
        # to make sure that there are no issues when calculate means etc
        # for standardization
        # however actually it is not required for all cases.
        # e.g. is use stats from file but it is safer in this way...
        # however to just calculate the stats could also read the
        # files with mask nan (will then be float) but this isnot
        # possibl ehere since need inisially integer values to
        # calculate the GLCMs
        x_img = geo_utils.convert_img_to_dtype(
            x_img, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True)

        # scale and merging. x_img here is still just grey scale image
        # the GLCMs are red from file
        x_img = self.scale_merge(x_img, basepath, chunk=None)

        # ONLY for tiles
        # cropping is done below with augment (gives same result)
        # clears GLCM artefacts at tile edges
        if self.width_out is not None or self.height_out is not None:
            img_shape = x_img.shape
            x_img, x_crop_id, y_crop_id = geo_utils.center_crop_img(
                x_img, img_shape[1], img_shape[2],
                self.width_out, self.height_out)

        if self.save_files_debug:
            # save to file
            path_img = os.path.dirname(self.data_files[idx])
            file_path = os.path.join(
                path_img, basename_data  + '_merge_pred.tif')
            x_img.rio.to_raster(file_path, driver='GTiff')
            self.band_x_out = x_img.band.values.tolist()

        if self.debug_plot:
            self.plot_hist_within_loader(
                x_img, idx, 'scaled_' + self.dl_phase)

        x = x_img.values
        x_coords = x_img.coords['x'].values
        y_coords = x_img.coords['y'].values
        x_img.close()
        del x_img

        # is this needed???? yes otherwise it doesnt seem to
        # converge !!
        x = np.where(np.isnan(x), 0, x)

        if self.exclude_greyscale:
            x = x[1:, :, :]

        # add sensor number if this should be inclued in training
        # (might help to learn differences of imagery?)
        if self.add_sensor_nr:
            x = np.concatenate(
                [x, np.zeros([1] + list(x.shape[1:]))+self.sensor_nr_dict[sensor_id]],
                axis=0)

        x_torch = torch.tensor(x, dtype=torch.float32)

        # check image size
        if self.width_out is None or self.height_out is None:
            img_size = x.shape[-2:]
            h_r = img_size[0]%32
            w_r = img_size[1]%32
            if h_r > 0 or w_r > 0:
                # pad input is hw namy pixels are added to
                # (left, right, top, bottom)
                x_torch = torch.functional.F.pad(
                    input=x_torch, pad=(0, 32 - w_r, 0, 32 - h_r),
                    mode='constant', value=0)
                # after the prediction the file additional padding is removed again

        # in addition use data file name and coordinates as output.
        # This is for saving predictions as .tif files
        return x_torch, self.data_files[idx], x_coords, y_coords

    def __len__(self):
        return len(self.data_files)


class CustomDataset_ScaleMerge_ML(
        Dataset, CustomDataset_ScaleMerge_base):
    '''
    Scale features and merge to single xarray data_array and then converts
    gdf. This is for cuml RF segmentation
    This is was taken from CustomDataset_ScaleMerge()

    !!! Weights are not considered here
    '''
    def __init__(self, data_files, seg_files,
                 file_id_lst, file_suffix_lst,
                 aoi_key_lst, sensor_lst,
                 z_stats,
                 path_proc, bands_all,
                 x_bands, y_bands=None, add_bands=None,
                 calc_pca=False, PCA_components=3,
                 standardize_add=True, norm_on_std=True,
                 norm_min_max_band1=None, take_log=False,
                 gpu_no=7,
                 save_files_debug=False, set_to_nan_lst=None,
                 dict_relabel=None,
                 standardize_individual='No', if_std_band1=False,
                 debug_plot=False, dl_phase='train',
                 norm_clip=True):
        '''
        1) If required standardize and scale all features to 0 - 1
        2) If required calculate PCA and either derive stats
            (for prediction data) or use stats from predition data to
            scale chips (!!! PCAs wil vary from chip to chip if
            calculate separately)
        3) Merge all features
            output: into separate data and segmentatio (with labels files)
            ..._data_merge.tif & ..._seg_merge.tif

        proc steps:
        1) From base file name: get all required files
            e.g. for
            BLyakhE_HEX1979_B01_train_00_00-00_data.tif
            BLyakhE_HEX1979_B01_train_00_00-00_seg.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r05_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_a0-1-2-3_r10_norm_C01.tif
            BLyakhE_HEX1979_B01_train_00_00-00_data_r10_calc_std.tif
            file_suffix_lst = ['a0-1-2-3_r05_norm_C01',
                'a0-1-2-3_r10_norm_C01', 'r10_calc_std']
        2) scale img
        3) merge img
        '''
        self.debug_plot = debug_plot
        self.data_files = data_files
        self.file_id_lst = file_id_lst
        self.seg_files = seg_files  # this is None for prediction data
        # (full set)
        self.aoi_key_lst = aoi_key_lst # specifies aoi this different
        # images when tranforming gdf back to xarray
        self.sensor_lst = sensor_lst  # speciffies which sensor has been
        # used could be used as feature in training to separate HEX from SPOT

        self.len_orig_files = len(self.data_files)  # amount files
        self.z_stats = z_stats  # is module to calculate PCA

        self.dl_phase = dl_phase

        if x_bands is None:
            self.x_bands = geo_utils.get_bands(self.data_files[0])
        else:
            self.x_bands = x_bands
        if y_bands is None and self.seg_files is not None:
            self.y_bands = geo_utils.get_bands(self.seg_files[0])
        else:
            self.y_bands = y_bands

        self.n_y_bands = len(self.y_bands)
        self.add_bands = add_bands  # list with additional bands to take from each
        # file defined in file_suffix_lst
        self.bands_all = bands_all # all bands to use (incl grey scale)

        # colors to be used for hist plot
        if len(self.add_bands) > 0:
            self.n_bands_plot = len(self.add_bands[0])
        else:
            self.n_bands_plot = 10  # set to 10 for colorplot otherwise get error!

        self.file_suffix_lst = file_suffix_lst  # file suffix of files
        # to be merge with grey scale imagery

        self.calc_pca = calc_pca  #  None is no calculation
        # 'separate' is PCA calculation for each GLCM file separately
        # (e.g. specific window size)
        # 'all' is over all merged additional files (but not including
        # grey scale imagery)
        self.PCA_components = PCA_components  # how many components

        # standardisation
        self.standardize_add = standardize_add  # if want to standardscale
        # channels from file_suffix_lst (True or False)
        self.standardize_individual = standardize_individual  # if standardize
        # according to current patch. No normalization (if == "add"
        # then only the additional bands are standardized with "all"
        # all bands are standardized)
        self.if_std_band1 = if_std_band1  # if standard scale band 1
        # with values form stats file

        self.norm_min_max_band1 = norm_min_max_band1  # if normalize grey
        # scale band (e.g. [0, 255]). If set to None then no normalzation
        # is done
        if norm_on_std == 999:
            self.norm_on_std = False
            self.norm_min_max = [0.5, 99.5]
            self.norm_clip = False
        else:
            self.norm_on_std = True  # # if normalize standardized
            self.norm_min_max = [norm_on_std, 100 - norm_on_std]
            self.norm_clip = norm_clip

        self.take_log = take_log  # True or False

        self.gpu_no = gpu_no

        self.path_out = path_proc
        #if path_suffix is not None:
        #    self.path_out = os.path.join(path_proc, path_suffix)
        #    if not os.path.isdir(self.path_out):
        #        os.mkdir(self.path_out)

        self.save_files_debug = save_files_debug  # saved intermediate files if debug

        self.set_to_nan_lst = set_to_nan_lst
        self.dict_relabel = dict_relabel

        self.n_channels = len(np.array(self.add_bands).ravel()) + len(self.x_bands)

        # pots augmented and non augmented dataset  for idx 0 and idx 5
        self.count_plot = 0
        self.count_plot_hist = 0
        self.width_out = None
        self.height_out = None

    def __getitem__(self, idx):
        '''
        !!! Weights are not considered here
        '''
        # get data

        x_img = self.open_data_img(idx, chunk=None)  # open greyscale image
        # (the GLCMs are added within scale_merge())
        basepath = self.data_files[idx].split('.')[0].split('_raw')[0]
        basename_data = os.path.basename(basepath)

        y_img = self.open_seg_img(idx, chunk=None)
        file_id = self.file_id_lst[idx]
        basename_seg = os.path.basename(
            self.seg_files[idx]).split('.')[0]
        # file_id is e.g. 'BLyakhE_SPOT2018_A01_std4_8bit'

        # get stats for specific file
        # (stats file is read before. Which stats file is choosen is
        #  defined per labelling area and is defined in parameter file
        #  e.g. PARAM06_train_model...)
        self.xarr_stats = self.xarr_stats_dict[file_id]

        if self.debug_plot:
            self.plot_within_loader(x_img, y_img[0, :, :], 'orig', idx)
            self.plot_hist_within_loader(
                    x_img, idx, 'raw_' + self.dl_phase)

        # convert to float due to merge later
        # to make sure that there are no issues when calculate means etc
        # for standardization
        # however actually it is not required for all cases.
        # e.g. is use stats from file but it is safer in this way...
        # however to just calculate the stats could also read the
        # files with mask nan (will then be float) but this isnot
        # possibl ehere since need inisially integer values to
        # calculate the GLCMs
        x_img = geo_utils.convert_img_to_dtype(
            x_img, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True)

        # scale and merging. x_img here is still just grey scale image
        # the GLCMs are read from file
        x_img = self.scale_merge(x_img, basepath, chunk=None,
                                 y_img=y_img)

        # mask merged data according to y (clears AOI edge effects)
        x_img = x_img.where(~(y_img.sel(band=self.y_bands[0]) == 0).data.squeeze())

        if self.save_files_debug:
            # save to file
            #path_img = os.path.dirname(self.data_files[idx])
            file_path = os.path.join(
                self.path_out, basename_data  + '_merge.tif')
            x_img.rio.to_raster(file_path, driver='GTiff')
            self.band_x_out = x_img.band.values.tolist()

            file_path = os.path.join(
                self.path_out, basename_seg + '_merge.tif')
            y_img.rio.to_raster(file_path, driver='GTiff')
            self.band_y_out = y_img.band.values.tolist()

        if self.debug_plot:
            self.plot_within_loader(
                x_img, y_img[0, :, :], 'scaled_' + self.dl_phase, idx)
            self.plot_hist_within_loader(
                x_img, idx, 'scaled_' + self.dl_phase)

        # use only fist class band (no weights in case there are some)
        class_lst = y_img.band.values.tolist()[:1]
        y_img = y_img.sel(band=class_lst)

        # relabel classes where required
        if self.dict_relabel is not None:
            y_img = xarray_utils.relabel_xarray(
                y_img, self.dict_relabel)
            #y_out = relabel_arr(y_out, self.dict_relabel)

        # if required set some classes to nan
        #if self.set_to_nan_lst is not None:
        #    x, y_out = set_to_nan(
        #        x, y_out, self.set_to_nan_lst, fill_val=0)
        if (self.set_to_nan_lst is not None
            and len(self.set_to_nan_lst) > 0):
            y_img = xarray_utils.set_label_to_nan(
                y_img, self.set_to_nan_lst, fill_val=0)

        # convert y_img to float and set zeros to nan
        # this is different to torch (in torch nans are not allowed)
        # but here for ML will remove all nans from the dataframe completely
        # as we only look at single pixels
        y_img = geo_utils.convert_img_to_dtype(
            y_img, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True)

        if '1' not in self.bands_all:
            x_img = x_img.drop_sel(band='1')
        lst_merge = [x_img] + [y_img]

        # merge xarray
        img_merged = xarray.concat(lst_merge, dim='band')
        img_merged.attrs['long_name'] = tuple(img_merged.band.values)

        # create data frame
        gdf_merged, gdf_crs, feat_cols = conversion_utils.img_to_df_proc(
                img_merged, rename_dict=None)
        gdf_merged['aoi_key'] = self.aoi_key_lst[idx]
        gdf_merged['sensor'] = self.sensor_lst[idx]

        class_lst = y_img.band.values.tolist()
        feat_cols = np.setdiff1d(feat_cols, class_lst)

        del x_img
        return  gdf_merged, {file_id: y_img}, feat_cols, class_lst

    def __len__(self):
        return len(self.data_files)


class CustomDataset_EnsemblePrediction():
    def __init__(self, proba_files, labels_files, window_divider, dict_relabel):
        self.proba_files = proba_files
        self.labels_files = labels_files
        self.window_divider = window_divider
        self.dict_relabel = dict_relabel
        return

    def __getitem__(self, idx):

        proba_inp = geo_utils.read_rename_according_long_name(
            self.proba_files[idx], mask_nan=False, chunk='auto')
        labels_files = self.labels_files[idx]

        proba_torch_lst = []
        y_torch_lst = []
        x_coords_lst = []
        y_coords_lst = []
        for i in labels_files:
            labels = geo_utils.read_rename_according_long_name(
                i, mask_nan=False, chunk='auto')[0, :, :] # only keep labels here not weights

            if self.dict_relabel is not None:
                labels = relabel_arr(labels, self.dict_relabel)

            proba = proba_inp.rio.reproject_match(
                labels, Resampling=Resampling.bilinear)

            proba_torch = torch.tensor(proba.values, dtype=torch.float32)
            y_torch = torch.tensor(labels.values, dtype=torch.torch.int64)

            # if padding os not required then output is same as input
            proba_torch, y_torch, x_coords, y_coords = pad_to_extent(
                proba_torch, y_torch, labels.x.values, labels.y.values,
                w_div=self.window_divider)

            proba_torch_lst.append(proba_torch)
            y_torch_lst.append(y_torch)
            x_coords_lst.append(x_coords)
            y_coords_lst.append(y_coords)

        proba_torch_b = torch.stack(proba_torch_lst, dim=0)
        y_torch_b = torch.stack(y_torch_lst, dim=0)
        #label_batch = xarray.concat(labels_lst, dim="batch")
        #proba_batch = xarray.concat(proba_lst, dim="batch")

        return proba_torch_b, y_torch_b, labels_files, x_coords_lst, y_coords_lst

    def __len__(self):
        return len(self.proba_files)


def set_device(GPU_LST_STR):
    """
    Set device usage for torch
    device numbering starts from 0

    here just initialize the first device
    several device IDs are used for torch.nn.DataParallel when setting up model

    if GPUs are not available the CPUs are used
    """
    # GPU usage: create list from cmd line string defining GPUs to use for training
    GPU_lst = [int(x) for x in GPU_LST_STR.split(':')]

    # set parallel usage on different GPUs
    if GPU_lst[0] != 99:
        device = torch.device(
            "cuda:" + str(GPU_lst[0])
            if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    return device, GPU_lst


def get_class_counts(meta, dict_relabel, class_num_lst,
                     set_to_nan_index_inp=None, n_augment=1):
    """
    Derive class counts and class weights from the metadata.

    Parameters:
    ----------
    meta: pandas.DataFrame
        DataFrame containing metadata, including class counts.
    dict_relabel: dict
        Dictionary mapping old class labels to new class labels.
    class_num_lst: list
        List of class numbers to consider.
    set_to_nan_index_inp: list, optional
        List of class numbers to set to NaN in the output. Defaults to None.
    n_augment: int, optional
        Number of augmentations per class. Defaults to 1.

    Returns:
    -----------
    tuple: A tuple containing two elements:
        - count_sum (pandas.DataFrame): DataFrame with class counts and new class labels.
        - class_weights (pandas.Series): Series with class weights.
    """
    if set_to_nan_index_inp is None:
        set_to_nan_index = [0]
    else:
        set_to_nan_index = set_to_nan_index_inp + [0]

    count_col = [x for x in meta.columns
                 if x.find('count-class') > -1]
    count_sum = pd.DataFrame(
        meta.loc[:, count_col].sum(axis=0), columns=['count'])
    count_sum.index.name = 'class_name'
    count_sum.reset_index(inplace=True)

    # adjust class counds if used dict relabeling
    count_sum['class_num_old'] = count_sum['class_name'].transform(
        lambda x: int(x.split('_')[1]))
    count_sum['class_num'] = count_sum['class_num_old'].copy()
    for i_old, i_new in dict_relabel.items():
        count_sum.loc[count_sum['class_num_old']==i_old, 'class_num'] = i_new

    count_sum.set_index('class_num', inplace=True)
    # sum up if merged some classes
    count_sum = count_sum.groupby('class_num').sum()[['count']]
    # add labels (e.g. if all ponding areas were labelled to another
    # class, however this should then also be set to masn nan)
    for i in np.setdiff1d(class_num_lst, count_sum.index):
        count_sum.loc[i, 'count'] = 0
    count_sum = count_sum.loc[class_num_lst, :]

    if set_to_nan_index is not None:
        for i in set_to_nan_index:
            count_sum.loc[i, 'count'] = 0
    n_class_actual = count_sum.query('count > 0').shape[0]

    # derive weghts
    count_total = count_sum['count'].sum()
    class_weights = (
        count_total*n_augment)/((count_sum['count']*n_augment)*n_class_actual)
    class_weights = class_weights.replace(
        [np.inf, -np.inf], 0).sort_index()
    return count_sum, class_weights


def calc_stats_all(img):
    '''
    funct from img_preproc class
    (get_stats)
    '''

    img = img.chunk(dict(x=-1, y=-1))

    if img.dtype.name.find('int') > -1:
        # convert to float due to merge later
        xarr = geo_utils.convert_img_to_dtype(
            xarr, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True)

    stats1 = stats_utils.get_stats_df(
        img, perc_min=0.5, perc_max=99.5)

    inp_min = stats1['min'].to_xarray()
    # shift to avoid inf values due to zeros
    img_shift = img + abs(inp_min) + 0.1
    stats2 = stats_utils.get_stats_df(
        np.log(img_shift), perc_min=0.5, perc_max=99.5,
        prefix='log_')
    stats3 = stats_utils.get_stats_df(
        np.exp(img), perc_min=0.5, perc_max=99.5,
        prefix='exp_')

    return pd.concat([stats1, stats2, stats3], axis=1)


def read_stats_to_xarray(path, stats_file, bands_ordered_lst=None):
    '''
    path = PATH_EXPORT

    for ordering, use bands_ordered_lst e.g. from texture.image[img_key].band.values
    (however normally for xarray operations the order is not important if the labels
    are correct. But ordering might avoids errors when convert to numpy array for
    further calcualtions.)

    extract specific statistics as input e.g. for normalize_xarray_perc_clip()
    with: xarr_stats['perc_99-5'].sel(name=img_key, drop=True)
    with img_key e.g. being 'a0-1-2-3_r05_norm_C01'

    '''
    file_path = os.path.join(path, stats_file)
    file_stats = pd.read_csv(
        file_path, index_col=[0, 1], header=0, delimiter='\t')

    xarr_stats = file_stats.to_xarray()

    if bands_ordered_lst:
        xarr_stats = xarr_stats.sel(band=bands_ordered_lst)

    return xarr_stats


def read_append_stats(
        path, file_name, df_stats, drop_subset):

    path_file = os.path.join(path, file_name)
    if os.path.isfile(path_file):
        stats_file = pd.read_csv(
            path_file, sep='\t', header=0, index_col=[0, 1])

        df_stats = pd.concat([stats_file, df_stats],
                             axis=0, names=['name', 'band'])
        # new stats are replaced in file
        df_stats = df_stats.reset_index().drop_duplicates(
            subset=drop_subset, keep='last').set_index(['name', 'band'])

    df_stats.to_csv(
        path_file, '\t', header=True)
    return


def create_img_stack(file_path, band_req, mask_nan=False):
    '''
    1) selects required bands
    2) transfers xarray to numpy array making sure that dimension is
        [n_bands, z, x]
    '''
    img = geo_utils.read_rename_according_long_name(
        file_path, mask_nan=mask_nan, chunk=None)

    return img.sel(band=band_req).values


def merge_img_to_analyse(
        ref_img, img_merge_lst, resampling_type='bilinear',
        band_prefix_lst=None, ref_img_add=False, nodata_out=np.nan):
    '''

    7) Reproject match validation image grid to analyse
            (cluster images and texture measures)
    8) merge images

    class image needs to have been reprojected and matched to ref_img before

    file_band_lst contains sublist with bands to take for each file in file_inp_lst
    if no sublist but None then take all bands. E.g.
    [['pca_x0', 'pca_x1', 'pca_x2'], None]

    band_prefix_lst is sublist with same lenght as file_inp_lst.
        This can define a prefix to be added to the band name.

    Note: This function is very similar to
    training_data_prep_utils.merge_img_to_analyse() but it does not
    read the image
    '''
    if band_prefix_lst is None:
        band_prefix_lst = [None]*len(img_merge_lst)

    img_analyse_lst = []
    count = 0
    for i in img_merge_lst:
        img_analyse = i
        # reproject
        img_analyse = img_analyse.rio.reproject_match(
            ref_img, Resampling=Resampling[resampling_type],
            nodata_out=np.nan)

        if band_prefix_lst[count] is not None:
            band_prefix = band_prefix_lst[count]
        else:
            band_prefix = str(count)

        renamed_bands = [band_prefix + '_' + str(x) for x in
                         img_analyse.band.values.tolist()]
        img_analyse['band'] = ('band', renamed_bands)
        img_analyse_lst.append(img_analyse)
        count += 1

    # merge images
    if ref_img_add:
        img_analyse_lst = [ref_img] + img_analyse_lst
    img_merged = xarray.concat(img_analyse_lst, dim='band')
    img_merged.attrs['long_name'] = tuple(img_merged.band.values)

    return img_merged


def xy_xarray_to_img_save(x_img, x, y_img, y, path_out,
                          file_name_data, file_name_seg, y_bands=None):
    #prefix = ('_aug' + '{0:02d}'.format(1))
    # ---- save augmented files
    # - save data
    attrs = {
        'AREA_OR_POINT': 'Area', 'orig_type': str(x.dtype),
        'scale_factor': 1.0, 'add_offset': 0.0, '_FillValue': 0,
        'long_name': x_img.attrs['long_name']}
    x_img_aug = geo_utils.create_multiband_xarray_from_np_array(
        x_img, x, x_img.band.values.tolist(), attrs=attrs)
    if path_out is not None:
        out_path = os.path.join(path_out, file_name_data)
        x_img_aug.rio.to_raster(out_path, driver='GTiff')

    # - save classes
    if y_bands is None:
        y_bands = ['class']
    if len(y_bands) == 1:
        y_img_aug = geo_utils.create_multiband_xarray_from_np_array(
            y_img, y[:, :, np.newaxis], y_bands, attrs=attrs)
    else:
        y_img_aug = geo_utils.create_multiband_xarray_from_np_array(
            y_img, y, y_bands, attrs=attrs)

    attrs = {
        'AREA_OR_POINT': 'Area', 'orig_type': str(y.dtype),
        'scale_factor': 1.0, 'add_offset': 0.0, '_FillValue': 0,
        'long_name': y_img.attrs['long_name']}
    if path_out is not None:
        out_path = os.path.join(path_out, file_name_seg)
        y_img_aug.rio.to_raster(out_path, driver='GTiff')

    return x_img_aug, y_img_aug


def relabel_arr(arr_inp, rename_classes):
    arr = arr_inp.copy()
    mask_rename = {}
    for i_key in rename_classes.keys():
        mask_rename[i_key] = arr != i_key

    for i_key, i_val in rename_classes.items():
        arr = np.where(mask_rename[i_key], arr, i_val)

    return arr


def set_to_nan(data_arr_inp, class_arr_inp, set_to_nan_lst,
               fill_val=0):
    '''
    should use 0 here nan does not work in torch!!!
    '''
    data_arr = data_arr_inp.copy()
    class_arr = class_arr_inp.copy()

    # get masks
    mask_keep = {}
    for i_class in set_to_nan_lst:
        mask_keep[i_class] = class_arr != i_class

    for i_class in set_to_nan_lst:
        class_arr = np.where(mask_keep[i_class], class_arr, 0)
        data_arr[:, ~mask_keep[i_class]] = fill_val

    return data_arr, class_arr


def plot_augmentations(dataset, idx=0, n_augmentations=2):
    '''
    change according to
    https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
    '''
    # dataset = copy.deepcopy(dataset)
    # dataset.if_augment = [True]*dataset.__len__()
    idx_orig = idx
    set_size = int(dataset.len_orig_files/(n_augmentations + 1))
    n_bands = len(dataset.x_bands)
    fig, ax = plt.subplots(
        nrows=1 + n_augmentations, ncols=n_bands + 2,
        figsize=(15, 24))

    # orig image
    image, mask, weights = dataset[idx_orig]
    plotting_utils.plot_all_bands_mask(
        fig, ax[0, :], image, mask, dataset.x_bands, 'orig',
        weights=weights)

    for i in range(n_augmentations):
        idx = idx + set_size
        image, mask, weights = dataset[idx]
        plotting_utils.plot_all_bands_mask(
            fig, ax[i + 1, :], image, mask, dataset.x_bands, 'augmented',
            weights=weights)

    plt.tight_layout()
    plt.show()

    return


def pad_to_extent(x_torch, y_torch, x_coords, y_coords, w_div=32):
    '''
    x_coords are from image: x_img.coords['x'].values
    y_coords are from image: x_img.coords['y'].values

    x_arr = x_img.values
    y_arr = y_img.values[0, :, :] (no weights here)
    x_torch = torch.tensor(x_arr, dtype=torch.float32)
    y_torch = torch.tensor(y_out, dtype=torch.torch.int64)
    '''
    # Use padding in case file is not dividable by required size (
    # for training or test 32)

    # check image size
    img_size = x_torch.shape[-2:]
    h_r = img_size[0]%w_div
    w_r = img_size[1]%w_div
    if h_r > 0 or w_r > 0:
        # pad input is hw namy pixels are added to
        # (left, right, top, bottom)
        x_torch = torch.functional.F.pad(
            input=x_torch, pad=(0, w_div - w_r, 0, w_div - h_r),
            mode='constant', value=0)
        y_torch = torch.functional.F.pad(
            input=y_torch, pad=(0, w_div - w_r, 0, w_div - h_r),
            mode='constant', value=0)
        # after the prediction the file additional padding is
        # removed again

        # need also to bad coordinates as input for training
        # from data loader needs alwayas to have same size, otherwise
        # get resizing error
        # (nans will be removed in custom_train run_test)
        x_coords = np.pad(x_coords, (0, w_div - w_r),
                            'constant', constant_values=(np.nan, np.nan))
        y_coords = np.pad(y_coords, (0, w_div - h_r),
                            'constant', constant_values=(np.nan, np.nan))

    return x_torch, y_torch, x_coords, y_coords