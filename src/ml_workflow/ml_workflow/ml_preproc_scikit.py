"""
Module that can be used as an extension to allow scikit learn
pre-processing

The class PreprocScikitML() would need to be inherited by
ml_classification_utils.ClassifierML()

"""
import os
import sys
import numpy as np
import xarray
import torch

if torch.cuda.is_available():
    from cuml.decomposition import PCA
    from cuml.preprocessing import StandardScaler, Normalizer, MinMaxScaler
else:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

# ----------------- import custom utils -----------------
import utils.plotting_utils as plotting_utils
import utils.conversion_utils as conversion_utils
import utils.geo_utils as geo_utils
import utils.xarray_utils as xarray_utils


class PreprocScikitML():  # image_preproc.ImagePreproc
    '''
    Preprocessing class for preprocessing input features with scikit-learn
    This module is inherited by ClassifierML
    '''
    def __init__(self, X_inp, Y_inp, X_inp_test, Y_inp_test,
                 features_col, path_export, file_prefix,
                 preproc_type=None, n_components=None, train_count=1):
        self.file_prefix = file_prefix  # to save hdf file

        self.X = X_inp
        self.Y = Y_inp
        self.X_test = X_inp_test
        self.Y_test = Y_inp_test

        self.features_col = features_col  # list of column names used for
        self.features_col_orig = features_col.copy()
        self.col_prefix = ''  # prefix used for scikit preprocessing
        # labels in PreprocScikitML

        self.path_export = path_export

        self.preproc_type = preproc_type
        self.initialize_preproc(preproc_type, n_components)

    def initialize_preproc(
            self, preproc_type, n_components):
        '''
        '''
        self.set_preproc_type(preproc_type)
        self.create_pipeline(n_components)

        return

    def set_preproc_type(self, preproc_type):
        if preproc_type is None:
            self.preproc_type = []
        else:
            self.preproc_type = preproc_type
        return

    def create_pipeline(self, n_components=None):
        '''
        create preprocessing pipeline
        '''
        # preproc_type options:
        #  for nothing put NO
        #  1) preprocessing: scale
        #  2) dimensionality reduction: PCA
        #  e.g. NO: does nothing, scalePCA does PCA and scale
        #  ------ create pipeline
        # preprocessing
        pipeline = []
        for i in self.preproc_type:
            if i.find('standardscaler') > -1:
                pipeline.append(StandardScaler())
            if i.find('normalizer') > -1:
                pipeline.append(Normalizer())
            if i.find('minmaxscaler') > -1:
                pipeline.append(MinMaxScaler())
            # dimensionality reduction with pca
            if n_components is None:
                n_components = np.min([self.X.shape[1], 2])
            n_components = np.min([self.X.shape[1], n_components])
            if i.find('pca') > -1:
                pipeline.append(PCA(n_components=n_components))
        self.pipeline = make_pipeline(*pipeline)
        return

    def take_log_exp(self, log_dict, phase='train_validate_test',
                     log_shift=0.01):
        '''
        log_dict has as keys either log or exp ands as items
            the parameter name from which to calculate log or exp
            e.g. HOMOGENEITY

        Constant shift (log_shift) is used to avid negative or zero values.
        However, with the GLCMs all values should be between 0 and 1
        anyway.
        '''
        for i_log_key, i_col in log_dict.items():
            # extract columns
            # here use orig names since if only update test then
            # self.features_col would have been adjusted
            col_convert = [x for x in self.features_col_orig
                           if x.split('_')[-1] in i_col]

            col_rename = {
                x: i_log_key  + '_'  +  x for x in col_convert}

            # use log_shift to avoid negative and zero values
            if i_log_key == 'log':
                if 'train' in phase:
                    self.X[col_convert] = np.log(
                        self.X.loc[:, col_convert] + log_shift)
                if 'validate' in phase or 'test' in phase:
                    self.X_test[col_convert] = np.log(
                        self.X_test.loc[:, col_convert] + log_shift)
            elif i_log_key == 'exp':
                if 'train' in phase:
                    self.X[col_convert] = np.exp(
                        self.X.loc[:, col_convert] + log_shift)
                if 'validate' in phase or 'test' in phase:
                    self.X_test[col_convert] = np.exp(
                        self.X_test.loc[:, col_convert] + log_shift)

            if 'train' in phase:
                self.X.rename(columns=col_rename, inplace=True)
            if 'validate' in phase or 'test' in phase:
                self.X_test.rename(columns=col_rename, inplace=True)

            self.features_col = self.X.columns.tolist()
            # check that colums of self.X_test are correct
            if 'validate' in phase:
                if not np.all(self.X.columns.tolist() == self.X_test.columns.tolist()):
                    sys.exit(
                        '!!! inconsistent column naming with X and X_test !!!!')
        return

    def preprocess_data(self, log_dict=None, keep_all=True):
        '''
        preprocesses train and test data (self.X and self.X_test are
        overwritten)
        !!! pipeline is updated through fitting parameters of self.X
        '''
        proc_steps = self.preproc_type.split('_')

        if log_dict is not None:
            plt_add = 2
        else:
            plt_add = 1
        plt_count = len(proc_steps) + plt_add
        if '1' in self.features_col_orig:
            plt_count *= 2

        # ----- plot distributions
        # initialize plots
        fig, axs = plt.subplots(plt_count, figsize=[15, 11])
        fig_t, axs_t = plt.subplots(plt_count, figsize=[15, 11])
        plt_ct = 0
        plt_ct_t = 0

        # -- plot unprocessed
        # Note greyscale band ['1'] is plotted onto a separate figure
        # axis as it might have a different range in the orignnal data
        # [1, 255] not [0, 1]
        plt_ct = plot_distribution(
            self.X, axs, plt_ct, col_excl_lst=['1'])
        plt_ct_t = plot_distribution(
            self.X_test, axs_t, plt_ct_t, col_excl_lst=['1'])

        if log_dict is not None:
            self.take_log_exp(log_dict)
            plt_ct = plot_distribution(
                self.X, axs, plt_ct, col_excl_lst=['1'])

            plt_ct_t = plot_distribution(
                self.X_test, axs_t, plt_ct_t, col_excl_lst=['1'])

        if ['1'] in self.features_col_orig:
            col_num_ind = [self.features_col_orig.index('1')]
        else:
            col_num_ind = []

        # ---- preprocess data by looping through preprocessing steps
        for e, i in enumerate(proc_steps):
            proc_step = self.pipeline.named_steps[i]

            # -- preprocess training data
            # !!! the following is just a copy. Thus, parameters of
            # self.pipeline will be accordingly adjusted
            # e.g. check self.pipeline.named_steps[i].mean_ will be same
            # as proc_step.mean_
            self.X = proc_step.fit_transform(self.X)
            # plot distribution
            plt_ct = plot_distribution(self.X, axs, plt_ct, col_num_ind)

            # -- preprocess test data
            # !!! since use here transform only then e.g. mean and std is taken from self.X
            # thus transformation is done based on training data
            # if would use proc_step.fit_transform(self.X_test) then
            # transormation would be done based on test data
            # check this with
            # t_new = proc_step.fit_transform(self.X_test)
            # np.all((self.X_test - np.mean(self.X_test, axis=0))/np.std(self.X_test, axis=0) == t_new)
            # output would be true
            self.X_test = proc_step.transform(self.X_test)
            # plot distribution
            plt_ct_t = plot_distribution(
                self.X_test, axs_t, plt_ct_t, col_num_ind)

            if i == 'pca':
                # make plot with check importance of each component
                fig_pca, ax_pca = plt.subplots(1, figsize=[11.7, 8.3])
                ax_pca.plot(
                    np.cumsum(proc_step.explained_variance_ratio_).get())
                ax_pca.xlabel('Number of components')
                ax_pca.ylabel('Cumulative explained variance')
                file_name = (self.file_prefix
                             + '_PCA_explained_variance.pdf')
                fig_pca.savefig(
                    os.path.join(self.path_export, file_name),
                    format='pdf')

        # update column naming
        features_col_new, features_text_new = self.get_col_naming_after_preproc(
            proc_steps, keep_all=keep_all)

        # save distriution plots
        fig.tight_layout()
        fig_t.tight_layout()
        file_path = os.path.join(
            self.path_export, self.file_prefix + '_data distribution')
        fig.savefig(file_path + '_train.pdf', format='pdf')
        fig_t.savefig(file_path + '_validate.pdf', format='pdf')
        plt.close('all')

        self.features_col = features_col_new

        # rename self.X and self.X_test columns
        col_rename = {
            y: features_col_new[x] for x, y in enumerate(self.X.columns)}
        self.X.rename(columns=col_rename, inplace=True)
        self.X_test.rename(columns=col_rename, inplace=True)

        if not np.all(self.X.columns.tolist() == self.X_test.columns.tolist()):
            sys.exit(
                '!!! inconsistent column naming with X and X_test !!!!')

        return features_col_new, features_text_new

    def get_col_naming_after_preproc(self, proc_steps, keep_all=True):
        """
        Adjust feature column names according to preprocessing steps
        """
        features_col_new = self.features_col
        if 'pca' in proc_steps:
            features_col_new = ['pca_x' + str(x)
                                     for x in range(self.X.shape[1])]
            features_text_new = '-'.join(features_col_new)
        else:
            if keep_all and 'standardscaler' in proc_steps:
                self.col_prefix += 'standardscaler_'
                features_col_new = [self.col_prefix + str(x)
                                        for x in self.features_col]
                features_text_new = '-'.join(features_col_new)

            if keep_all and 'normalizer' in proc_steps:
                self.col_prefix += 'normalizer_'
                features_col_new = [self.col_prefix + str(x)
                                         for x in self.features_col]
                features_text_new = '-'.join(features_col_new)

            if keep_all and 'minmaxscaler' in proc_steps:
                self.col_prefix += 'minmaxscaler_'
                features_col_new = [self.col_prefix + str(x)
                                         for x in self.features_col]
                features_text_new = '-'.join(features_col_new)
        return features_col_new, features_text_new

    def update_preprocess_test_data(
            self, x_test_new, y_test_new=None,
            log_dict=None, phase_suffix='test', keep_all=True):
        '''
        Preprocess and update test data ONLY (i.e. self.X_test)
        This can be usd if e.g. want to check additional test data

        !!! this dunction overwrites self.X_test and self.Y_test !!!
        '''
        self.X_test = x_test_new.copy()
        # for inference self.Y_test should be None
        self.Y_test = y_test_new.copy()

        proc_steps = self.preproc_type.split('_')

        if log_dict is not None:
            plt_add = 2
        else:
            plt_add = 1
        plt_count = len(proc_steps) + plt_add
        if '1' in self.features_col_orig:
            plt_count *= 2

        # - initialize distribution plots
        fig_t, axs_t = plt.subplots(plt_count, figsize=[15, 11])
        plt_ct_t = 0

        # - plot unprocessed
        plt_ct_t = plot_distribution(self.X_test, axs_t, plt_ct_t)

        # -- take log or exp of columns defined in log_dict
        if log_dict is not None:
            self.take_log_exp(log_dict, phase='test')
            plt_ct_t = plot_distribution(self.X_test, axs_t, plt_ct_t)

        for e, i in enumerate(proc_steps):
            proc_step = self.pipeline.named_steps[i]

            # !!! since use here transform only then e.g. mean and std is
            # taken from self.X. Thus, transformation is done based on
            # training data. If would use proc_step.fit_transform(self.X_test)
            # then transformation would be done based on test data
            # check this with:
            # t_new = proc_step.fit_transform(self.X_test)
            # np.all((self.X_test - np.mean(self.X_test, axis=0))/np.std(self.X_test, axis=0) == t_new)
            # --> output would be true
            self.X_test = proc_step.transform(self.X_test)
            plt_ct_t = plot_distribution(self.X_test, axs_t, plt_ct_t)

        # save distribution plot
        fig_t.tight_layout()
        file_path = os.path.join(
            self.path_export, self.file_prefix + '_data distribution')
        fig_t.savefig(f'{file_path}_{phase_suffix}.pdf', format='pdf')
        plt.close('all')

        # rename self.X and self.X_test columns
        self.features_col = self.X.columns.tolist()
        col_rename = {
            y: self.features_col[y] for x, y in enumerate(self.X_test.columns)}
        self.X_test.rename(columns=col_rename, inplace=True)

        # check if colum naming is correct
        if not np.all(self.X.columns.tolist() == self.X_test.columns.tolist()):
            sys.exit(
                '!!! inconsistent column naming with X and X_test !!!!')
        return


# ---------------- subfunctions ----------------------
def read_merge_extract(data_file, glcm_file_lst, class_file, aoi_key,
                       sensor_id, band_lst, class_lst=None,
                       dict_relabel=None, mask_lst=None):
    '''
    Function to read prepare the input for the ML classification (feature
    inputs and labels):
    1) read GeoTiffs (data and labels) to xarray.DataArray
    2) relabel or ignore classes if required
    3) convert images to correct dtype
    4) read additional input features (GLCMs)
    5) merge all DataArrays into a single xarray.DataArray
    6) Convert merged DataArray to DataFrame which can be used as input
       for scikit-learn or cuml

    Note:
    - Zeros in data_img (grey scale image) are epxected to correspond
        to nan !!!!
    - class image needs to have been reprojected and matched to ref_img before
    '''
    # 1) -- read image and and ground truth labels
    data_img = geo_utils.read_rename_according_long_name(
            data_file, mask_nan=False, chunk='auto')
    class_img = geo_utils.read_rename_according_long_name(
            class_file, mask_nan=False, chunk='auto')

    # check for bands to be included
    data_bands_incl = np.intersect1d(
        data_img.band.values.tolist(), band_lst)

    # use only fist class band (no weights in case there are some)
    if class_lst is None:
        class_lst = class_img.band.values.tolist()[:1]
    class_img = class_img.sel(band=class_lst)

    # 2) -- relabel is required
    # relabel classes where required
    if dict_relabel is not None:
        class_img = xarray_utils.relabel_xarray(
            class_img, dict_relabel)

    # if required set some classes to 0
    if mask_lst is not None and len(mask_lst) > 0:
        class_img = xarray_utils.set_label_to_nan(
            class_img, mask_lst, fill_val=0)

    # 3) -- convert data_img and class_img to float and set zeros to nan.
    # This is different to torch (in torch nans are not allowed)
    # But here for ML will remove all nans from the dataframe completely
    # as we only look at single pixels (pixel based classification).
    class_img = geo_utils.convert_img_to_dtype(
        class_img, dtype_out='float64', nodata_out=np.nan,
        replace_zero_with_nan=True)
    if len(data_bands_incl) > 0:
        data_img = geo_utils.convert_img_to_dtype(
            data_img, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True).sel(band=data_bands_incl)

    # 4) -- read additional input features (GLCMs)
    # select and read glcm bands to be used
    glcm_img_lst = []
    glcm_band_lst = np.setdiff1d(band_lst, data_img.band.values.tolist())
    for e, i in enumerate(glcm_file_lst):
        glcm_img = geo_utils.read_rename_according_long_name(
            i, mask_nan=False, chunk='auto').sel(band=glcm_band_lst)

        band_prefix = str(e)
        renamed_bands = [f'{band_prefix}_{x}' for x in
                         glcm_img.band.values.tolist()]
        glcm_img['band'] = ('band', renamed_bands)
        glcm_img_lst.append(glcm_img)

    # 5) -- merge all DataArrays into a single xarray.DataArray
    data_bands_incl = np.intersect1d(
        data_img.band.values.tolist(), band_lst)
    # add images to be meged to list
    if len(data_bands_incl) > 0:
        lst_merge = [data_img] + glcm_img_lst + [class_img]
    else:
        lst_merge = glcm_img_lst + [class_img]

    # merge all into one xarray
    img_merged = xarray.concat(lst_merge, dim='band')
    img_merged.attrs['long_name'] = tuple(img_merged.band.values)

    # 6) -- Convert merged DataArray to DataFrame
    # create DataFrame
    gdf_merged, gdf_crs, feat_cols = conversion_utils.img_to_df_proc(
            img_merged, rename_dict=None, use_GPU=False)
    gdf_merged['aoi_key'] = aoi_key
    gdf_merged['sensor'] = sensor_id

    # get column names of input features
    feat_cols = np.setdiff1d(feat_cols, class_lst)

    # keep class_img to retrieve coordinates later
    return gdf_merged, {aoi_key: class_img}, feat_cols, class_lst


def plot_distribution(
        X_df, axs, plt_count, col_excl_lst=None):
    """
    Adds a histogram plots of the input features to the image axis (axs)
    """
    # exclude input features from being plotting onto the first axis
    # (e.g. if have very different data range)
    if col_excl_lst is None:
        col_excl_lst = []
    col_incl = np.setdiff1d(X_df.columns, col_excl_lst)

    # create color scale
    col_lst = plotting_utils.cmap_to_hex(
        'Dark2', num_bin=None)[:7]
    mult = int(np.ceil(len(col_incl)/len(col_lst)))
    cmap_cont = plotting_utils.convert_col_lst_to_cmap(
        col_lst*mult)

    # plot histogram onto axis
    X_df[col_incl].to_pandas().plot.hist(
        bins=100, alpha=0.5, ax=axs[plt_count], cmap=cmap_cont)
    axs[plt_count].legend(
        loc='upper left', bbox_to_anchor=(1, 1), ncol=3)

    plt_count += 1

    # plot excluded features on an additional subplot
    cols_excl = np.intersect1d(col_excl_lst, X_df.columns)
    if len(cols_excl) > 0:
        X_df[col_excl_lst].to_pandas().plot.hist(
            bins=100, alpha=0.5, ax=axs[plt_count], cmap=cmap_cont)
        axs[plt_count].legend(
            loc='upper left', bbox_to_anchor=(1, 1), ncol=3)

        plt_count += 1

    return plt_count