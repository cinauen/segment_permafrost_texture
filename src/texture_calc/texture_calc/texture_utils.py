'''
This module contains fucnctions and classes for GLCM matrix calculation
(using glcm-cupy https://github.com/Eve-ning/glcm-cupy)

'''
import sys
import os
import numpy as np
import pandas as pd
import logging
import datetime as dt

from joblib import Parallel, delayed, cpu_count
try:
    from skimage.feature import graycomatrix, graycoprops
except:
    # if use old scikit-image version
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops

# GPU library
import cupy
import glcm_cupy

PATH_SEARCH_SEG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))

# ===== import specific utils =====
sys.path.append(PATH_SEARCH_SEG)

import utils.geo_utils as geo_utils
import utils.image_preproc as image_preproc


class TextureCalc(image_preproc.ImagePreproc):
    """
    Class to calculate texture features from an input image and save the
    result as GeoTIFF files.
    This class utilizes GPU-based processing for efficient texture
    calculation.


    Notes
    -----
    - The calculated texture are saved as numpy arrays in the storage
        dictionary self.texture_np. Its key relate to the direction angle,
        window size and channel. If the inpyut imagery has several channels
        then a separate numpy array item is created for each channel.
        The following functions loop thrpough all self.texture_np items
        as separate numpy array oitems are created per channel:
        - texture_np_add_inp_img() to add input array (e.g. greyscale image
            to geotiff image)
        - texture_to_img_save() to create GeoTiff from numpy arrays and
            save them
        - del_texture_and_img() to clear storage dictionary to free
            up memory
    - The `setup_window_param` and `setup_param_GPU` methods must be
        called before performing texture calculations.
    """

    def __init__(self, EPSG_target, path_export, file_prefix,
                 winW=5, winH=5, step=1,
                 angles=None, symmetric=True,
                 normed=True, img=None,
                 img_inp_key='raw', proc_type='tex',
                 bin_from=256, bin_to=256,
                 padding_const=np.nan, tex_suffix=''):

        # required inputs
        self.EPSG_target = EPSG_target  # Target EPSG coordinate system
        # for the output images (int).
        self.path_export = path_export  # Path to save output.
        self.file_prefix = file_prefix  # The prefix for the output
        # file names.

        # storage dictionaries with keys relating to calculated texture
        self.texture_np = {}  # numpy array with claculated texture features
        self.image = {}  # xarray dict to store final texture calculation
        self.img_np = {}
        self.image_meta = {}  # image metadata dictionary
        self.stats = {}  # texture stats
        self.hist_plot = {}

        self.img_inp_key = img_inp_key  # dict key defining from which
        # image should calculate the texture (can per default be left to 'raw')

        # suffix to be added to self.prefix_add
        # self.prefix_add is used to define keys for self.image and
        # self.trexture_np
        self.tex_suffix = tex_suffix

        # columns for metadata to be saved (for self.imgage_meta and
        # self.img_np_meta)
        self.meta_cols = ['name', 'derived_from', 'proc_type', 'bands',
                          'inp_file', 'date', 'file']
        self.scale_cols = ['from_min_range', 'from_max_range',
                           'to_min_range', 'to_max_range']
        # metadata dataframe
        self.image_meta_df = pd.DataFrame()

        # initialize input image if provided
        if img is not None:
            self.init_img(img_inp_key, img_inp_key, img_inp=img,
                         proc_type=proc_type)


        # ------ glcm calculation parameters ----:
        # (distance between pixel pair is always 1 pixel this can not
        # be changed with glcm_cupy)

        self.angles = angles  # List of moving direction for window to
        # calculate GLCM features. If None default is [0, 45, 90, 135]
        # (is EAST, SOUTH_EAST, SOITH, SOUTH_WEST)

        self.symmetric = symmetric  # bool indicating whether to use
        # symmetric GLCM.
        self.normed = normed  # bool indicating whether to normalize the
        # GLCM features.

        # param for moving window
        self.winW = winW  # int, width of the moving window.
        self.winH = winH  # int, height of the moving window.
        self.step = step  # int, The step size for moving the window across
        # the image.

        self.padding_const = padding_const  # A constant value to use
        # for padding the image.

        # for texture calculation images can be transferred to lower bits
        # (per default use 4 bits). This can provide better results since
        # otherwisemight get too sparse matrix
        self.bin_from = bin_from  # bits of input image
        self.bin_to = bin_to  # bits to use for texture calculation

        #self.image[self.img_key] = img
        #self.img_proc = pd.DataFrame(
        #    columns=['name', 'derived_from', 'proc_type', 'bands', 'file', 'date'])

        #self.img_key = img_key

        #self.img_np = {}
        #self.img_np[self.img_np_key] = img_np
        # texture img out
        #if self.img_np[self.img_np_key] is not None:
        #    self.initialize_texture_array()
        #self.img_np_proc = pd.DataFrame(
        #    columns=['name', 'derived_from', 'proc_type', 'bands', 'file'])

    def setup_window_param(self, angles, winW, winH):
        """
        Setup window parameters (direction and size)

        Parameters:
        ----------
        angles: list
            A list containing the direction angles of the window in degrees.
            e.g. [0, 45, 90, 135]
        winW: int
            The width of the window in pixels.
        winH: int
            The height of the window in pixels.

        Returns:
        None: This method does not return anything.
        It updates the instance variables `winW`, `winH`, and `angles`.
        """

        self.winW = winW
        self.winH = winH

        self.angles = angles

        return

    def setup_param_GPU(self):
        """
        Setup parameters for GPU-based texture calculation using glcm_cupy.
        Parameters:
        - self.tex_measures: list of texture measues to calculate
        - self.radius: window radius derived from window sizes
            e.g. self.winH = 11 self.winW = 11 --> self.radius = 5 (the center pixel is not counted)
        - self.padding: is defined according to radius and step (= distance in pizels between windows)
        - self.angles: angles correctly names as required for glcm_cupy
        - self.prefix_add: prefix used for self.texture_np and self.image keys

        Notes
        -----
        - For this method the instance variables `winH` `winW` and `angle`
          must have been previousely defined e.g. with self.setup_window_param()
        - For this method the following instance variables must have been defied
            `step`, `normed`, and `tex_suffix` to be correclty defined (is done on initialization).
        - this function is e.g. run from within derive_texture_loop_GPU()
        """

        self.tex_measures = ['HOMOGENEITY', 'CONTRAST', 'ASM', 'MEAN',
                             'VAR', 'CORRELATION', 'DISSIMILARITY']
        self.n_tex = len(self.tex_measures)

        radius_x, radius_y = get_radius((self.winH, self.winW))
        self.radius = radius_x

        self.padding = self.radius + self.step

        if self.angles is None:
            # usd for directions
            # is not used for cross GLCM
            self.angles = (
                glcm_cupy.Direction.EAST,
                glcm_cupy.Direction.SOUTH_EAST,
                glcm_cupy.Direction.SOUTH,
                glcm_cupy.Direction.SOUTH_WEST)
        elif (not isinstance(self.angles, list)
              and not isinstance(self.angles, tuple)):
            self.angles = [self.angles]

        dict_convert = {
            0: glcm_cupy.Direction.EAST,
            45: glcm_cupy.Direction.SOUTH_EAST,
            90: glcm_cupy.Direction.SOUTH,
            135: glcm_cupy.Direction.SOUTH_WEST}

        self.angles = [dict_convert[x] if isinstance(x, int) else x
                       for x in self.angles]

        if self.normed:
            pref_add = '_norm'
        else:
            pref_add = ''

        self.set_prefix_add(pref_add)
        return

    def set_prefix_add(self, pref_add):
        angles_str = '-'.join([str(x.value) for x in self.angles])
        self.prefix_add = f'a{angles_str}_r{self.radius:02d}{pref_add}{self.tex_suffix}'
        return

    def derive_texture_loop_GPU(
            self, img_step_size, overlap_scale=2, parallel=True):
        """
        Calculate texture features of a large array by splitting the array
        into smaller sub-array and processing them in parallel on a GPU.
        This approach is necessary to avoid GPU memory overflow issues
        with very large images.

        Parameters
        ----------
        img_step_size: int
            The size of each sub-image in pixels.
        overlap_scale: int or float
            The scale factor by which the overlap between sub-images is
            increased.
        param parallel: bool
            Boolean indicating whether to use parallel processing.

        The texture output is saved as numpy array in self.texture_np dict.
        Dimension is [H, W, texture_features]. Is input image had several
        channels then self.texture_np array is created for each
        with the channel specified in the dict key (XX'_C0X')
        """

        self.setup_param_GPU()

        overlap = self.padding*overlap_scale*2
        self.sub_img_size = img_step_size + overlap

        # -- Split image into subimages --
        img_np_shape = self.img_np[self.img_inp_key].shape
        y_start = np.array(
            range(0, img_np_shape[0], img_step_size))[:, np.newaxis]
        y_end = y_start + self.sub_img_size
        y_loop = np.hstack([y_start, y_end])

        x_start = np.array(
            range(0, img_np_shape[1], img_step_size))[:, np.newaxis]
        x_end = x_start + self.sub_img_size
        x_loop = np.hstack([x_start, x_end])

        # --- Convert image to float and handle zero values
        # for cupy glcm it is best to set 0 no-values to nan and minus
        # all values minus 1 (thus 4bit data has values from 0 to 15)
        if str(self.img_np[self.img_inp_key].dtype)[:5] != 'float':
            img_inp = self.img_np[self.img_inp_key].astype(float).copy()
            img_inp[img_inp == 0] = np.nan
        else:
            img_inp = self.img_np[self.img_inp_key].copy()
        if np.nanmax(img_inp) >= self.bin_from:
            img_inp = img_inp - 1.0

        # --- Create sliding window for sub-images
        sli_win = sliding_win_GPU(img_inp, y_loop, x_loop)

        # --- Calculate texture features on sub-images
        if parallel:
            # parallelization doesn't bring  lot of speedup here
            # would be better if could distribute onto 6 GPUs
            n_jobs = int(cpu_count()/10)
            img_stack = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(self.derive_texture_np_sub_img_GPU)(k) for k in sli_win)
        else:
            img_stack = []
            for i in sli_win:
                img_stack.append(self.derive_texture_np_sub_img_GPU(i))

        # --- Remove padding from sub-images
        edge_clip = self.padding*overlap_scale
        stack_lst = [x[edge_clip:-edge_clip, edge_clip:-edge_clip, :, :]
                     for x in img_stack]
        n_stack = len(stack_lst)

        # --- Merge sub-images
        # merge in x direction
        x_ind = list(range(0, n_stack+1, x_loop.shape[0]))
        texture_np = [np.concatenate(stack_lst[x_start:x_end], axis=1)
                      for x_start, x_end in zip(x_ind[:-1], x_ind[1:])]
        # Concatenate in y direction and clip back to original image
        # shape. Remove added cells for equal subimage size.
        texture_np  = np.concatenate(
            texture_np, axis=0)[:img_np_shape[0], :img_np_shape[1], :, :]

        # --- correct for window center
        # pad and cut such that point corresponds to windows center
        texture_np = np.pad(
            texture_np, pad_width=((edge_clip,), (edge_clip,), (0,), (0,)),
            constant_values=np.nan)[:-edge_clip*2, :-edge_clip*2, :, :]
        # there are some remaining edge effects !!!!
        # those will be clipped in texture_to_img()

        # --- Split into channels and update texture_np
        n_channels = texture_np.shape[-2]
        key_lst = [self.prefix_add + '_C' + '{0:02d}'.format(x + 1)
                   for x in range(n_channels)]
        # use update here such that can keep texture_np for cross calc
        self.texture_np.update({
            key_lst[x]: texture_np[:, :, x, :].copy()
            for x in range(n_channels)})

        return

    def derive_texture_np_sub_img_GPU(
            self, img_inp, max_partition_size=1000, max_threads=512):
        '''
        Derive texture features for the provided sub image.

        Parameters:
        ----------
        img_inp: numpy array
            numpy array for which to calculate texture, is sub array
            derived from sliding_win_GPU().
        max_partition_size: int
            number of windows parsed per GLCM Matrix
        max_threads: int
            number of threads per CUDA block

        Returns:
        -------
        texture_np: numpy array
            the derived texture features got the subarray.

        This method performs the following steps:
        1. Pads the input image to ensure that the texture feature
           computation can handle the image edges correctly.
        2. Computes texture features using the `glcm_cupy.glcm`
           function, which utilizes GPU acceleration for efficient processing.
        3. Adjusts the shape of the resulting texture array to match the
           expected sub-image size if necessary.
        4. Returns the derived texture features as a numpy array.
        '''

        # Pad the input image
        ar_pad = np.pad(
            img_inp,
            pad_width=((self.padding,), (self.padding,), (0,)),
            constant_values=self.padding_const)
        # ar_g shape is [y, x, channels, tex_measures]

        # Compute texture features using GPU
        texture_np = glcm_cupy.glcm(
            ar_pad, bin_from=self.bin_from, bin_to=self.bin_to,
            radius=self.radius, step_size=self.step,
            max_partition_size=max_partition_size,
            max_threads=max_threads, directions=self.angles,
            normalized_features=self.normed)

        # Adjust shape if necessary
        tex_shape = texture_np.shape
        if (tex_shape[0] != self.sub_img_size
            or tex_shape[1] != self.sub_img_size):
            im_corr = np.zeros(
                (self.sub_img_size, self.sub_img_size,
                 tex_shape[-2], tex_shape[-1]))*self.padding_const

            im_corr[:tex_shape[0], :tex_shape[1], :, :] = texture_np.copy()
            texture_np = im_corr.copy()

        return texture_np

    def derive_texture_np_GPU(
            self, max_partition_size=1000, max_threads=512,
            gpu_output=False, mask_nan=True):
        """
        Calculate texture features for a numpy array of small size.
        For large arrays and if there are memory issues one should use
        derive_texture_loop_GPU() which splits the image into subarrays.

        Parameters:
        max_partition_size: int
            Number of windows parsed per GLCM Matrix
        max_threads: int
            Number of threads per CUDA block.
        gpu_output: bool
            If True, the output will be in GPU memory. If False, it will
            be converted to CPU memory. Default is False.
        mask_nan: bool
            If True, NaN values will be masked in the output texture
            features. Default is True.

        Returns:
        -------
        dict
            A dictionary containing the calculated texture features,
            with keys formatted as 'prefix_add_Cxx', where 'xx' is the
            channel number. (!!! several dictionary nu,py items are created
            if input imagery has seceral channels)

        Notes:
        - The function sets-up parameters for GPU processing, converts
            the input image to float if necessary, and pads it to ensure
            proper handling of boundaries.
        - After computing the texture features, if `mask_nan` is True,
            any NaN values in the output are set to NaN. The feature
            are then stored in a dictionary with structured keys.
        - The function returns the dictionary containing the calculated
            texture features.
        """

        # setup texture parameters
        self.setup_param_GPU()

        # --- Convert the image to float if necessary and set 0 values to NaN
        # for cupy glcm it is best to set 0 no-values to nan and minus
        # all values minus 1 (thus 4bit data has values from 0 to 15)
        if str(self.img_np[self.img_inp_key].dtype)[:5] != 'float':
            img_inp = self.img_np[self.img_inp_key].astype(float).copy()
            img_mask = img_inp == 0
            img_inp[img_mask] = np.nan
        else:
            img_inp = self.img_np[self.img_inp_key].copy()
            img_mask = np.isnan(img_inp)

        # Adjust the image values if the maximum value is greater than
        # or equal to bin_from
        if np.nanmax(img_inp) >= self.bin_from:
            img_inp = img_inp - 1.0

        # --- Pad the image to handle boundary conditions
        ar_pad = np.pad(
            img_inp, pad_width=((self.padding,), (self.padding,), (0,)),
            constant_values=self.padding_const)

        # --- Convert the padded image to a CuPy array if gpu_output is True
        if gpu_output:
            ar_pad = cupy.asarray(ar_pad)

        # --- Compute the texture features using GPU-based GLCM
        texture_np = glcm_cupy.glcm(
            ar_pad, bin_from=self.bin_from, bin_to=self.bin_to,
            radius=self.radius, step_size=self.step,
            max_partition_size=max_partition_size,
            max_threads=max_threads, directions=self.angles,
            normalized_features=self.normed)
        # ar_g shape is [y, x, channels, tex_measures]
        # channel here is 1 due to gray scale

        # If mask_nan is True, set NaN values to NaN in the output
        if mask_nan:
            texture_np[img_mask, :] = np.nan

        # Determine the number of channels and create keys for the
        # output dictionary
        n_channels = texture_np.shape[-2]
        key_lst = [self.prefix_add + '_C' + '{0:02d}'.format(x + 1)
                   for x in range(n_channels)]

        # Update the texture_np dictionary with the calculated texture
        # features such that that can keep texture_np for cross calc
        self.texture_np.update({
            key_lst[x]: texture_np[:, :, x, :].copy()
            for x in range(n_channels)})

        return self.texture_np

    def texture_np_add_inp_img(self, inp_key='raw'):
        """
        Concatenate a numpy array of the self.img_np dictionary to the
        existing texture_np array.
        This can be used to add the greyscale imagery as an additional
        channel. The additional channel is added on top along the third
        axis.

        The `tex_measures` list is updated to to include `inp_key` at the
        beginning of the list.

        Parameters:
        -------
        inp_key: str, optional
            The key corresponding to the input image in the self.img_np
            dictionary. Default is 'raw'.
        """

        for i_key, i_val in  self.texture_np.items():
            self.texture_np[i_key] = np.concatenate(
                [self.img_np[inp_key], i_val], axis=2)

        self.tex_measures = [inp_key] + self.tex_measures
        return

    def texture_to_img_save(
            self, AOI_poly=None, save=True):
        """
        Converts texture numpy arrays to xarray images (GeoTIFF).
        !!! this function is not made to be used in parallel
        as loops trough all texture items !!!

        Parameters
        ----------
        AOI_poly : shapely.geometry.Polygon, optional
            Area of Interest polygon to clip the calculated textures to
            avoid edge effects.
        file_name_prefix : str, optional
            Prefix for the saved image file names.
        save : bool, optional
            Whether to save the image files to disk.

        Returns
        -------
        None
        """

        for i_key, i_val in self.texture_np.items():
            name = i_key
            # Adjust image attributes. This is impotant for plotting later
            attrs = {
                'AREA_OR_POINT': 'Area', 'orig_type': str(i_val.dtype),
                'scale_factor': 1.0, 'add_offset': 0.0, '_FillValue': np.nan}
            self.image[name] = geo_utils.create_multiband_xarray_from_np_array(
                self.image[self.img_inp_key], i_val, self.tex_measures,
                attrs=attrs, nodata=np.nan)

            if AOI_poly is not None:
                # clip calculated texture to avoid edge effects
                AOI_scaled = geo_utils.scale_AOI(
                    AOI_poly, self.image[name].rio.resolution()[0],
                    self.padding*2, self.EPSG_target)

                self.image[name] = geo_utils.clip_to_aoi_gdf(
                    self.image[name], AOI_scaled)

            filename = '-'
            if save:
                filename = (self.file_prefix + '_' + i_key)
                geo_utils.save_to_geotiff_options(
                    self.image[name], self.path_export, filename,
                    single=False)
                filename += '.tif'

            # update metadata
            bands_txt = ':'.join(self.tex_measures)
            self.update_meta(
                name, [name, 'calc tex', 'tex', bands_txt, '-',
                dt.datetime.now().strftime('%Y-%m-%d_%H%M'), filename],
                self.meta_cols)

        return

    def del_texture_and_img(self):
        for i_key in self.texture_np.keys():
            del self.image[i_key]

        del self.texture_np
        self.texture_np = {}



def sliding_win_GPU(img_np, loop_y, loop_x):
    for y in loop_y:
        for x in loop_x:
            yield img_np[y[0]:y[1], x[0]:x[1]]


def get_radius(windowSize):
    if (windowSize[0] % 2) == 0 or (windowSize[1] % 2) == 0:
        radius_y = int(windowSize[0]/2)
        radius_x = int(windowSize[1]/2)
    else:
        radius_y = int((windowSize[0] - 1)/2)
        radius_x = int((windowSize[1] - 1)/2)

    return radius_x, radius_y


def read_calc_cross_image_stats(img_meta_dict, image_key_prefix, PARAM):

    img_dict = image_preproc.read_file_from_proc_file(
        PARAM['PATH_EXPORT'], img_meta_dict, chunk='auto')

    img_out, img_meta_df = calc_cross_image_stats(
        img_dict, image_key_prefix, PARAM)

    return img_out, img_meta_df


def calc_cross_image_stats(img_dict, image_key_prefix, PARAM):
    """
    Calculate cross-image statistics from a dictionary of images.

    INPUT
    img_dict : dict
        A dictionary containing all images from which to calculate cross-image statistics.
        Example: {x: texture.image[x] for x in i_var}

    image_key_prefix : str
        A prefix to append to the calculated image keys to avoid
            conflicts with existing keys.

    PARAM : dict
        A dictionary containing parameters necessary for the calculation.
        Required keys:
        - 'EPSG_TARGET' : int e.g. 32654
            Target EPSG code for the output images.
        - 'PATH_EXPORT' : str
            Path to export the output images.
        - 'FILE_PREFIX' : str
            Prefix for the output file names.
        - 'cross_calc' : list e.g. ['std', 'var', 'mean']
            List of cross-image statistics to calculate.

    OUTPUT
    calc_img.image : dict
        A dictionary containing the calculated cross-image statistics.

    calc_img.image_meta : dict
        A dictionary containing metadata for the calculated cross-image
            statistics.

    DESCRIPTION
    This function takes a dictionary of images, a key prefix, and a parameter dictionary,
    and calculates specified cross-image statistics. The results are stored in a new key
    in the `img_dict` dictionary, and the metadata is updated accordingly.

    The function uses the `image_preproc.ImagePreproc` class to handle image preprocessing,
    the `geo_utils.concat_arrays_and_calc` function to concatenate and calculate the statistics,
    and updates the image metadata using the `image_preproc.update_meta` method.

    The calculated images and their metadata are then saved using the
    `image_preproc.image_save_several` method.

    """
    # Initialize the image processing object
    calc_img = image_preproc.ImagePreproc(PARAM['EPSG_TARGET'],
                                          PARAM['PATH_EXPORT'],
                                          PARAM['FILE_PREFIX'])

    #for i_calc in PARAM['cross_calc']:
    i_key = image_key_prefix + '_calc'# + i_calc
    #calc_img.image[i_key], bands_calc = geo_utils.merge_arrays_and_calc(
    #    img_dict, calc=i_calc)

    # Concatenate the input images and calculate the specified cross-image
    # (pixel wise) statistics
    calc_out, bands_calc = geo_utils.concat_arrays_and_calc(
        img_dict, calc=PARAM['cross_calc'], logging_inp=logging,
        dict_prefix=i_key)
    calc_img.image.update(calc_out)

    # Update metadata for each calculated statistic
    bands_txt = ':'.join(bands_calc)
    for x in PARAM['cross_calc']:
        img_key = i_key + '_' + x
        calc_img.update_meta(
            img_key,
            [img_key, 'calc ' + x + ' ' + image_key_prefix,
             'calc', bands_txt, '-',
             dt.datetime.now().strftime('%Y-%m-%d_%H%M')],
            calc_img.meta_cols[:-1])

    # get stats
    calc_img.get_stats_set(
        np.setdiff1d(list(calc_img.image.keys()), ['raw']),
        aoi_dict=PARAM['AOI_stats_calc'],
        path_inp=PARAM['PATH_INP'])

    # Save the calculated images and update the metadata dataframe
    calc_img.image_save_several(
        'proc_type', 'calc', del_saved_array=False)

    del calc_img.image

    return calc_img.image_meta_df, calc_img.stats


