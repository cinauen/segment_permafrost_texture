

"""
This module contains a class and some subfunctions to pre-process imagery
e.g. for texture calculation afterwards

"""

import os
import operator
import functools
import numpy as np
import pandas as pd

import datetime as dt
import holoviews as hv
hv.extension('matplotlib')

import utils.geo_utils as geo_utils
import utils.stats_utils as stats_utils
import utils.plotting_utils as plotting_utils


class ImagePreproc():
    '''
    Class to preprocess imagery e.g. for texture calculation afterwards
    (ImagePreproc is inherited by texture_utils.TextureCalc())

    Processing on image:
        - translate
        - resample
        - clip

    Processing on nparray
        - change bit (GPU option)
        - RGBtoGrey (GPU option)
        - plotting arrays

    Processing steps are summarized in dict self.image_meta.
    Dict key is img_key. At the end the metadata are they are
    concatenated to a DrataFrame and saved.

    '''
    def __init__(self, EPSG_target, path_export, file_prefix):

        # columns for metadata to be saved (for self.imgage_meta and
        # self.img_np_meta)
        self.meta_cols = ['name', 'derived_from', 'proc_type', 'bands',
                          'inp_file', 'date', 'file']
        self.scale_cols = ['from_min_range', 'from_max_range',
                           'to_min_range', 'to_max_range']
        # metadata dataframe
        self.image_meta_df = pd.DataFrame()

        # storage dictionalries
        self.image = {}
        self.image_meta = {}
        self.img_np = {}
        self.stats = {}
        self.hist_plot = {}

        self.EPSG_target = EPSG_target
        self.path_export = path_export
        self.file_prefix = file_prefix

    def init_img(self, img_key, img_key_inp, img_inp=None,
                 img_np_inp=None, proc_type='img', bands='-', file='-',
                 meta_update=True):
        """update storage dictionary"""

        # initialize new image item
        if img_inp is not None:
            self.image[img_key] = img_inp.copy()
            bands_txt = ':'.join([str(x) for x in
                              self.image[img_key].band.values.tolist()])
        else:
            bands_txt = '-'

        # initialize new numpy array item
        if img_np_inp is not None:
            self.img_np[img_key] = img_np_inp.copy()

        # update metadata
        if meta_update:
            self.update_meta(img_key,
                    [img_key, img_key_inp, proc_type, bands_txt, file,
                    dt.datetime.now().strftime('%Y-%m-%d_%H%M')],
                    self.meta_cols[:-1])

    def img_to_nparray(self, img_key, band=None):
        """
        convert image to numpy array

        img_np array will have name of img_key
        this always creates new array
        """

        # create numpy array
        self.img_np[img_key], bands_sel = geo_utils.create_np_array_from_xarray(
            self.image[img_key], band)

        return

    def nparray_to_img(self, img_np_key, img_key_ref='raw',
                       bands=None, proc_type='img', attrs=None,
                       nodata_out=None):
        '''
        convert nparray to image
        !!! this always creates new image and img_key
        '''
        if bands is None:
            bands = list(range(1, self.img_np[img_np_key].shape[-1] + 1))

        if not isinstance(bands, list):
            bands = [bands]

        self.image[img_np_key] = geo_utils.create_multiband_xarray_from_np_array(
            self.image[img_key_ref], self.img_np[img_np_key],
            bands, attrs=attrs, nodata=nodata_out)

        bands_txt = ':'.join([str(x) for x in bands])

        # update metadata
        self.update_meta(
            img_np_key,
            [img_np_key, 'create from nparray ' + img_np_key,
             proc_type, bands_txt, '-',
             dt.datetime.now().strftime('%Y-%m-%d_%H%M')],
            self.meta_cols[:-1])

        return bands

    def preproc_img(self, PATH, FILE, img_key='raw',
                    AOI_coords=None, RESOLUTION_target=0, mask_nan=False,
                    resampling_type='nearest', chunk='auto'):
        """
        Preprocesses an image for texture calculation by performing the
        following steps:

        1. Read the image from the specified path and file.
        1. Check for missing nodata values (if missing assigns
           either 0 or nan).!!! to change nodata values use
           convert_bit_img_np later
        2. Transform the image resolution and resamples it to the
           target resolution.
        3. Clip the image to the specified Area of Interest (AOI)
           coordinates.
        4. Add metadata about the preprocessing steps to the processing
           dictionary.

        Parameters:
        ----------
        PATH: str
            The directory path where the image file is located.
        FILE: str
            The name of the image file.
        AOI_coords: list of tuples, optional
            The coordinates defining the Area of Interest in the same
            coordinate reference system as EPSG_target.
        RESOLUTION_target: float, optional
            The target resolution for the image in the target coordinate
            system. Default is 0, which means no change.
        mask_nan: bool, optional
            Whether to mask NaN values in the image. Default is False.
        nodata_inp: float, int or None
            if want to specify the nodata value from the imagery.
            If None it is derived with img.rio.nodata and if this is None
            it is set to 0 or nan depending on the data type
        resampling_type: str, optional
            The type of resampling to use ('nearest', 'bilinear', 'bicubic').
            Default is 'nearest'.
        chunk: str, dict or None
            for verly large images can use chunking with dask
            can be set specifically with {'x': 2000, 'y': 2000}

        Returns:
        ----------
        - None

        Notes:
        - AOI_coords must have the same coordinate reference system (CRS)
          as EPSG_target.
        - The processed image is stored in the `image` attribute with
          the key `img_key`.
        - Metadata about the preprocessing steps, including the image
          bands, file name, and timestamp, is added to the `img_proc`
          attribute.
        """
        # read image
        self.read_single_image(PATH, FILE, mask_nan=mask_nan,
                               chunk=chunk)

        # check for missing nodata value
        # (to change the nan values, the function convert_bit_img_np()
        # is used)
        geo_utils.check_for_missing_fill_val(self.image[img_key])

        # transform crs and resample to specified resolution
        self.image[img_key] = geo_utils.transform_resample(
            self.image[img_key], EPSG_TARGET=self.EPSG_target,
            RESOLUTION_TARGET=RESOLUTION_target,
            resampling_type=resampling_type,
            )

        # clip image
        if AOI_coords is not None:
            self.image[img_key] = geo_utils.clip_to_aoi(
                self.image[img_key], AOI_coords)

        # update metadata
        bands_txt = ':'.join([str(x) for x in
                              self.image[img_key].band.values.tolist()])
        self.update_meta(
            img_key,
            [img_key, 'inp', 'img', bands_txt,
             FILE, dt.datetime.now().strftime('%Y-%m-%d_%H%M')],
            self.meta_cols[:-1])

        return img_key

    def convert_bit_img(
            self, nodata_inp, save_img=True, del_img=False,
            plot_img=True, overlay_plot=True, img_key_ref=None,
            img_np_key_inp='raw', img_np_key=None, nodata_out=0,
            to_bit_num=2**8, how='scale', perc=None, min_max=None,
            std_fact=None, gamma=None, log_adj=None):
        """
        Converts an input image to an image to specified bits and
        scaling, and saves it if required.
        The input and output arrays are stored in dictionaries.
        The conversion is done on the numpy array (self.img_np[img_np_key]).
        The array is then converted to an xarray by using the coordinates
        and attributes from self.image[img_key_ref].

        Parameters
        ----------
        nodata_inp : int
            The no-data value of the input image.
        save_img : bool
            Whether to save the converted image.
        del_img : bool
            Whether to delete the original image after conversion.
        plot_img : bool
            Whether to include the converted image in the histogram plot.
        overlay_plot : bool
            Whether to overlay the histogram on the plot.
        img_key_ref : str, optional
            Reference image from which coordinates are used when
            transferring the numpy array back to an xarray.
            (default None --> uses img_np_key_inp)
        img_np_key_inp : str
            Key to select the input numpy array (self.img_np[img_np_key_inp]).
        img_np_key : str
            Output dictionary key for image or numpy array.
        -- scaling parameters --
        (defined in seg_param/PARAM01_img_preproc.py as PARAM['SCALING_DICT'])
        nodata_out : int, optional
            The no-data value of the output image.
            If None then the nodata value from input array is taken.
        to_bit_num : int
            Number of bits for the output image.
        how : str
            Scaling method ('scale', 'stretch', etc.).
        perc : list
            Percentiles to use for scaling. (e.g. [0.2, 99.8])
        min_max : tuple
            Tuple of minimum and maximum values for scaling.
        std_fact : float
            Standard deviation factor for scaling (e.g. 4).
        gamma : float
            Gamma correction factor for scaling (e.g. 0.3).
        log_adj : float
            Logarithmic adjustment factor for scaling.

        Returns
        -------
        tuple
            A tuple containing the hist_plot instance and the metadata
            of the updated metadata.
        """

        if img_key_ref is None:
            img_key_ref = img_np_key_inp

        if nodata_out is None:
            nodata_out = nodata_inp

        # initialize new numpy array
        self.init_img(img_np_key, img_np_key_inp,
                      img_np_inp=self.img_np[img_np_key_inp],
                      meta_update=False)

        # scale intensity range
        self.img_np[img_np_key], scale_val = geo_utils.scale_convert_bit_np(
            self.img_np[img_np_key], nodata_inp,
            nodata_out=nodata_out, to_bit_num=to_bit_num,
            how=how, perc=perc, min_max=min_max,
            std_fact=std_fact, gamma=gamma, log_adj=log_adj)

        # create image
        attrs = {'AREA_OR_POINT': 'Area',
                 'orig_type': str(self.img_np[img_np_key].dtype),
                 'scale_factor': 1.0, 'add_offset': 0.0}

        # creat xarray from numpy array and update metadata
        self.nparray_to_img(img_np_key, img_key_ref,
                            attrs=attrs)


        # add scale vaues to metadata
        self.update_meta(
            img_np_key, scale_val, self.scale_cols)

        if plot_img:
            # add histogram to plot
            # !!! limit to 256 bins because it gets very slow !!!
            self.add_img_hist_plot_hv(
                img_np_key, nodata_out, min(256, to_bit_num),
                plot_name=img_np_key, range_tuple=(0, to_bit_num),
                overlay_plot=overlay_plot)

        if save_img:
            # save image and update metadta
            self.img_save(img_np_key)
            # remove image from self.image
            if del_img:
                del self.image[img_np_key]
                del self.img_np[img_np_key]

        return self.hist_plot, self.image_meta

    def hist_match_img(
            self, ref_img, nodata_inp, save_img=True, del_img=False,
            plot_img=True, overlay_plot=True,
            img_key_inp='raw', img_key=None, how='scikit',
            to_bit_num=2**8, hist_blend=0.5,
            file_placeh_out='',
            ax_hist_match_plt=None, set_mask_to=255,
            discard_nodata=False):
        """
        Histogram matches the input image to a reference image.

        Parameters
        ----------
        ref_img : numpy.ndarray
            Reference image used for histogram matching.
        nodata_inp : int
            Nodata value to be used during the matching process.
        save_img : bool
            Whether to save the image after processing.
        del_img : bool
            Whether to delete the image from the storage dictionary after
            saving.
        plot_img : bool
            Whether to plot the histogram of the matched image.
        overlay_plot : bool
            Whether to overlay the histogram plot on the image plot.
        img_key_inp : str
            Key of the input image in the storage dictionary.
        img_key : str, optional
            Key of the output image in the storage dictionary.
            If None, it will be the same as img_key_inp. (default is 'raw')
        how : str, optional
            Method to use for histogram matching. Can be 'scikit' or
            'augmentation'. (default is 'scikit')
        to_bit_num : int, optional
            Number of bits per pixel to use for the output image.
            (default is 256).
        hist_blend : float, optional
            Blend value for the histogram matching in the 'augmentation'
            method. (default is 0.5)
        file_placeh_out: place holder to be added to output key: corresponds to ref_id
        set_mask_to: float, int or string
            for hist matching with albumentation defines who nans ar treated
            optoins are:
            - is numeric: nans are replaces with the given number
                (e.g. use this to replace with np.nan, 0 or 255)
            - "mean": nans are replaced with the mean value
        discard_nodata: bool
            if nodata values should not be taken into account of the
            histogram matching with albumentations


        Returns
        -------
        tuple
            A tuple containing the histogram plot and the updated image
            metadata.
        """

        if img_key_inp not in self.image.keys():
            self.read_single_image(
                self.path_export,
                self.image_meta_df.loc[img_key_inp, 'file'].values[-1],
                img_key=img_key_inp, mask_nan=False)


        img_key_out = img_key.format(file_placeh_out)
        # intialize new ouput image and add metadata (makes copy of img_inp)
        self.init_img(img_key_out, img_key_inp,
                      img_inp=self.image[img_key_inp], meta_update=True)

        if how == 'scikit':
            self.image[img_key_out], scale_val = geo_utils.apply_hist_match(
                self.image[img_key_out], ref_img, path_exp=self.path_export,
                fig_suffix=img_key_out,
                ax_hist_match_plt=ax_hist_match_plt,
                file_prefix=self.file_prefix)
        else:
            self.image[img_key_out], scale_val = geo_utils.augmentation_hist_match(
                ref_img, self.image[img_key_out],
                hist_blend=hist_blend, to_bit_num=to_bit_num,
                nodata_val=nodata_inp, path_exp=self.path_export,
                fig_suffix=img_key_out,
                ax_hist_match_plt=ax_hist_match_plt,
                file_prefix=self.file_prefix,
                set_mask_to=set_mask_to, discard_nodata=discard_nodata)

        # add scale values to metadata
        self.update_meta(img_key_out, scale_val, self.scale_cols)

        if plot_img:
            # add histogram to plot
            # !!! limit to 256 bins because it gets very !!!
            self.add_img_hist_plot_hv(
                img_key_out, nodata_inp, min(256, to_bit_num),
                plot_name=img_key_out, range_tuple=(0, to_bit_num),
                overlay_plot=overlay_plot)

        if save_img:
            # save image and update metadta
            self.img_save(img_key_out)
            # remove image from self.image
            if del_img:
                del self.image[img_key_out]

        return self.hist_plot, self.image_meta

    def get_stats_set(self, key_lst, aoi_dict=None, path_inp=None,
                      epsg_target=None):
        """
        Calculates the statistics of the image stored in 'self.image'
        for the full omiage as well as subimage sections as defined by
        an input AOI.

        Parameters:
        ----------
        - key_lst: list
            A list of keys for which texture statistics need to be
            calculated.
        - aoi_dict: dict, optional
            A dictionary where keys are AOI names and values are GeoJSON
            file paths for those AOIs.
        - path_inp: str, optional
            The input path to the directory containing the AOI GeoJSON files.
        - epsg_target: int, optional
            The target EPSG code for the AOI geometries. Defaults to None.

        Returns:
        ----------
        - None: The method modifies the 'self.image' dictionary in place,
             adding or updating texture statistics for specified keys
             and AOIs (if provided).

        TODO could write parallelized version of this
        """

        for i_key in key_lst:
            # get stats of full image
            self.get_stats(i_key)

            if aoi_dict is not None:
                for i_aoi, i_aoi_file in  aoi_dict.items():
                    # get stats of aoi clipped image
                    AOI_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
                        os.path.join(path_inp, i_aoi_file),
                                    epsg_target)
                    self.image[f'{i_key}:{i_aoi}'] = geo_utils.clip_to_aoi(
                        self.image[f'{i_key}'], AOI_coords)
                    self.get_stats(f'{i_key}:{i_aoi}')
                    del self.image[f'{i_key}:{i_aoi}']
        return

    def get_stats(self, name):
        """
        Calculate and store statistics for the specified image.

        Parameters:
        ----------
        - name: str
            The key name of the image dcitionary for whoich to
            to calculate statistics.

        Returns:
        ----------
        None

        Notes:
        - It computes mean, standard deviation, minimum, maximum
            and quantile values for the raw and the standardised image.
        - The function also calculates the statistics fo rthe
            logarithmic and exponential transformations of the image
            (this is mainly useful for e.g. texture features).
        - The logarithmic and exponential transformation the image values
            are shifted to avoid infinite values due to zeros.
        - The results are stored in the `self.stats` dictionary with
            the image name as key.
        """

        # Stats are calculated on self.image and per band. Thus cannot
        # allow chunking along x or y "-1" means no chunking along that
        # direction
        img = self.image[name].chunk(dict(x=-1, y=-1))

        # -- get statistics for image
        stats1 = stats_utils.get_stats_df(
            img, perc_min=0.5, perc_max=99.5)

        # --- get statistics for logratihmic and exponential
        # transformations
        inp_min = stats1['min'].to_xarray()
        # shift to avoid inf values due to zeros
        img_shift = img + (inp_min + 0.1)
        stats2 = stats_utils.get_stats_df(
            np.log(img_shift), perc_min=0.5, perc_max=99.5,
            prefix='log_')
        stats3 = stats_utils.get_stats_df(
            np.exp(img_shift), perc_min=0.5, perc_max=99.5,
            prefix='exp_')

        # -- concatenate all stat values into a dataframe
        self.stats[name] = pd.concat([stats1, stats2, stats3], axis=1)
        return

    def save_stats(self, proc_suffix, proc_nr_str=''):
        """
        Save the collected statistics to a txt file.

        Parameters:
        ----------
        proc_suffix: str
            The processing suffix to be used in the file name.
        proc_nr_str: str, optional
            A string to append to the file name to indicate
            a specific process or iteration number.
            Defaults to an empty string.

        Returns:
        ----------
        None
        """

        df_stats = pd.concat(self.stats, axis=0, names=['name'])

        if proc_nr_str != '':
            proc_nr_str = '_' + proc_nr_str

        file_name = (
            self.file_prefix + proc_nr_str + '_' + proc_suffix
            + '_stats_file.txt')

        path_file = os.path.join(self.path_export, file_name)

        try:
            df_stats.to_csv(
                path_file, '\t', lineterminator='\n', header=True)
        except:
            df_stats.to_csv(path_file, '\t', header=True)
        return

    def add_img_hist_plot_hv(
            self, img_key, nodata_val, num_bin,
            plot_name=None, band_lst=None, range_tuple=None,
            overlay_plot=True):

        if plot_name is None:
            plot_name = img_key

        if overlay_plot:
            width = 1200
            height = 600
        else:
            width = 600
            height = 600

        if band_lst is None:
            band_lst = self.image[img_key].band.values.tolist()

            self.hist_plot[img_key] = []
            for i_band in band_lst:
                self.hist_plot[img_key].append(plotting_utils.plot_hist(
                    self.image[img_key].sel(band=i_band).values,
                    f'{plot_name}_{i_band}', alpha=0.3, nodata=nodata_val,
                    bin_num=num_bin, width=width, height=height,
                    range_tuple=range_tuple))
        return

    def save_img_hist_plot_hv(self, name_suffix, col_num=2,
                              fig_size=None, overlay_plot=False):

        if fig_size is None:
            fig_size=(8.3, 11.7)  # [wsize, hsize] in inch

        hist_lst = [[x.opts(**y) for x, y in xx]
                    for xx in self.hist_plot.values()]

        hist_plt = [functools.reduce(operator.mul, x) for x in hist_lst]

        if overlay_plot:
            plot_all = functools.reduce(operator.mul, hist_plt)
        else:
            try:
                plot_all = functools.reduce(
                    operator.add, hist_plt).cols(col_num).opts(**{'fig_size':200})
            except:
                plot_all = functools.reduce(
                    operator.add, hist_plt).opts(**{'fig_size':200})

        file_name = f"{self.file_prefix}_hist_plot_{name_suffix}"
        plotting_utils.save_img_hv(
            plot_all, self.path_export, file_name, file_type='pdf')

        del self.hist_plot
        self.hist_plot = {}

        return

    def update_meta(self, inp_key, inp_lst, inp_cols):

        dict_add = {x: y for x, y in zip(inp_cols, inp_lst)}

        if inp_key in self.image_meta.keys():
            self.image_meta[inp_key].update(dict_add)
        else:
            self.image_meta[inp_key] = dict_add

        return

    def update_meta_df(self):
        self.image_meta_df = pd.concat(
            [self.image_meta_df] + [pd.DataFrame(y, index=[x])
            for x, y in self.image_meta.items()], axis=0).drop_duplicates(keep='last')
        self.image_meta = {}
        return

    def update_save_metadata(self, proc_suffix, proc_nr_str=''):

        if proc_nr_str != '':
            proc_nr_str = '_' + proc_nr_str

        # update meta dataframe
        self.update_meta_df()

        file_name = (
            self.file_prefix + proc_nr_str + '_' + proc_suffix
            + '_proc_file.txt')
        path_file = os.path.join(self.path_export, file_name)
        try:
            self.image_meta_df.to_csv(
                path_file, sep='\t', lineterminator='\n', header=True)
        except:
            self.image_meta_df.to_csv(path_file, '\t', header=True)
        return

    def read_files(self, path_import, proc_file_name_prefix,
                   proc_suffix, img_meta_inp=None,
                   query_text="proc_type==@proc_suffix",
                   proc_nr_str='', chunk='auto'):

        if img_meta_inp is None:
            img_meta_inp = read_proc_file(
                path_import, proc_file_name_prefix, proc_suffix,
                proc_nr_str=proc_nr_str).query(query_text)

        for i_name, i_file, i_band in zip(img_meta_inp['name'], img_meta_inp['file'], img_meta_inp['bands']):
            if i_file == '-':
                continue

            self.image[i_name] = geo_utils.read_to_xarray(
                os.path.join(path_import, i_file), mask_nan=False,
                chunk=chunk)

            if 'long_name' not in self.image[i_name].attrs.keys():
                try:
                    self.image[i_name]['band'] = (
                        'band', i_band.split(':'))
                except:
                    self.image[i_name]['band'] = ('band', [i_band])
            else:
                geo_utils.add_band_from_long_name(self.image[i_name])

        self.image_meta_df = pd.concat(
            [self.image_meta_df, img_proc], axis=0).drop_duplicates(keep='last').sort_index()
        del img_proc

        return

    def read_single_image(
            self, PATH, FILE, img_key='raw', mask_nan=False, chunk='auto',
            meta_update=True):

        # read image
        self.image[img_key] = geo_utils.read_to_xarray(
            os.path.join(PATH, FILE),
            mask_nan=mask_nan, chunk=chunk)
        self.image[img_key] = geo_utils.check_for_missing_fill_val(
            self.image[img_key])

        geo_utils.add_band_from_long_name(self.image[img_key])

        bands = self.image[img_key].band.values.tolist()
        if meta_update:
            self.update_meta(
                img_key,
                [img_key, 'inp', 'raw',
                ':'.join([str(x) for x in bands]), FILE,
                dt.datetime.now().strftime('%Y-%m-%d_%H%M')],
                self.meta_cols[:-1])
        return

    def img_save(self, img_key):
        """
        Save single image specified by key of storage dictionary
        """

        filename = (self.file_prefix + '_' + img_key)
        geo_utils.save_to_geotiff_options(
            self.image[img_key], self.path_export, filename,
            single=False)

        # add export file name to metadata
        self.update_meta(
            img_key,
            [filename + '.tif'], self.meta_cols[-1:])

        return

    def image_save_several(
            self, search_col, search_val, del_saved_array=False):
        """
        Saves multiple images that match the specified search criteria.

        Parameters:
        ----------
        search_col: str:
            The column name in the image metadata to search
            for the specified value.
        search_val: str
            The value to search for in the specified column.
        del_saved_array: bool, optional
            If True, deletes the saved image arrays from the memory
            after saving. Defaults to False.

        Returns:
        ----------
            None
        """

        self.update_meta_df()
        df_search = self.image_meta_df.loc[self.image_meta_df[search_col] == search_val, :]

        for i in df_search['name']:
            filename = (self.file_prefix + '_' + i)
            geo_utils.save_to_geotiff_options(
                self.image[i], self.path_export, filename, single=False)

            self.image_meta_df.loc[i, 'file'] = (filename + '.tif')

            if del_saved_array:
                del self.image[i]

        return


def convert_bit_img_parallel(
        image_inp, img_np_imp, nodata_inp, scale_dict, PARAM,
        img_key_inp='raw_inp', proc_type='img'):
    """
    Function to be called if run convert_bit_img in parallel

    Note: This functions creates separate image class instance for
    parallel processing to ensure thread safety (although the
    convert_bit_img should be thread safe as new disctionary keys are
    created for each image conversion)
    """

    img_subproc = ImagePreproc(
        PARAM['EPSG_TARGET'], PARAM['PATH_EXPORT'], PARAM['FILE_PREFIX'])

    # initialize image and numpy array
    img_subproc.init_img(
        img_key_inp, 'inp', img_inp=image_inp, img_np_inp=img_np_imp,
        proc_type=proc_type, meta_update=False)


    img_out, meta_out = img_subproc.convert_bit_img(
        nodata_inp, del_img=True, img_np_key_inp=img_key_inp,
        **scale_dict)

    del img_subproc
    return img_out, meta_out


def hist_match_img_parallel(
        ref_img, nodata_inp, hist_match_dict, PARAM,
        proc_type='img', file_placeh_out='', image_inp=None):
    '''
    Function to be called is run hist_match_img in parallel

    Either image on which to perform hist match can be provided as input or otherwise it
    will be read in function hist_match_img()

    Note: This functions creates separate image class instance for parallel processing to
    ensure thread safety (althouhght the convert_bit_img should be thread
    safe as new disctionary keys are created for each image conversion)

    img_key_inp is defined in hist_match_dict,

    file_placeh_out: place holder to be added to output key: corresponds to ref_id
    '''

    img_subproc = ImagePreproc(
        PARAM['EPSG_TARGET'], PARAM['PATH_EXPORT'], PARAM['FILE_PREFIX'])

    # initialize image
    if image_inp is not None:
        img_subproc.init_img(
            hist_match_dict['img_key_inp'], 'inp', img_inp=image_inp,
            proc_type=proc_type, meta_update=False)

    img_out, meta_out = img_subproc.hist_match_img(
        ref_img, nodata_inp, del_img=True, file_placeh_out=file_placeh_out,
        **hist_match_dict)


    del img_subproc
    return img_out, meta_out


def read_file_from_proc_file(path_import, img_meta, chunk=None):
    '''
    used for parallel reading of files according to
    metadata file (image_meta_df)
    '''
    img = {}
    for i_name, i_file, i_band in zip(img_meta['name'], img_meta['file'], img_meta['bands']):
        if i_file == '-':
            continue

        img[i_name] = geo_utils.read_to_xarray(
            os.path.join(path_import, i_file), mask_nan=False, chunk=chunk)

        if 'long_name' not in img[i_name].attrs.keys():
            try:
                img[i_name]['band'] = ('band', i_band.split(':'))
            except:
                img[i_name]['band'] = ('band', [i_band])
        else:
            geo_utils.add_band_from_long_name(img[i_name])
    return img


def read_proc_file(
        path_import, proc_file_name_prefix, proc_suffix, proc_nr_str=''):

    if proc_nr_str != '':
            proc_nr_str = '_' + proc_nr_str

    file_name = (
        proc_file_name_prefix + proc_nr_str
        + '_' + proc_suffix + '_proc_file.txt')

    path_file = os.path.join(path_import, file_name)

    if not os.path.isfile(path_file):
        print('Proc file does NOT exist yet: ' + path_file)
        return pd.DataFrame()

    img_proc = pd.read_csv(
        path_file, sep='\t', header=0, index_col=0)
    return img_proc


