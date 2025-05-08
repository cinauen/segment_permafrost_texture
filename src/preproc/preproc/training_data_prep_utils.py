"""
Module with utility functions for training data preparation
"""

import os
import sys

import os
import sys
import numpy as np
import pandas as pd
import xarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt


PATH_BASE_SEG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

# ===== import specific utils =====
sys.path.append(PATH_BASE_SEG)

import utils.geo_utils as geo_utils
import utils.conversion_utils as conversion_utils
import utils.numpy_utils as numpy_utils



def prep_class_img(path_inp, file_inp, AOI_coords=None, data_img=None,
                   drop_na=False, from_disk=False):
    '''
    Prepare validation data:
    1) Read image class raster
    2) reproject to data_img
    3) change to int type
    4) Fill nans with 7
    5) Clip to training AOI ==> sets class outside AOI to zero again
    6) Renumber or merge classes: TODO if required!!!

    !!! it is here best not to use from_disk as from disk might introduce
    clipping errors at boarders
    '''
    class_img = geo_utils.read_to_xarray(
        os.path.join(path_inp, file_inp), mask_nan=False)

    if data_img is not None:
        class_img = class_img.rio.reproject_match(
            data_img, Resampling=Resampling.nearest)

    # convert to uint8
    class_img_uint = geo_utils.convert_img_to_dtype(
        class_img, dtype_out='uint8', nodata_out=0)

    # fill zeros with 7 because background undisturbed was defined as nan
    # use here first band only (incase second band is quali)
    #class_img_uint[0, :, :] = replace_class_nr_in_array(
    #    class_img_uint[0, :, :], 0, 3)  # !!! changed

    # clip to AOI
    if AOI_coords is not None:
        class_img_uint = geo_utils.clip_to_aoi(
            class_img_uint, AOI_coords, drop_na=drop_na,
            from_disk=from_disk)


    if class_img_uint.shape[0] == 1:
        class_img_uint['band'] = ('band', ['class'])
    else:
        class_img_uint['band'] = ('band', ['class', 'quali'])

    #class_img_uint.rio.to_raster(
    #    os.path.join(path_inp, 'test.tif'), driver='GTiff')

    return class_img_uint


def replace_class_nr_in_array(img, val_to_replace, val_new):

    # fill zeros with 7
    mask_array = img.values != val_to_replace
    img = img.where(mask_array, other=val_new)
    # Value to use for locations in this object where cond is False.
    # By default, these locations filled with NA.

    return img


def relabel_img(img_inp, rename_classes):
    img = img_inp.copy()

    nodata = img_inp.rio.nodata
    if nodata is None:
        try:
            nodata = img_inp.attrs['_FillValue']
        except:
            pass

    # with copy nodata is lost
    img.rio.write_nodata(nodata, inplace=True)

    mask_rename = {}
    for i_key in rename_classes.keys():
        mask_rename[i_key] = img.values != i_key

    for i_key, i_val in rename_classes.items():
        img = img.where(mask_rename[i_key], other=i_val)

    return img


def get_class_count(class_img, label_num, label_naming, data_set):
    '''
    label_naming is PARAM['LABEL_NAMING']
    '''
    class_name, class_count = np.unique(
        class_img.data, return_counts=True)

    nan_label = label_num[label_naming.index('nan')]
    counts_no_nan = [y for x, y in zip(class_name, class_count) if x != nan_label]
    sum_no_zero = np.sum(counts_no_nan)

    # class with highest occurence
    class_maj = class_name[np.argmax(class_count)]

    # derive percentage occurence per class
    count_sum = np.sum(class_count)
    class_name_lst = class_name.tolist()
    class_perc_lst = {y: [data_set, x, ((class_count[class_name_lst.index(x)]/count_sum)*100).round(3) if x in class_name_lst else 0]
                              for x, y in zip(label_num, label_naming)}
    class_perc_lst_no_zero = [((class_count[class_name_lst.index(x)]/sum_no_zero)*100).round(3) if (x in class_name_lst and x != 0) else 0 for x in label_num]

    out = pd.DataFrame.from_dict(
        class_perc_lst, orient='index',
        columns=['data_set', 'class_num', 'perc'])
    out['perc_no_nan'] = class_perc_lst_no_zero
    out.index.name = 'class'
    out = out.reset_index().set_index(['data_set', 'class', 'class_num'])
    return out.reset_index()


def read_append_df(
        path, file_name, df, drop_subset):

    path_file = os.path.join(path, file_name)
    if os.path.isfile(path_file):
        df_read = pd.read_csv(
            path_file, sep='\t', header=0, index_col=0)

        df = pd.concat([df_read, df], axis=0, names=['name'])
        df.drop_duplicates(subset=drop_subset, inplace=True)

    df.to_csv(path_or_buf=path_file, sep='\t', header=True)

    return


def merge_img_to_analyse(data_img, path_inp, file_inp_lst, file_band_lst,
                         AOI_coords, resampling_type='bilinear',
                        class_img=None, band_prefix_lst=None,
                        EPSG_TARGET=None):
    '''
    5) Read additional images to analyse
    6) Clip grid to analyse
    7) Reproject match validation image grid to analyse
            (cluster images and texture measures)
    8) merge images

    !!! types of images to include should habve been converted to same type as
    img_analyse
    !!!! class image needs to have been reprojected and matched to data_img before

    file_band_lst contains sublist with bands to take for each file in file_inp_lst
    if no sublist but None then take all bands. E.g.
    [['pca_x0', 'pca_x1', 'pca_x2'], None]

    band_prefix_lst is sublist with same lenght as file_inp_lst.
        This can define a prefix to be added to the band name.
    '''
    if band_prefix_lst is None:
        band_prefix_lst = [None]*len(file_inp_lst)

    img_analyse_lst = []
    all_bands = []
    count = 0
    for i_band, i in zip(file_band_lst, file_inp_lst):
        img_analyse = geo_utils.read_to_xarray(
            os.path.join(path_inp, i), mask_nan=False)

        if EPSG_TARGET is not None:
            img_analyse = img_analyse.rio.reproject(
                "EPSG:" + str(EPSG_TARGET))

        geo_utils.add_band_from_long_name(img_analyse)

        # selet required bands
        if i_band is not None:
            img_analyse = img_analyse.sel(band=i_band)

        # make sure that all data are float values
        img_analyse = geo_utils.set_fill_val_to_nan(img_analyse)

        # clip to AOI
        img_analyse = geo_utils.clip_to_aoi(img_analyse, AOI_coords)

        # match resolution etc to reference image
        img_analyse = img_analyse.rio.reproject_match(
            data_img, Resampling=Resampling[resampling_type])

        if band_prefix_lst[count] is not None:
            band_prefix = band_prefix_lst[count]
        else:
            band_prefix = str(count)

        renamed_bands = [band_prefix + '_' + str(x) for x in img_analyse.band.values.tolist()]
        img_analyse['band'] = ('band', renamed_bands)
        img_analyse_lst.append(img_analyse)
        count += 1

    # merge images
    if class_img is not None:
        img_lst = [data_img] + img_analyse_lst + [class_img]
    else:
        img_lst = [data_img] + img_analyse_lst
    # get fill values
    bands_all = [x.band.values.tolist() for x in img_lst]
    fill_vals = [[x.rio.nodata]*len(y) for x, y in zip(img_lst, bands_all)]
    fill_dict = {x: y for x, y in zip(np.hstack(bands_all), np.hstack(fill_vals))}
    img_merged = xarray.concat(img_lst, dim='band')
    img_merged.attrs['long_name'] = tuple(img_merged.band.values)
    img_merged.attrs['fill_vals'] = fill_dict
    #img_merged = geo_utils.convert_img_to_dtype(
    #    img_merged, dtype_out='float64', nodata_out=np.nan)

    return img_merged


def get_gdf_drop_missing_class(img, drop_col_subset=None):

    gdf_out, crs_img, feature_cols = conversion_utils.img_to_df_proc(
        img, rename_dict=None, use_GPU=False)

    # drop vaues without class (outside AOI)
    gdf_out.loc[gdf_out['class'] == 0, 'class'] = np.nan
    if drop_col_subset is not None:
        gdf_out.dropna(subset=drop_col_subset, how='all', inplace=True)
    elif 'class_pred' in gdf_out.columns:
        # only required for validating data
        gdf_out.dropna(
            subset=['class_pred', 'class'], how='any', inplace=True)
    else:
        gdf_out.dropna(subset=['class'], inplace=True)

    return gdf_out


def rotate_img(img, gdf_inp, rot_degree, EPSG_TARGET, interp_method_lst,
               AOI_coords=None, same_resolution=False):
    '''

    rotation can be used to help better clipping of training images

    rot degree: use neg values for counter clockwise

    interp_method for int data use nearest, otherwise use cubic

    gdf: all nan values must have beed dropped already (e.g. with
    get_gdf_drop_missing_class(img) or get_gdf_dropna(img))
    '''
    import utils.coord_utils as coord_utils

    gdf = gdf_inp.copy()
    gdf['z'] = 0

    # extract coords
    coords_inp = gdf.loc[:, ['x', 'y', 'z']]

    # --- rotate img
    # get min to normalize before rotation
    min_vals = np.nanmin(gdf.loc[:, ['x', 'y', 'z']].values, axis=0)
    # normalize and rotate
    coords_rot = coord_utils.rotate_coord_list(
        coords_inp.values, coord_utils.degree_to_rad(rot_degree), min_vals).round(6)

    # add rotated coords to gdf
    gdf['x'] = coords_rot[:, 0]
    gdf['y'] = coords_rot[:, 1]
    gdf.drop('z', axis=1, inplace=True)

    # -- create regular gird to build new xarray
    xy_min = gdf.loc[:, ['x', 'y']].min(axis=0).values
    xy_max = gdf.loc[:, ['x', 'y']].max(axis=0).values
    xy_corners = np.stack([xy_min, xy_max]).T.ravel().tolist()
    # get interpolation grid
    if same_resolution:
        # keep same resolution as original image
        resol = np.abs(img.rio.resolution()).round(4)
        interp_grid_x, interp_grid_z, dx, dz = numpy_utils.get_regular_grid_coords(
            x=None, y=None,  # topo_coords_rot[:, 0]
            dx=resol[0], dy=resol[1],
            nx=None, ny=None,
            grid_corners=xy_corners, cell_centers=True, corner=False)
    else:
        # keep same amount of pixels as original image
        interp_grid_x, interp_grid_z, dx, dz = numpy_utils.get_regular_grid_coords(
            x=None, y=None,  # topo_coords_rot[:, 0]
            dx=None, dy=None,
            nx=img.x.shape[0], ny=img.y.shape[0],
            grid_corners=xy_corners, cell_centers=True, corner=False)

    # interpolate all bands
    cols = [x for x in list(gdf.columns) if x not in ['x', 'y']]
    data_set = []
    for e, i in enumerate(cols):
        data_set.append(numpy_utils.interp2D_regular_grid_test(
            gdf.x.values, gdf.y.values, gdf[i].values,
            method=interp_method_lst[e],
            interp_grid_x=interp_grid_x, interp_grid_y=interp_grid_z))

        # for integer values and if interpolation was not nearest
        if interp_method_lst[e] != 'nearest' and i in ['1', 'class', 'quali']:
            data_round = np.round(data_set[-1])
            data_nearest = numpy_utils.interp2D_regular_grid_test(
                gdf.x.values, gdf.y.values, gdf[i].values,
                method='nearest',
                interp_grid_x=interp_grid_x, interp_grid_y=interp_grid_z)
            data_set[-1] = np.where(
                (data_round >= np.nanmin(data_nearest)) & (data_round <= np.nanmax(data_nearest)),
                data_round, data_nearest)

    # create and save xarray
    img_out = geo_utils.create_xarray(
        interp_grid_x[0, :].ravel(), interp_grid_z[:, 1].ravel(),
        np.stack(data_set), cols, EPSG_TARGET, reverse=False)

    if AOI_coords is not None:
        AOI_inp_rot = np.hstack([AOI_coords, np.zeros((len(AOI_coords), 1))])
        AOI_rot = coord_utils.rotate_coord_list(
            AOI_inp_rot, coord_utils.degree_to_rad(rot_degree), min_vals)[:, :2].tolist()
        img_out = geo_utils.clip_to_aoi(img_out, AOI_rot, drop_na=False)

    img_out.attrs.update(img.attrs)
    img_out.rio.write_nodata(img.rio.nodata, inplace=True)

    return img_out


def export_split_img(img_split, add_channels_file_lst, file_prefix, PARAM,
                    subfolder_save, suffix_nr,
                    window_size,
                    trim_sub_img=False,
                    min_perc_data=20, out_type_data=None,
                    num_class_band=1):
    '''
    if file has less than min_perc_data it will be rejected

    class image is clippped to exact window size (not padded size)
    data bands are not cliped due to GLCM calculation (will be clipped
    later after GLCM calculation)
    '''
    if not os.path.isdir(subfolder_save):
        os.mkdir(subfolder_save)

    band_lst = img_split.band.values.tolist()
    data_file_str = ':'.join(add_channels_file_lst)
    img_bands_str = ':'.join(band_lst)
    # the last or the two last bands are y_mask
    x_bands_str = ':'.join(band_lst[:-num_class_band])
    y_bands_str = ':'.join(band_lst[-num_class_band:])

    y_coarse_inp = img_split.y_coarse.values.tolist()
    x_coarse_inp = img_split.x_coarse.values.tolist()
    meta_list = []
    for i_y in y_coarse_inp:
        for i_x in x_coarse_inp:
            img_sep = img_split.sel(y_coarse=i_y, x_coarse=i_x).copy()
            img_sep = img_sep.swap_dims({'y_fine': 'y', 'x_fine': 'x'})
            #img_sep = img_sep.rename({'y_fine': 'y', 'x_fine': 'x'})
            #img_sep = img_sep.set_index(y='y', x='x')

            clip_y = int((img_sep.shape[1] - window_size)/2)
            clip_x = int((img_sep.shape[2] - window_size)/2)

            # handle sub images with mising values (if window size didn't
            # fit)
            if trim_sub_img:
                # trim image
                img_sep = img_sep.where(
                    (~np.isnan(img_sep.x) & ~np.isnan(img_sep.y)),
                    drop=True)
            else:
                # extrapolate/interpolate missing coordinates
                # thus keep missing as nan
                geo_utils.xarray_interp_coords(img_sep.x.values)
                geo_utils.xarray_interp_coords(img_sep.y.values)

            # extract segmentation
            if num_class_band == 1:
                class_sub_img = img_sep.sel(band=['class'])
            else:
                class_sub_img = img_sep.sel(band=['class', 'quali'])

            # change segmentation to integer numbers
            # change to uint (replaces nan with zero)
            class_sub_img = geo_utils.convert_img_to_dtype(
                class_sub_img, dtype_out='uint8', nodata_out=0)

            # exclue images where all are nan
            if np.all(class_sub_img.data == 0):
                continue

            # extract correct window size for class labels
            #class_name, class_count = np.unique(
            #    class_sub_img.data, return_counts=True)
            # select first band in case have quali in class_sub_img as well
            # and use center of image with correct window size
            class_count_sub_img = class_sub_img[0, clip_y:-clip_y, clip_x:-clip_x]
            if class_count_sub_img.shape != (window_size, window_size):
                sys.exit('error the window size is not correct')
            class_name, class_count = np.unique(
                class_count_sub_img.data, return_counts=True)

            try:
                class_name = class_name.compute()
                class_count = class_count.compute()
            except:
                pass

            # class with highest occurence
            class_maj = class_name[np.argmax(class_count)]

            # derive percentage occurence per class
            center_coord_x = np.mean(class_sub_img.x.values)
            center_coord_y = np.mean(class_sub_img.y.values)
            count_sum = np.sum(class_count)
            class_name_lst = class_name.tolist()
            # add also 0 for bacground value number
            class_perc_lst = [
                ((class_count[class_name_lst.index(x)]/count_sum)*100).round(3) if x in class_name_lst else 0
                for x in PARAM['LABEL_NAMING'][0]]
            class_count_lst = [
                class_count[class_name_lst.index(x)] if x in class_name_lst else 0
                for x in PARAM['LABEL_NAMING'][0]]
            perc_all = np.sum(class_perc_lst[1:])
            if perc_all < min_perc_data:
                print('more than 20% nan data: ' + str(100-perc_all))
                continue

            file_name_prefix = (
                file_prefix + '_' + '{0:02d}'.format(suffix_nr) + '_'
                + '{0:02d}'.format(i_y) + '-' + '{0:02d}'.format(i_x))
            file_name_class = file_name_prefix + '_seg.tif'
            file_name_data = file_name_prefix + '_data.tif'

            meta_list.append(
                [suffix_nr, i_y, i_x, center_coord_x, center_coord_y,
                 file_name_class, file_name_data, class_maj,
                 img_bands_str, x_bands_str, y_bands_str,
                 PARAM['DATA_IMG'], PARAM['LABEL_FILE_INP'][PARAM['labelling_area']], data_file_str] + class_perc_lst + class_count_lst)

            # save subfile
            class_sub_img.attrs.pop('fill_vals')
            geo_utils.save_to_geotiff(
                class_sub_img, subfolder_save,
                file_name_class, suffix='', add_crs=False)

            # save data file
            data_bands = [
                x for x in img_sep.band.values if x != 'class' and x != 'quali']
            img_out = img_sep.sel(band=data_bands)
            if out_type_data is not None:
                img_out = geo_utils.convert_img_to_dtype(
                    img_out, dtype_out=out_type_data)
            elif 'fill_vals' in img_out.attrs.keys():
                img_out.rio.write_nodata(
                    img_out.attrs['fill_vals'][data_bands[0]],
                    inplace=True, encoded=True)
            img_out.attrs.pop('fill_vals')
            geo_utils.save_to_geotiff(
                img_out, subfolder_save,
                file_name_data, suffix='', add_crs=False)

    perc_name = ['perc-class_' + str(x) for x in PARAM['LABEL_NAMING'][0]]
    count_name = ['count-class_' + str(x) for x in PARAM['LABEL_NAMING'][0]]
    cols_inp = ['shift_nr', 'y_coarse', 'x_coarse', 'center_coord_x', 'center_coord_y',
                'file_class', 'file_data', 'class_maj', 'bands', 'x_bands', 'y_bands',
                'data_img', 'class_img', 'data_files'] + perc_name + count_name
    df_meta = pd.DataFrame(meta_list, columns=cols_inp)
    meta_filename = os.path.join(subfolder_save, file_prefix + '_meta_data.txt')
    if os.path.isfile(meta_filename):
        df_meta_old = pd.read_csv(
            meta_filename, sep='\t', header=0, index_col=0)
        df_meta = pd.concat([df_meta_old, df_meta], axis=0)
        df_meta = df_meta.drop_duplicates(subset=['file_data', 'file_class'], keep='last')
    df_meta.to_csv(path_or_buf=meta_filename, sep='\t', header=True)

    return


def export_split_data_tile(
        img_split, add_channels_file_lst, file_prefix, PARAM,
        subfolder_save, suffix_nr,
        trim_sub_img=False, out_type_data=None):
    '''
    if tile has less than min_perc_data it will be rejected

    '''
    if not os.path.isdir(subfolder_save):
        os.mkdir(subfolder_save)

    band_lst = img_split.band.values.tolist()
    #file_prefix_class = PARAM['LABEL_FILE_INP'].split('.')[0]
    data_file_str = ':'.join(add_channels_file_lst)
    img_bands_str = ':'.join(band_lst)
    x_bands_str = ':'.join(band_lst)

    y_coarse_inp = img_split.y_coarse.values.tolist()
    x_coarse_inp = img_split.x_coarse.values.tolist()
    meta_list = []
    for i_y in y_coarse_inp:
        for i_x in x_coarse_inp:
            img_sep = img_split.sel(y_coarse=i_y, x_coarse=i_x).copy()
            img_sep = img_sep.swap_dims({'y_fine': 'y', 'x_fine': 'x'})
            # img_sep = img_sep.rename({'y_fine': 'y', 'x_fine': 'x'})
            # img_sep = img_sep.set_index(y='y', x='x')

            # handle sub images with mising values (if window size didn't
            # fit)
            if trim_sub_img:
                # trim image
                img_sep = img_sep.where(
                    (~np.isnan(img_sep.x) & ~np.isnan(img_sep.y)), drop=True)
            else:
                # extrapolate/interpolate missing coordinates
                # thus keep missing as nan
                geo_utils.xarray_interp_coords(img_sep.x.values)
                geo_utils.xarray_interp_coords(img_sep.y.values)

            # exclue images where all are nan
            if np.all(np.isnan(img_sep.data)):
                continue

            # derive metadata
            center_coord_x = np.mean(img_sep.x.values)
            center_coord_y = np.mean(img_sep.y.values)

            file_name_prefix = (
                file_prefix + '_' + '{0:02d}'.format(suffix_nr) + '_'
                + '{0:02d}'.format(i_y) + '-' + '{0:02d}'.format(i_x))
            file_name_data = file_name_prefix + '_data.tif'

            meta_list.append(
                [suffix_nr, i_y, i_x, center_coord_x, center_coord_y,
                 file_name_data,
                 img_bands_str, x_bands_str,
                 PARAM['DATA_IMG'], data_file_str])

            # save data file
            data_bands = img_sep.band.values.tolist()
            img_out = img_sep.sel(band=data_bands)
            if out_type_data is not None:
                img_out = geo_utils.convert_img_to_dtype(
                    img_out, dtype_out=out_type_data)
            elif 'fill_vals' in img_out.attrs.keys():
                img_out.rio.write_nodata(
                    img_out.attrs['fill_vals'][data_bands[0]],
                    inplace=True, encoded=True)
                img_out.attrs.pop('fill_vals')
            geo_utils.save_to_geotiff(
                img_out, subfolder_save,
                file_name_data, suffix='', add_crs=False)

    cols_inp = ['shift_nr', 'y_coarse', 'x_coarse',
                'center_coord_x', 'center_coord_y',
                'file_data', 'bands', 'x_bands',
                'data_img', 'data_files']
    df_meta = pd.DataFrame(meta_list, columns=cols_inp)
    meta_filename = os.path.join(subfolder_save, file_prefix + '_meta_data.txt')
    if os.path.isfile(meta_filename):
        df_meta_old = pd.read_csv(
            meta_filename, sep='\t', header=0, index_col=0)
        df_meta = pd.concat([df_meta_old, df_meta], axis=0)
        df_meta = df_meta.drop_duplicates(subset=['file_data'], keep='last')
    df_meta.to_csv(path_or_buf=meta_filename, sep='\t', header=True)

    return


def plot_chip_overview(img, phase, count, path, band='class'):
    '''
    plotting oveview of image files which will be used for training
    '''
    img_shape = img.shape
    if img_shape[1] > 1 and img_shape[3] > 1:
        fig = img.sel(band=band).plot(
            x="x_fine", y="y_fine", col="x_coarse", row="y_coarse",
            yincrease=False, subplot_kws={'aspect': 'equal'})
    elif img_shape[1] <= 1 and img_shape[3] <= 1:
        fig = img.sel(band=band).plot(
            x="x_fine", y="y_fine", yincrease=False)
    elif img_shape[1] <= 1:
        # for y only one tile
        fig = img.sel(band=band).plot(
            x="x_fine", y="y_fine", col="x_coarse",
            yincrease=False, subplot_kws={'aspect': 'equal'})
    elif img_shape[3] <= 1:
        # for x only one tile
        fig = img.sel(band=band).plot(
            x="x_fine", y="y_fine", row="y_coarse",
            yincrease=False, subplot_kws={'aspect': 'equal'})

    plot_name = 'img_overview_' + phase + '_' + '{0:02d}'.format(count) + '.png'

    # need double fig.fig to access the figure object from the FacetGrid
    fig.fig.savefig(os.path.join(path, plot_name), format='png')
    # plt.close() # commented since gave tkinter error
    return


