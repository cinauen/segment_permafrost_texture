
"""
Functions to manipulate raster and vector files.
They are mainly based on rioxarray, xarray, skimage and
albumentations (raster) and shapely, geopandas (vector files).
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy
import math
import rioxarray
import xarray

import geojson
import geopandas
import shapely
import shapely.geometry as shap_geom
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

import skimage
import skimage.exposure as skimage_exposure

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = '1'
import albumentations as A

import utils.file_utils as file_utils
import utils.plotting_utils as plotting_utils


def read_transl_geojson_AOI_coords_single(file_name, target_crs='None'):
    '''
    reads shapefile (!!!! but will only get first element !!!!
        --> thus assumes that shapefile contains only one polygon

    !!! NO MULTI-POLYGON ALLOWED !!!
    '''
    with open(file_name) as f:
        gj = geojson.load(f)

    coords = np.squeeze(gj['features'][0]['geometry']['coordinates'])
    crs = gj.crs['properties']['name'].split(':')[-1]  # output is string

    if not isinstance(target_crs, str):
        target_crs = str(target_crs)

    if target_crs == crs or target_crs == 'None':
        coords_out = coords.tolist()
    else:
        import utils.coord_utils as coord_utils
        os_E, os_N = coord_utils.coord_transformation(
            coords, in_syst=crs, out_syst=target_crs)

        coords_out = np.array([os_E, os_N]).T.tolist()

    poly = shap_geom.Polygon(coords_out)

    return coords_out, poly


def scale_AOI(inp_poly, resolution, cells, epsg, path=None,
              filename='AOI_scaled'):
    '''
    scale AOI to clip raster (scale factor is defined according to
    how many cells want to clip on each side)

    inp poly should be shapely polygon e.g. created with
    poly = shap_geom.Polygon(coords_out)

    cells is for how many cells should be shrinked in each direction on
    each side (will use resolution * cells * 2)

    resolution: e.g. resolution of raster image
    '''
    bounds = inp_poly.bounds
    xy_extent = np.diff(np.array(bounds).reshape(-1, 2), axis=0).ravel()
    fact = (xy_extent - (resolution * cells * 2))/xy_extent
    AOI_scaled = shapely.affinity.scale(
        inp_poly, xfact=fact[0], yfact=fact[1], zfact=1,
        origin='center')

    AOI_gdf = create_gdf_from_shape([AOI_scaled], epsg)
    if path is not None:
        AOI_gdf.to_file(os.path.join(path, filename + '.shp'))

    return AOI_gdf


def create_gdf_from_shape(shape_list, epsg):

    n_items = len(shape_list)
    gdf = geopandas.GeoDataFrame(
        geometry=shape_list, index=list(range(n_items)), crs=int(epsg))

    gdf['geometry'] = gdf['geometry'].to_crs({'proj':'cea'})
    gdf['area'] = gdf.area/10**6
    gdf = gdf.to_crs(int(epsg))

    return gdf


def read_to_xarray(file_name, mask_nan=False, chunk='auto'):
    rds = rioxarray.open_rasterio(
            file_name, masked=mask_nan, chunks=chunk)  # masked=True

    return rds


def create_np_array_from_xarray(img_inp, bands_sel=None, sortby=None):
    '''
    Converts an xarray image to a NumPy array. And adjust the dimension
    order from [bands, y, x] to [y, x, bands]
    (this order is selected since for glcm_cupy, OpenCV but also older
    versions of scikit image the dimensions are [height width, channels])

    Parameters:
    - img_inp: xarray DataArray representing the image.
    - bands_sel: List of band names or a single band name to select.
    - sortby: String representing the coordinate to sort by.
        If not specified, the original coordinate order is preserved.
        If 'y', the output array will have y coordinates in descending order.

    Returns:
    - img_sel: NumPy array with shape [y, x, bands] containing the
               selected bands and sorted coordinates.
    - bands_sel: List of the selected bands.
    '''

    # Copy the input image to avoid modifying the original
    img = img_inp.copy()
    if sortby is not None:
        # Sometimes this is required if create xarray Dataset
        # from DataFrame with xarray.Dataset.from_dataframe(df_out)
        # since from_dataframe sorts the coordinates in ascending order
        # but for images y should be in descending order
        img = img.sortby(sortby, ascending=(sortby != 'y'))

    if bands_sel is None:
        bands_sel = img.band.values.tolist()

    # Ensure that bands_sel is list to ensure correct dimensions
    if isinstance(bands_sel, str):
        bands_sel = [bands_sel]

    # extracts specific bands and changes order from [bands, y, x]
    # to [y, x, bands]
    # get position of bands (in case ther is also batch in front)
    band_index = img.dims.index('band')
    img_sel = np.moveaxis(
        img.sel(band=bands_sel).values, band_index, -1)

    return img_sel, bands_sel


def check_for_missing_fill_val(img, nodata_inp=None):
    """
    Check and set the nodata value for a given raster image.

    Parameters:
    img (rasterio.DatasetReader): The input raster image.
    nodata_inp (int or float, optional): The nodata value to set.
       If not provided, the function will attempt to infer a nodata value
       based on the image's data type and content.

    Returns:
    rasterio.DatasetReader: The input image with the nodata value set.

    Notes:
    - If `nodata_inp` is provided, it will be used as the nodata value.
        If rio.nodata was previousely set but is not consistent
        with nodata_inp, the function will exit with an error message.
    - If `nodata_inp` and rio.nodata are None and the image's data type
        is integer, the function will set the nodata value to 0.
    - If `nodata_inp` and rio.nodata are None and the image's data type
      is NOT integer, the function will check for the presence of
      NaN values in the image data.
      - If NaN values are present, the function will set the nodata value to NaN.
      - If NaN values are not present and no other nodata value has been
         set, the function will exit with an error message.
    """

    if nodata_inp is not None:
        if img.rio.nodata is None or img.rio.nodata == nodata_inp:
            img.rio.write_nodata(nodata_inp, inplace=True)
        else:
            sys.exit(
                '!!! there might be an error reading in data. \n'
                + 'the specified nodata differs from the predefined rio.nodata')
    elif (img.rio.nodata is None
        and img.dtype.name.find('int') > -1):
        img.rio.write_nodata(0, inplace=True)
    elif img.rio.nodata is None:
        if np.any(np.isnan(img.data)):
            img.rio.write_nodata(np.nan, inplace=True)
        else:
            sys.exit(
                '!!! there might be an error reading in data. \n'
                + 'nodata is not specified and nans are present in data')
    return img


def transform_resample(img, EPSG_TARGET=None, RESOLUTION_TARGET=0,
                        nodata_out=None, resampling_type='nearest'):
    """
    Reprojects and resamples a raster image if necessary.

    Parameters:
    img (rasterio.DatasetReader): The input raster image to be transformed.
    EPSG_TARGET (int, optional): The target EPSG code for the coordinate
        reference system. If not provided, the existing CRS is used.
    RESOLUTION_TARGET (int or float, optional): The target resolution in
        the units of the CRS. If set to 0, no resampling is performed.
        Default is 0.
    nodata_out (int or float, optional): The nodata value for the output
        image. If not provided, the nodata value from the input image is
        used.
    resampling_type (str, optional): The resampling method to use when
        changing the resolution. Options include 'nearest', 'bilinear',
        'cubic', 'cubic_spline', 'lanczos', 'average', and 'mode'.
        Default is 'nearest'.

    Returns:
    rasterio.DatasetReader: The transformed and resampled raster image.

    Notes:
    - The function reprojects the image to the target EPSG code if
        `EPSG_TARGET` is provided and different from the current CRS.
    - The function resamples the image to the target resolution if
        `RESOLUTION_TARGET` is greater than 0 and the target resolution
        differes from the current image resolution.
    - If the output image does not have a nodata value set, it will
        be set to `nodata_out`.
    """

    if nodata_out is None:
        nodata_out = img.rio.nodata

    # define target crs
    crs_orig = img.rio.crs.to_epsg()
    if EPSG_TARGET is None:
        crs_target = crs_orig
    else:
        crs_target = EPSG_TARGET

    # specify reprojection parameters
    inp_dict = {'nodata': nodata_out,
                'resampling': Resampling[resampling_type]}

    # add resolution parameter if resolution needs changing
    if RESOLUTION_TARGET > 0:
        res_curr = img.rio.resolution()
        if (abs(res_curr[0]) != RESOLUTION_TARGET
            or abs(res_curr[1]) != RESOLUTION_TARGET):
            inp_dict.update({'resolution': RESOLUTION_TARGET})

    # reproject if required
    if 'resolution' in inp_dict.keys() or crs_target != crs_orig:
        img = img.rio.reproject(
            "EPSG:" + str(crs_target), **inp_dict)

    # Set nodata value if not already set
    if img.rio.nodata is None:
        img.rio.write_nodata(nodata_out, inplace=True)

    return img


def clip_to_aoi(rds, AOI_COORDS, AOI_EPSG=None, from_disk=True,
                drop_na=True):
    '''
    aoi_coords: [[x1, y1], [x2, y2], ...]

    AOI_EPSG: epsg of the clip geom if none it is assumed to be the same
    as the dataset

    check option with geopndas:
    https://corteva.github.io/rioxarray/stable/examples/clip_geom.html

    from disk is much faster for large files !!! but might cause some
    small inconsistencies at border..
    https://corteva.github.io/rioxarray/stable/examples/clip_geom.html

    drop_na if drop values outise clip area. Otherwise will get same raster but
    with clipped values masked
    '''
    geometries = [
        {
            'type': 'Polygon',
            'coordinates': [AOI_COORDS]
        }]

    if AOI_EPSG is not None and AOI_EPSG != 'None':
        AOI_EPSG = int(AOI_EPSG)
    else:
        AOI_EPSG = rds.rio.crs.to_epsg()

    try:
        clipped = rds.rio.clip(
            geometries, crs="EPSG:" + str(AOI_EPSG),
            from_disk=from_disk, drop=drop_na)
    except:
        return None

    if rds.dtype != clipped.dtype:
        # this is becasue from_disk=True might changes nodata to nan (float)
        clipped = clipped.astype(rds.dtype)

    # !!!! encoding needs to be updated first!!!
    # otherwise nodata is not properly updated
    clipped.rio.update_encoding(rds.encoding, inplace=True)

    if clipped.rio.nodata is None:
        clipped.rio.write_nodata(rds.rio.nodata, inplace=True)

    return clipped


def clip_to_aoi_gdf(rds, AOI_gdf, from_disk=True, drop_na=True):
    '''
    clip to geopandas geodataframe
    if drop_na=True then data outsice clipping are will be dropped.
    if drop_na=False the a masked raster with the same grid extent will
    be returned
    '''
    try:
        clipped = rds.rio.clip(AOI_gdf.geometry.values, AOI_gdf.crs,
                               from_disk=from_disk, drop=drop_na)
    except:
        return None


    if rds.dtype != clipped.dtype:
        # this is becasue from_disk=True might changes nodata to nan (float)
        clipped = clipped.astype(rds.dtype)

    clipped.rio.update_encoding({'dtype': clipped.dtype}, inplace=True)

    # !!!! encoding needs to be updated first!!!
    # otherwise nodata is not properly updated
    clipped.rio.update_encoding(rds.encoding, inplace=True)
    clipped.rio.write_nodata(rds.rio.nodata, inplace=True)

    return clipped


def convert_img_to_dtype(img, dtype_out='uint8', nodata_out=None,
                         replace_zero_with_nan=False):

    # replace nans with 0
    if 'int' in dtype_out:
        img = replace_nan_with_zero(img)
    elif replace_zero_with_nan and 'float' in dtype_out:
        img = img.where(img != 0, other=np.nan)

    if nodata_out is None:
        if 'int' in dtype_out:
            nodata_out = 0
        elif 'float' in dtype_out:
            nodata_out = np.nan
    img_out = img.astype(dtype_out)
    new_encoding = img_out.encoding.copy()
    new_encoding.update({'rasterio_dtype': dtype_out})
    img_out.rio.update_encoding(
       new_encoding, inplace=True)

    img_out.attrs.update({'orig_type': dtype_out})

    img_out.rio.write_nodata(nodata_out, inplace=True)

    return img_out


def replace_nan_with_zero(img):

    mask_array = ~np.isnan(img.values)
    img = img.where(mask_array, other=0)

    return img


def set_fill_val_to_nan(img, how='all_invalid'):

    # get invalid values mask and add to image
    mask_invalid = get_invalid_mask(
        img, how=how)

    img = xarray_change_fill_val(
        img, mask_invalid,
        fill_val_new=np.nan)

    return img


def xarray_change_fill_val(im_inp, mask_array, fill_val_new=np.nan):
    '''
    change fill value to a different one

    all_invalid: sets only to fill_val_new if values in all bands are invalid
    '''

    out = im_inp.where(mask_array, other=fill_val_new)
    encoding_orig = im_inp.encoding
    out.rio.update_encoding(encoding_orig, inplace=True)
    out.rio.write_nodata(fill_val_new, inplace=True)

    return out


def get_invalid_mask(im_inp, how='all_invalid'):
    '''
    get mask with invalid values
    False = invalid
    True = ok
    '''
    fill_val = im_inp.rio.nodata
    if fill_val is None:
        print('!!! ' + im_inp.band.values[0]
              + ": rio.nodata not set can NOT get invalid mask")
        return np.ones(im_inp.shape).astype(bool)

    if how == 'all_invalid':
        if np.isnan(fill_val):
            mask_array = np.any(~np.isnan(im_inp.values), axis=0)
        else:
            mask_array = np.any(im_inp.values != fill_val, axis=0)
    elif how == 'any_invalid':
        if np.isnan(fill_val):
            mask_array = np.all(~np.isnan(im_inp.values), axis=0)
        else:
            mask_array = np.all(im_inp.values != fill_val, axis=0)
    elif how == 'single':
        #out = im_inp.copy()
        if np.isnan(fill_val):
            mask_array = ~np.isnan(im_inp.values)
        else:
            mask_array = im_inp.values != fill_val
    else:
        print('!!!! type of replacement undefined !!!!')

    return mask_array


def read_rename_according_long_name(
        file_name, mask_nan=False, chunk='auto'):

    rds = rioxarray.open_rasterio(
            file_name, masked=mask_nan, chunks=chunk)  # masked=True

    if 'long_name' in rds.attrs.keys():
        if isinstance(rds.attrs['long_name'], str):
            try:
                rds['band'] = ('band', eval(rds.attrs['long_name']))
            except:
                rds['band'] = ('band', [rds.attrs['long_name']])
        else:
            rds['band'] = ('band', list(rds.attrs['long_name']))

    return rds


def add_band_from_long_name(image):

    if 'long_name' in image.attrs.keys():
        long_name = image.attrs['long_name']
        if isinstance(long_name, str):
            try:
                long_name = [int(long_name)]
            except:
                long_name = [long_name]
        else:
            long_name = [x for x in long_name]

        image['band'] = (
            'band', long_name)
    return


def get_bands(file_path):
    img = read_rename_according_long_name(
        file_path, mask_nan=False, chunk=None)
    return img.band.values.tolist()


def get_add_img_attrs(img_orig, img_new):

    img_attrs = img_orig.attrs.copy()
    img_new.attrs.update(img_attrs)
    # make sure that long name is correct
    write_long_name(img_new)
    img_new.rio.write_nodata(img_orig.rio.nodata, inplace=True)
    # check crs
    if img_new.rio.crs is None:
        img_new.rio.write_crs(img_orig.rio.crs, inplace=True)

    return


def write_long_name(img):
    # make sure that long name is correct
    bands_long = img.band.values.tolist()
    if len(bands_long) == 1:
        img.attrs['long_name'] = bands_long[0]
    else:
        img.attrs['long_name'] = bands_long
    return


def save_to_geotiff_options(img, path_export, file_name_pefix,
                            single=True):

    if single:
        # --- save bands in separate files
        # get avail bands
        bands = img.band.values.tolist()

        for i in bands:
            out = img.sel(band=i)

            # save file
            filename_out = file_name_pefix + '_' + i + '.tif'
            save_to_geotiff(
                out, path_export, filename_out, add_crs=False)
    else:
        # -- save all bands in same files
        save_to_geotiff(
            img, path_export, file_name_pefix + '.tif', add_crs=False)

    return


def save_to_geotiff(image, path, file_name, suffix='', add_crs=True):

    add_long_name_attrs(image)

    file_name = file_utils.remove_file_extension(file_name)
    if add_crs:
        current_epsg = image.rio.crs.to_epsg()  #.data['init'].split(':')[-1]
        file_name = (file_name + '_' + str(current_epsg) + suffix + '.tif')
    else:
        file_name = (file_name + suffix + '.tif')
    out_path = os.path.join(path, file_name)

    image.rio.to_raster(out_path, driver='GTiff')
    return out_path


def add_long_name_attrs(image):
    long_name_list = image.band.values.tolist()
    if isinstance(long_name_list, list):
        long_name_list = [str(x) for x in long_name_list]
    if len(long_name_list) == 1 and not isinstance(long_name_list, str):
        long_name_list = str(long_name_list[0])
    image.attrs.update({'long_name': long_name_list})
    return


def create_multiband_xarray_from_np_array(
        img_orig, data_np_array, band_name_lst,
        attrs=None, nodata=None):
    '''
    !!!!
    numpy arrays for imagesckit have dimension [z, y, band]
    xarray have dimension [band, y, x]

    !!! it is better to set nodata explicitly
    '''

    # change axis order because xarray is [band, y, x]
    data_inp = np.moveaxis(data_np_array, -1, 0).copy()
    x_coords = img_orig.coords['x'].values
    y_coords = img_orig.coords['y'].values
    bands = band_name_lst

    out = xarray.DataArray(
        data_inp, coords=[bands, y_coords, x_coords],
        dims=["band", "y", "x"])

    if attrs is None:
        attrs = img_orig.attrs.copy()
    out.attrs.update(attrs)

    out = out.rio.write_crs(img_orig.rio.crs)
    if nodata is None:
        # need to update both encoding and nodata
        # since if mask_nan was used when reading in data then would
        # otherwise get an error if try to export an array with nan
        # nodata as integer (however it is safer to explicitly specify
        # the nodata value)
        out.rio.update_encoding(
            img_orig.encoding.copy(), inplace=True)
        out.encoding['rasterio_dtype'] = out.dtype.name
        out.encoding['dtype'] = out.dtype.name

        nodata = img_orig.rio.nodata

    out.rio.write_nodata(nodata, inplace=True)

    return out


def scale_convert_bit_np(
        img_np, nodata_inp,
        nodata_out=0,
        to_bit_num=2**4, how='scale', perc=None, min_max=None,
        std_fact=None, gamma=None, log_adj=None):
    '''
    for using this function, it is best to read image with mask_nan
    '''
    # convert no data to no minus value since these will not be
    # taken into account when scaling
    # img_as_ubyte scales data first

    if ~np.isnan(nodata_inp):
        img_np = np.where(
            img_np==nodata_inp, np.nan,
            img_np)

    if perc is not None:
        val_min = int(
            np.nanpercentile(
                img_np, perc[0]))  # - 1
        val_max = int(
            np.nanpercentile(
                img_np, perc[1]))
    elif std_fact is not None:
        mean = np.nanmean(img_np)
        std = np.nanstd(img_np)
        val_min = mean - abs(std*std_fact)
        val_max = mean + abs(std*std_fact)
    elif min_max is not None:
        val_min = min_max[0]
        val_max = min_max[1]
    else:
        val_min = np.nanmin(img_np)  # - 1
        val_max = np.nanmax(img_np)

    mask = np.isnan(img_np)
    img_np[mask] = val_min
    #img_np = img_np.astype(int)

    if how == 'equalize':
        img_np = skimage_exposure.equalize_hist(
            img_np)
        val_min = 0
        val_max = 1

    elif how != 'scale':
        try:
            img_np = skimage_exposure.equalize_adapthist(
                img_np, clip_limit=0.0, nbins=256)
        except:
            print('!!! equalize_adapthist did not work. '
                    + 'Zero division error !!!')
        val_min = 0
        val_max = 1

    if gamma is not None:
        # value below 1 makes image brighter, above 1 darker
        img_np = skimage_exposure.rescale_intensity(
            img_np, in_range=(val_min, val_max),
            out_range=(0, 1))#.astype('uint8')  # uint8 is minimum availbale
        img_np = skimage_exposure.adjust_gamma(
            img_np, gamma)
        val_min = 0
        val_max = 1
    if log_adj is not None:
        # default value is one. Makes img brighter
        img_np = skimage_exposure.rescale_intensity(
            img_np, in_range=(val_min, val_max),
            out_range=(0, 1))
        img_np = skimage_exposure.adjust_log(
            img_np, log_adj)
        val_min = 0
        val_max = 1

    img_np = skimage_exposure.rescale_intensity(
        img_np, in_range=(val_min, val_max),
        out_range=(1, to_bit_num-1))

    img_np[mask] = 0
    # make sure that dtype is correct
    img_np = check_and_change_type_to_uint(
        img_np, to_bit_num)

    # set other nodata values (e.g. if used masked nan can change
    # this to nan, later when creating a rasterio img can
    # specify the encoding according to the original file)
    img_np[mask] = nodata_out

    return img_np, [val_min, val_max, 1, to_bit_num - 1]


def check_and_change_type_to_uint(img_np, to_bit_num):
    # Do not check for floast here as skimage.exposure.rescale_intensity()
    # can lead to float numbers see:
    # https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    # if 'float' in img_np.dtype.name:
    #    sys.exit(
    #        '!!! Error: conversion to uint: there are float numbers '
    #        + 'present in the array !!!')

    arr_min = np.nanmin(img_np)
    arr_max = np.nanmax(img_np)
    if arr_min < 0 or arr_max >= to_bit_num:
        sys.exit('!!! Error: conversion to uint: values are outside '
                 + f'uint range !!! arr min: {arr_min}'
                 + f'arr max: {arr_max}')

    if to_bit_num <= 2**8:
        out = img_np.astype('uint8')
    else:
        out = img_np.astype('uint16')
    return out


def apply_hist_match(inp_img, ref_img, nodata_val=None,
                     path_exp=None, fig_suffix=None,
                     ax_hist_match_plt=None,
                     file_prefix=''):
    """
    Apply histogram matching to an input image based on a reference image.

    Parameters
    ----------
    inp_img : xarray DataArray
        The input image to be adjusted.
    ref_img : xarray DataArray
        The reference image.
    nodata_val : scalar, optional
        The nodata value to use for the input and reference images. If not
        provided, the nodata value will be taken from the input and reference
        images' metadata.

    Returns
    -------
    matched_img : xarray DataArray
        The histogram-matched input image.
    val_range : list
        A list containing the minimum and maximum values of the matched
        image (ONLY the last band), as well as two NaN values
        (for consistency with other functions).

    Notes
    -----
    This function assumes that the input and reference images have the same
    number of bands.
    """
    dtype_orig = inp_img.dtype

    # check for imagery consistency
    if dtype_orig != ref_img.dtype:
        print('!!! Images have different bit range or dtpye. Scikit hist match does not work properly !!!')

    if nodata_val is None:
        img_nodata_val_orig = inp_img.rio.nodata
        ref_nodata_val_orig = ref_img.rio.nodata
    else:
        img_nodata_val_orig = nodata_val
        ref_nodata_val_orig = nodata_val

    # get numpy array
    img_np =  inp_img.values.copy()
    ref_np =  ref_img.values.copy()

    # loop through each band
    for e, i in enumerate(inp_img.band.values):
        # get mask
        img_mask = get_nodata_mask(img_np[e, :, :], img_nodata_val_orig)
        ref_mask = get_nodata_mask(ref_np[e, :, :], ref_nodata_val_orig)

        # get hist adjust
        # (nodata values are ignored)
        img_np_out = histogram_matching_with_nan_mask_np(
            img_np[e, :, :], ref_np[e, :, :], img_mask, ref_mask,
            img_nodata_val_orig, path_exp=path_exp, fig_suffix=fig_suffix,
            ax_hist_match_plt=ax_hist_match_plt, file_prefix=file_prefix)

        # if need to make sure that dtype is correct
        # if img_np_out.dtype != dtype_orig:
        #     img_np_out = img_np_out.astype(dtype_orig)
        val_min = np.nanmin(img_np_out[~img_mask])
        val_max = np.nanmax(img_np_out[~img_mask])

        # Replace the values from the respective band
        inp_img.loc[{'band': i}] = img_np_out

    # !!! val_min and val_max are only values form last band !!!
    return inp_img, [val_min, val_max, np.nan, np.nan]


def histogram_matching_with_nan_mask_np(
        source_arr, ref_arr, source_mask, ref_mask,
        nodata_val_out, path_exp=None, fig_suffix=None,
        ax_hist_match_plt=None, file_prefix=''):
    """
    Apply histogram matching, ignoring nodata values.

    This function takes two input arrays, a source array and a reference
    array, and their corresponding nodata masks.
    It applies histogram matching including the valid pixels
    (i.e., those that are not nodata) and returns the matched array.

    Parameters
    ----------
    source_arr : numpy array 2D
        The source array to be matched.
    ref_arr : numpy array 2D
        The reference array.
    source_mask : numpy array
        A boolean mask indicating nodata values in the source array.
    ref_mask : numpy array
        A boolean mask indicating nodata values in the reference array.
    nodata_val_out : scalar
        The value to use for nodata pixels in the output array.

    Returns
    -------
    matched_image : numpy array
        The histogram-matched array.

    Notes
    -----
    This function only works for single-channel data.
    """

    # Only use non nodata values for histogram matching
    source_valid = source_arr[~source_mask]
    reference_valid = ref_arr[~ref_mask]

    # Perform histogram matching using only valid (non-NaN) pixels
    matched_valid = skimage_exposure.match_histograms(
        np.atleast_3d(source_valid), np.atleast_3d(reference_valid),
        channel_axis=-1)
    #if source_valid.dtype.name.find('int') > -1:
    #    matched_valid = matched_valid.round()
    #matched_valid = matched_valid.astype(source_valid.dtype)

    if path_exp is not None:
        plotting_utils.plot_hist_match_curves(
            np.atleast_3d(source_valid), np.atleast_3d(reference_valid),
            np.atleast_3d(matched_valid),
            path_exp, fig_suffix, axes=ax_hist_match_plt,
            file_prefix=file_prefix)

    # Create an empty array to store the final matched image
    matched_image = np.full_like(source_arr, nodata_val_out)

    # Insert the matched pixels back into the image, ignoring NaN positions
    matched_image[~source_mask] = matched_valid.squeeze()

    return matched_image


def augmentation_hist_match(ref_img, img, hist_blend=0.5,
                            to_bit_num=2**8, nodata_val=None,
                            path_exp=None, fig_suffix=None,
                            ax_hist_match_plt=None,
                            file_prefix='', set_mask_to=255,
                            discard_nodata=False):
    '''
    !!! discard_nodata is extremely slow: unclear why --> do not use
     but raterher use the hist match apply_hist_match() using scikit if
     want to discard all nodata values !!!
    '''
    if nodata_val is None:
        img_nodata_val = img.rio.nodata
        ref_nodata_val = ref_img.rio.nodata
    else:
        img_nodata_val = nodata_val
        ref_nodata_val = nodata_val

    # use -2 here since add 1 further below
    # this ensures that no not nan values are zero at end
    transform_to_int = A.FromFloat(
        dtype=f'uint{int(math.log(to_bit_num, 2))}',
        max_value=to_bit_num - 2, p=1)

    # loop through each band
    for e, i in enumerate(img.band.values):
        ref_np = np.moveaxis(ref_img.values[[e], :, :], 0, -1).copy()
        img_np = np.moveaxis(img.values[[e], :, :], 0, -1).copy()

        # get mask
        ref_mask = get_nodata_mask(ref_np, ref_nodata_val)
        img_mask = get_nodata_mask(img_np, img_nodata_val)

        # conversion since for histmatch possible input types
        # are uint8, float32 use float 32 to also allw uin16 data
        # do not set nodata values to nan since it creates an errorfurther
        # below
        # !!!! with albumentation having nans in the reference image can cause
        # issues since the nans are converted to uint8 (thus nan would
        # just be converted to zeros but ONLY in the reference) !!!
        ref_np_float_val = convert_int_np_arr_to_float32(
            ref_np, ref_mask, set_mask_to=set_mask_to,
            bit_out=to_bit_num-1)
        img_np_float_val = convert_int_np_arr_to_float32(
            img_np, img_mask, set_mask_to=set_mask_to,
            bit_out=to_bit_num-1)

        # Only use non nodata values for histogram matching
        if discard_nodata:
            img_np_float_val_orig = img_np_float_val.copy()

            # Note use [np.newaxis, np.newaxis, :] this is for speed up to
            # have last dimension the largest
            ref_np_float_val = ref_np_float_val[~ref_mask][:, np.newaxis, np.newaxis]
            img_np_float_val = img_np_float_val[~img_mask][:, np.newaxis, np.newaxis]
            # Create an empty array to store the final matched image
            matched_image = np.full_like(img_np_float_val_orig, img_nodata_val)

        # possible input types for hist mach are uint8, float32
        hist_transform = A.HistogramMatching(
                reference_images=[ref_np_float_val],
                read_fn = lambda x: x,
                p=1, blend_ratio=(hist_blend, hist_blend))

        hist_transformed0 = hist_transform(
            image=img_np_float_val)["image"]

        if path_exp is not None:
            if discard_nodata:
                plotting_utils.plot_hist_match_curves(
                    img_np_float_val,
                    ref_np_float_val,
                    hist_transformed0,
                    path_exp, fig_suffix, axes=ax_hist_match_plt,
                    file_prefix=file_prefix)
            else:
                # if didnt mask nan mask them in the plot since they are
                # clipped at the end anyway
                # Note last dimension is treated as "band". Thus should there
                # use [np.newaxis, :, np.newaxis]
                plotting_utils.plot_hist_match_curves(
                    img_np_float_val[~img_mask][np.newaxis, :, np.newaxis],
                    ref_np_float_val[~ref_mask][np.newaxis, :,np.newaxis],
                    hist_transformed0[~img_mask][np.newaxis, :, np.newaxis],
                    path_exp, fig_suffix, axes=ax_hist_match_plt,
                    file_prefix=file_prefix)

        # Insert the matched pixels back into the image, ignoring NaN positions
        if discard_nodata:
            matched_image[~img_mask] = hist_transformed0.squeeze()
            hist_transformed = matched_image
        else:
            hist_transformed = hist_transformed0

        # change np.nans back to 0 since get error message otherwise
        hist_transformed[img_mask] = img_nodata_val
        # revert back to to int

        # !!! add plus 1 to go back to range 1 to 255 or 2**16 - 1
        hist_transformed = transform_to_int(image=hist_transformed)['image'] + 1
        # use mask to mask nodata values
        hist_transformed[img_mask] = img_nodata_val

        val_min = np.nanmin(hist_transformed[~img_mask])
        val_max = np.nanmax(hist_transformed[~img_mask])

        hist_img_val = np.moveaxis(hist_transformed, -1, 0)
        img.loc[{'band': i}] = hist_img_val.squeeze()

    return img, [val_min, val_max, 1, to_bit_num]


def convert_int_np_arr_to_float32(arr_int_inp, nodata_mask,
                                  set_mask_to=np.nan,
                                  bit_out=None):
    '''
    Brings the uint type data to the range 0-1 (and float32)

    max_val must corrspond to the uint type. Thus must e.g. be 255 for uint8
        with 0 corresponding to nan

    set_mask_to: if 'NAN' then np.nan is set at mask
                 if 'mean' then np.nanmean is set at mask
    '''
    if bit_out is None:
        max_val = np.iinfo(arr_int_inp.dtype).max
    else:
        max_val = bit_out

    arr_int = arr_int_inp.copy()

    if arr_int.max() > max_val:
        arr_int = skimage.exposure.rescale_intensity(
                arr_int, in_range=(1, arr_int.max()),
                out_range=(1, max_val))

    transform_to_float = A.ToFloat(max_value=max_val, p=1)
    arr_float_val = transform_to_float(image=arr_int)['image']

    # replace nodata values (i.e. zeros) with nans
    if set_mask_to == 'mean':
        mean_inp = np.nanmean(arr_float_val[~nodata_mask])
        arr_float_val[nodata_mask] = mean_inp
    elif isinstance(set_mask_to, (int, float)):
        arr_float_val[nodata_mask] = set_mask_to
    else:
        sys.exit('input option for nodata replacement is not implemented')

    return arr_float_val


def convert_txt_to_gdf(df, crs_num, geom_col='geometry'):

    s = geopandas.GeoSeries.from_wkt(df[geom_col])
    gdf = geopandas.GeoDataFrame(data=df, geometry=s, crs=f'EPSG:{crs_num}')
    return gdf


def add_intersection_area(gdf, shapely_poly, shapely_crs):
    '''
    add intersection area to gdf
    for the correct area calculation need to translate coords to cea
    '''

    geom_t = gdf['geometry'].to_crs({'proj':'cea'})
    # create gdf os aoi
    aoi_gdf = geopandas.GeoDataFrame(
        geometry=[shapely_poly], crs=shapely_crs)
    aoi_t = aoi_gdf.to_crs({'proj':'cea'})
    aoi_area = aoi_t.area/10**6

    area_out = geom_t.intersection(aoi_t['geometry'][0]).area/10**6

    gdf['aoi_intersect_area'] = area_out
    gdf['aoi_intersect_perc'] = area_out/aoi_area.values[0]*100

    return


def get_img_log(img, img_min_inp=None, use_min_shift=False,
                img_shift=0.01):
    '''
    here shift values with min to allow log calculation as log of
    negative values is nan
    '''
    img = img.chunk(dict(x=-1, y=-1))

    if not use_min_shift:
        img_shift_val = img_shift
    elif img_min_inp is None and use_min_shift:
        img_min = img.min(dim=('x', 'y'), skipna=True).compute()
        img_shift_val = abs(img_min) + 0.1
    else:
        img_shift_val = abs(img_min_inp) + 0.1

    # shift to avoid inf values due to zeros
    img_shifted = img + img_shift_val

    img_log = np.log(img_shifted)
    get_add_img_attrs(img, img_log)
    return img_log


def get_nodata_mask(np_arr, nodata_val):
    """
    Create a boolean mask indicating nodata values in an array.

    Parameters
    ----------
    np_arr : numpy array
        The input array.
    nodata_val : scalar
        The nodata value to use.

    Returns
    -------
    nodata_mask : numpy array
        A boolean mask indicating nodata values in the input array.
    """
    if np.isnan(nodata_val):
        nodata_mask = np.isnan(np_arr)
    else:
        nodata_mask = np_arr == nodata_val
    return nodata_mask


def concat_arrays_and_calc(
        dict_list, calc='var', bands_suffix='', reorder_y=True,
        test_coords=False, concat_coords_inp='minimal',
        logging_inp=None, dict_prefix=''):
    '''
    Concatenates several xarrays with the same dimensions and calculates
    statistics across bands.

    Parameters:
    dict_list (dict): A dictionary where keys are identifiers and
        values are xarrays with the same dimensions.
    calc (str or list of str): The type(s) of statistical calculation
        to perform. Types include 'var' (variance),
        'std' (standard deviation), and 'sum' (sum).
        If a list is provided, multiple calculations will be performed.
    bands_suffix (str): A suffix to append to the band names in
        the output. This is useful for distinguishing the type of
        calculation performed.
    reorder_y (bool): If True, the output xarrays will be sorted by the
        'y' dimension in descending order.
    test_coords (bool): If True, the function will check if all input
        xarrays have the same coordinates.
    concat_coords_inp (str): Determines how coordinates should be
        handled during concatenation (xarray option).
        Options include 'minimal' (default), 'all', and 'drop'.
    logging_inp (logging.Logger): A logging object to use for
        logging information and warnings.
    dict_prefix (str): A prefix to apply to the output dictionary keys.

    Returns:
    tuple: A tuple containing the output dictionary with the calculated
    statistics and a list of the new band names.

    '''
    if not isinstance(calc, list):
        calc = [calc]

    # check if coordinates of the arrays are identical
    if test_coords:
        text_out = check_if_coords_equal(dict_list)
        if logging_inp is not None:
            logging_inp.info(text_out)

    # concatenate all arrays add dict keys as with name "new" as new dimension
    new_arr = xarray.concat(
        dict_list.values(), dim=pd.Index(dict_list.keys(), name='new'),
        coords=concat_coords_inp)

    band_names = new_arr.band.values.tolist()

    # calculate stats
    out = {}
    for i_calc in calc:
        i_key = dict_prefix + '_' + i_calc
        if i_calc == 'var':
            out[i_key] = new_arr.var(dim='new', keep_attrs=True)
        elif i_calc == 'std':
            out[i_key] = new_arr.std(dim='new', keep_attrs=True)
        elif i_calc == 'sum':
            out[i_key] = new_arr.sum(dim='new', keep_attrs=True)

        # rename band names
        new_names = [x + bands_suffix for x in band_names]
        out[i_key]['band'] = ('band', new_names)

        # if require reorder y cooridnate
        if reorder_y:
            out[i_key] = out[i_key].sortby('y', ascending=False)

    return out, new_names


def check_if_coords_equal(da_dict, coords_check=None):
    '''
    coord_lst is list of coords to compare
    e.g. ['month'] or ['latitude', 'longitude']
    '''
    text_out = []
    keys_lst = list(da_dict.keys())
    for i_key, ii_key in zip(keys_lst[:-1], keys_lst[1:]):
        if coords_check is None:
            coords_check = da_dict[i_key].coords.dims
        for i in coords_check:
            if np.all(da_dict[i_key][i].values == da_dict[ii_key][i].values):
                # print(i + ': ' + i_key + ' & ' + ii_key + ': are equal')
                text_out.append(
                    i + ': ' + i_key + ' & ' + ii_key + ': are equal')
            else:
                # print(i + ': ' + i_key + ' & ' + ii_key + ': are NOT the same !!!!!!')
                text_out.append(
                    i + ': ' + i_key + ' & ' + ii_key + ': are NOT the same !!!!!!')
    return '\n'.join(text_out)


def check_if_coords_equal_lst(da_lst, coords_check=None):
    '''
    '''
    text_out = []
    for i_da0, i_da1 in zip(da_lst[:-1], da_lst[1:]):
        if coords_check is None:
            coords_check = i_da0.coords.dims
        for i in coords_check:
            if np.all(i_da0[i].values == i_da1[i].values):
                text_out.append(f'{i} are the same')
            else:
                # print(i + ': ' + i_key + ' & ' + ii_key + ': are NOT the same !!!!!!')
                text_out.append(f'{i} are not the same')
    return '\n'.join(text_out)


def xarray_interp_coords(coord_array):
    '''
    Interpolate coords which are nan
    nans in coordinates can be the result of
    xarray.DataArray.coarsen().construct()

    !!! input should be view e.g. from:
        class_sub_img.x.values
        then new coords are direclty applied onto the xarray
        thus need no return.
    '''
    # get idx for nan vals
    is_na = np.where(np.isnan(coord_array))[0]
    if is_na.shape[0] == 0:
        return
    # get idx for not nan vals
    not_na = np.where(~np.isnan(coord_array))[0]

    # create interp function and interp
    f = scipy.interpolate.interp1d(
        not_na, coord_array[not_na], fill_value = "extrapolate")
    new_val = f(is_na)

    # assign to coords, as coord_array is niew this is directly applied
    # on to xarray
    coord_array[is_na] = new_val

    return


def create_xarray(x_coords, y_coords, data, band_names, crs_data,
                 reverse=False):

    xarr_out = xarray.DataArray(
        data, coords=[band_names, y_coords, x_coords],
        dims=['band', 'y', 'x'])

    xarr_out = xarr_out.rio.write_crs(crs_data)

    return xarr_out


def create_save_xarray(
        img_orig, np_array, band_name_lst,
        path, file_name_out, attrs=None, nodata=None,
        width_out=None, height_out=None):
    '''
    !!!!
    xarray have dimension [band, y, x]
    '''
    x_coords = img_orig.coords['x'].values
    y_coords = img_orig.coords['y'].values
    bands = band_name_lst

    out = xarray.DataArray(
        np_array, coords=[bands, y_coords, x_coords],
        dims=["band", "y", "x"])

    if attrs is None:
        attrs = img_orig.attrs
    else:
        out.attrs.update(attrs)

    out = out.rio.write_crs(img_orig.rio.crs)
    if nodata is None:
        nodata = img_orig.rio.nodata
    out.rio.write_nodata(nodata, inplace=True)

    if width_out is not None and height_out is not None:
        img_shape = out.shape
        out, x_crop_id, y_crop_id = center_crop_img(
            out, img_shape[1], img_shape[2],
            width_out, height_out)

    # save to file
    out_path = os.path.join(path, file_name_out)
    out.rio.to_raster(out_path, driver='GTiff')

    return out_path


def create_save_xarray_from_numpy(
        np_array, x_coords, y_coords, band_name_lst,
        path, file_name_out, EPSG_int, attrs=None, nodata=None):
    '''
    !!!!
    xarray have dimension [band, y, x]
    '''
    bands = band_name_lst

    out = xarray.DataArray(
        np_array, coords=[bands, y_coords, x_coords],
        dims=["band", "y", "x"])

    if attrs is not None:
        out.attrs.update(attrs)
    write_long_name(out)

    out.rio.write_crs(EPSG_int, inplace=True)
    out.rio.write_nodata(nodata, inplace=True)

    # save to file
    out_path = os.path.join(path, file_name_out)
    out.rio.to_raster(out_path, driver='GTiff')

    return out_path


def get_center_crop_start_end(n_in, n_out):

    mid = int(n_in/2)
    x_start, x_end = int(mid - np.floor(n_out/2)), int(mid + np.ceil(n_out/2))

    return x_start, x_end


def center_crop_img(
        img, img_x_shape, img_y_shape, x_out_shape, y_out_shape):

    x_start, x_end = get_center_crop_start_end(img_x_shape, x_out_shape)
    y_start, y_end = get_center_crop_start_end(img_y_shape, y_out_shape)

    img_out = img.isel(x=slice(x_start,x_end),
                       y=slice(y_start,y_end)).copy()

    return img_out, [x_start,x_end], [y_start, y_end]


def center_crop_numpy(
        np_arr, np_arr_x_shape, np_arr_y_shape,
        x_out_shape, y_out_shape):

    x_start, x_end = get_center_crop_start_end(
        np_arr_x_shape, x_out_shape)
    y_start, y_end = get_center_crop_start_end(
        np_arr_y_shape, y_out_shape)

    img_out = np_arr[:, y_start:y_end, x_start:x_end].copy()

    return img_out, [x_start,x_end], [y_start, y_end]


def tif_to_cog_rio(file_path, resave_rio=False):
    '''
    use resave_rio if want to resave the raster reading and adding the
    nodata value

    !!! Note: with this method the nodata value is not written to file
    even if specified (thus maybe better use tif_to_cog_rasterio())
    '''
    img = read_to_xarray(file_path)
    img = check_for_missing_fill_val(img)
    out_path = file_path.split('.')[0] + '_cog.tif'

    img.rio.to_raster(raster_path=out_path, driver="COG",
                      nodata=img.rio.nodata)

    if resave_rio:
        img.rio.to_raster(raster_path=file_path, write_nodata=True)
    img.close()

    return


def tif_to_cog_rasterio(file_path):
    '''Save tif to cog via rasterio'''
    # Open the source GeoTIFF
    with rasterio.open(file_path, 'r') as src:
        # Define COG creation options
        cog_profile = src.profile.copy()

        blocksize = 256
        padded_data, pad_width, pad_height = pad_raster_to_multiple(
            src, blocksize)
        nodata_val = src.nodata
        cog_profile.update({
            'driver': 'GTiff',
            'compress': 'DEFLATE',
            'blockxsize': blocksize,  # Tile size
            'blockysize': blocksize,  # Tile size
            'tiled': True,
            'width': src.width + pad_width,
            'height': src.height + pad_height,
            'nodata': nodata_val
            })

        out_path = file_path.split('.')[0] + '_cog.tif'
        # Write the COG file
        with rasterio.open(out_path, 'w', **cog_profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(padded_data[i-1], i)
            dst.descriptions = src.descriptions
            dst.update_tags(**src.tags().copy())
    return


def pad_raster_to_multiple(src, blocksize):
    width, height = src.width, src.height
    pad_width = (blocksize - width % blocksize) % blocksize
    pad_height = (blocksize - height % blocksize) % blocksize

    # Pad the raster data
    padded_data = []
    for i in range(1, src.count + 1):
        band = src.read(i)
        padded_band = np.pad(
            band,
            ((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=src.nodata
        )
        padded_data.append(padded_band)

    return padded_data, pad_width, pad_height


def read_clip_to_aoi(AOI_path, file_path, EPSG_AOI=None, chunk='auto',
                     path_out=None, to_cog=True,
                     from_disk=True, drop_na=True, img=None):

    if img is None:
        img = read_rename_according_long_name(
            file_path, mask_nan=False, chunk=chunk)

    if EPSG_AOI is None:
        EPSG_AOI = img.rio.crs.to_epsg()

    AOI_coords, AOI_poly = read_transl_geojson_AOI_coords_single(
        AOI_path, target_crs=EPSG_AOI)

    img_clipped = clip_to_aoi(img, AOI_coords, AOI_EPSG=EPSG_AOI,
                              from_disk=from_disk, drop_na=drop_na)

    if path_out is None:
        path_out = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
    file_path_out = save_to_geotiff(
        img_clipped, path_out, file_name,
        suffix='_clip', add_crs=False)

    if to_cog:
        tif_to_cog_rasterio(file_path_out)

        # other option but his gives an error due to different configuration
        # however, cog is created anyway
        # file_path_out = file_path_out.split('.')[0] + '_cog.tif'
        # img_clipped.rio.to_raster(
        #    raster_path=file_path_out, driver="COG", write_nodata=True)

    del img_clipped
    return


def remove_small_shapes_from_raster(
        img, min_px, val_band_name, EPSG, plot_hist=False):

    resol = np.abs(img.rio.resolution())
    pix_area = np.multiply(*resol)

    min_area = (pix_area*min_px)  #/1000000

    gdf = create_shapefile_from_raster(
        img, val_band_name, EPSG, unit='m')

    # get index of ones that do not want to keep
    rem_ind = gdf.query('area_m < @min_area').index
    gdf.loc[rem_ind, val_band_name] = 99

    img_rem = create_raster_from_gdf(
        img, gdf, val_band_name, fillval=0)

    # set nodata for interpolation
    img_rem.rio.write_nodata(99, inplace=True)
    img_rem = img_rem.rio.interpolate_na(method='nearest')

    # reset nodata value
    img_rem.rio.set_nodata(0, inplace=True)

    # add ii=nitial band name
    img_rem = convert_img_to_dtype(
        img_rem, dtype_out='uint8', nodata_out=0,
        replace_zero_with_nan=False)

    if plot_hist:
        gdf.groupby(val_band_name)['area_m'].hist(bins=50)

    return img_rem


def create_raster_from_gdf(inp_arr, gdf, raster_val_name, fillval=0):
    '''
    inp_arr is xarray from which will take geometry

    gdf is geopandas dataframe with shapely geometries to rasterize

    raster_val_name name of column with the values which should be used
    for rasterization

    fill_val: is background value which will be used where have no shape
    '''

    inp = inp_arr.copy()
    # set inp to fill val (otherwise values fro inp will be used)
    inp = inp.where(False, fillval)
    # !!! since out is provided (out=inp) inp is adapted directly
    rasterio.features.rasterize(
        list(zip(gdf.geometry, gdf.loc[:, raster_val_name])),
        fill=fillval, out=inp, transform=inp.rio.transform(recalc=False))

    return inp


def create_shapefile_from_raster(
        inp_xarr, value_name, EPSG_TARGET, connectivity_px=4, unit='km',
        dtype_out=None, replace_zero_with_nan=False):
    '''
    inp_arr is xarray

    shapefile are created according to band values
    value_name can be band name

    '''
    if inp_xarr.dtype.name.find('int') > -1:
        fct = lambda x: int(x)
    else:
        fct = lambda x: x

    if dtype_out is not None:
        inp_xarr = convert_img_to_dtype(
            inp_xarr.copy(), dtype_out=dtype_out, nodata_out=None,
            replace_zero_with_nan=replace_zero_with_nan)
        #inp_xarr = inp_xarr.astype(dtype_out)

    if inp_xarr.dtype.name == 'float64':
        # float64 seems not to be allowed in the rasterio shape function
        inp_xarr = inp_xarr.astype('float32')
    elif inp_xarr.dtype.name.find('int64') > -1:
        inp_xarr = inp_xarr.astype('int32')


    # create to generator of GeoJSON features
    results = (
        {'properties': {value_name: fct(v)}, 'geometry': s}
        for i, (s, v) in enumerate(
            rasterio.features.shapes(inp_xarr, mask=None,
                                     connectivity=connectivity_px,
                    transform=inp_xarr.rio.transform(recalc=False))))
    geoms = list(results)

    gdf  = geopandas.GeoDataFrame.from_features(
        geoms, crs=int(EPSG_TARGET))
    # add area
    add_area_to_gdf(gdf, EPSG_TARGET, unit=unit)
    #geo_utils.create_gdf_from_shape(geoms, EPSG_TARGET)

    return gdf


def add_area_to_gdf(gdf, epsg, unit='km'):
    '''
    calculation of area with shapely is wrong. It needs to be projected
    to cylindrical equal area first
    # --> see: https://gis.stackexchange.com/questions/218450/getting-polygon-areas-using-geopandas
    therefore use egopandas
    '''

    gdf['geometry'] = gdf['geometry'].to_crs({'proj':'cea'})
    if unit == 'km':
        gdf['area_km'] = gdf.area/10**6
    else:
        gdf['area_m'] = gdf.area
    gdf['geometry'] = gdf['geometry'].to_crs(int(epsg))

    return


def filter_shapes_according_to_intersection(
        gdf_inp, gdf_ref, FILTER_TYPE='intersection',
        PATH_EXPORT=None, EXPORT_PREFIX=None):
    '''
    filter_type options are:
        "intersection": this selects shapes that intersect the required
            area but do not only touch its border
        "within": selects only shapes that are fully within the required
            raster area

    gdf_inp: geodataframe with shapes to be filtered
    gdf_ref: geodataframe
        shapes based on whose intersection the gdf_inp shapes are filtered
    RASTER_BAND_NAME: str or int
        band name of raster based on whose intersection the
        gdf_inp shapes are filtered
    RASTER_VAL: int or float
        values of raster baesed on whose intersection the
        gdf_inp shapes are filtered

    '''
    # merge reference to multipolygon
    gdf_ref = gdf_ref.union_all()

    # filter input shapes (gdf_inp) according to intersection/within
    if FILTER_TYPE == 'intersection':
        test_select = (
            gdf_inp.geometry.intersects(gdf_ref)
            & ~gdf_inp.geometry.touches(gdf_ref))
    elif FILTER_TYPE == 'within':
        test_select = (
            gdf_inp.geometry.within(gdf_ref))
    gdf_out = gdf_inp.loc[test_select, :]

    if PATH_EXPORT is not None:
        file_n = os.path.join(PATH_EXPORT, f"{EXPORT_PREFIX}_filter_{FILTER_TYPE}.gpkg")
        gdf_out.to_file(file_n, driver='GPKG')

    return gdf_out


def plot_shape_distribution(gdf, path_fig_export, class_attr, attr_col='area'):
    # --- plot histogram of the shape distribution
    classes_avail = np.unique(gdf.loc[:, class_attr])

    fig, ax = plt.subplots(len(classes_avail) + 1, 1, figsize=(8.3, 11.7))
    gdf[class_attr].hist(bins=50, ax=ax[0])
    ax[0].set_title(f'{class_attr} distribution')
    ax[0].set_ylabel('counts')
    ax[0].set_xlabel('class')

    for e_ax, i_class in enumerate(classes_avail):
        gdf.loc[gdf.raster_val == i_class][attr_col].hist(bins=50, ax=ax[e_ax + 1])
        ax[e_ax + 1].set_ylim(0, 4)
        ax[e_ax + 1].set_title(
            f'{attr_col} distribution of {class_attr}: {i_class}')
        ax[e_ax + 1].set_ylabel('counts')
        ax[e_ax + 1].set_xlabel(attr_col)

    fig.tight_layout()
    fig.savefig(path_fig_export, format='pdf')

    return


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