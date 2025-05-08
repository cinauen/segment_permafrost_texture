'''
'''

import os
import argparse


# ===== import specific utils =====
import utils.geo_utils as geo_utils
import utils.xarray_utils as xarray_utils


def main(PRED_IMG=None, TRUE_IMG=None, AOI_PATH=None, EPSG=32654,
         DICT_RELABEL=None, MASK_TO_NAN_LST=None, PREFIX_OUT=None,
         additionally_save_with_min_px_size=3):
    '''
    pred_img and true img can be path or xarray image
    '''
    # ----- get prediction image -----
    if isinstance(PRED_IMG, str):
        pred_xr = geo_utils.read_rename_according_long_name(
            PRED_IMG, chunk=None)
        pred_xr = geo_utils.check_for_missing_fill_val(pred_xr)
        # change to integer if required
        if (pred_xr.dtype.name.find('float') > -1
            or pred_xr.dtype.name.find('int64') > -1):
            pred_xr = geo_utils.convert_img_to_dtype(
                pred_xr, dtype_out='uint8', nodata_out=0,
                replace_zero_with_nan=False)
        pred_xr = pred_xr.assign_coords(band=('band', ['class']))
    else:
        pred_xr = PRED_IMG.copy()

    # ----- get ground truth image -----
    if isinstance(TRUE_IMG, str):
        true_xr = geo_utils.read_rename_according_long_name(
            TRUE_IMG, chunk=None)
        # ignore weights in case they are given as second band
        if true_xr.shape[0] > 1:
            true_xr = true_xr[:1, :, :]
        true_xr = geo_utils.check_for_missing_fill_val(true_xr)
        # change to integer if required
        if (true_xr.dtype.name.find('float') > -1
            or true_xr.dtype.name.find('int64') > -1):
            true_xr = geo_utils.convert_img_to_dtype(
                true_xr, dtype_out='uint8', nodata_out=0,
                replace_zero_with_nan=False)

        # relabel classes where required
        if DICT_RELABEL is not None and DICT_RELABEL != 'None' and len(DICT_RELABEL) > 0:
            if isinstance(DICT_RELABEL, str):
                DICT_RELABEL = eval(DICT_RELABEL)
            true_xr = xarray_utils.relabel_xarray(
                true_xr, DICT_RELABEL)

        # if required set some classes to 0
        if MASK_TO_NAN_LST is not None and MASK_TO_NAN_LST != 'None':
            if isinstance(MASK_TO_NAN_LST, str):
                MASK_TO_NAN_LST = eval(MASK_TO_NAN_LST)
            if  len(MASK_TO_NAN_LST) > 0:
                true_xr = xarray_utils.set_label_to_nan(
                    true_xr, MASK_TO_NAN_LST, fill_val=0)
    else:
        true_xr = TRUE_IMG.copy()

    # if required clip images
    if AOI_PATH is not None and AOI_PATH != 'None':
        AOI_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
            AOI_PATH, EPSG)

        pred_xr = geo_utils.clip_to_aoi(
            pred_xr, AOI_coords, EPSG, from_disk=True)
        true_xr = geo_utils.clip_to_aoi(
            true_xr, AOI_coords, EPSG, from_disk=True)

    # save as .geojson
    if PREFIX_OUT is None and isinstance(PRED_IMG, str):
        PATH_OUT = os.path.join(
            os.path.dirname(PRED_IMG),
            os.path.basename(PRED_IMG).split('.')[0] + '_pred')
    else:
        PATH_OUT = PREFIX_OUT + '_pred'

    if additionally_save_with_min_px_size is not None:
        # calculate predicted WITH pixel removal
        pred_rem = geo_utils.remove_small_shapes_from_raster(
            pred_xr, additionally_save_with_min_px_size, 'class', EPSG,
            plot_hist=True)

        gdf_rem = geo_utils.create_shapefile_from_raster(
            pred_rem, 'class', EPSG, unit='km')
        gdf_rem['area_m'] = gdf_rem['area_km']*1000000
        gdf_rem['px'] = gdf_rem['area_m']/(1.5*1.5)
        gdf_rem['perimeter'] = gdf_rem.length
        gdf_rem['circularity'] = (gdf_rem.length**2)/gdf_rem['area_m']

        # save predicted WITH pixel removal
        gdf_rem.to_csv(
            path_or_buf=f"{PATH_OUT}_{additionally_save_with_min_px_size}px_rem.txt", sep='\t',
            lineterminator='\n', header=True)
        gdf_rem.to_file(
            f"{PATH_OUT}_{additionally_save_with_min_px_size}px_rem.geojson", driver='GeoJSON')


    # calculate predicted without pixel removal
    gdf_pred = geo_utils.create_shapefile_from_raster(
        pred_xr, 'class', EPSG, unit='km')
    gdf_pred['area_m'] = gdf_pred['area_km']*1000000
    gdf_pred['px'] = gdf_pred['area_m']/(1.5*1.5)
    gdf_pred['perimeter'] = gdf_pred.length
    gdf_pred['circularity'] = (gdf_pred.length**2)/gdf_pred['area_m']

    # save predicted without pixel removal
    gdf_pred.to_csv(path_or_buf=PATH_OUT + '.txt', sep='\t',
                   lineterminator='\n', header=True)
    gdf_pred.to_file(PATH_OUT + '.geojson', driver='GeoJSON')

    # calculate true
    gdf_true = geo_utils.create_shapefile_from_raster(
        true_xr, 'class', EPSG, unit='km')
    gdf_true['area_m'] = gdf_true['area_km']*1000000
    gdf_true['px'] = gdf_true['area_m']/(1.5*1.5)
    gdf_true['perimeter'] = gdf_true.length
    gdf_true['circularity'] = (gdf_true.length**2)/gdf_true['area_m']

    # save true
    if PREFIX_OUT is None and isinstance(PRED_IMG, str):
        PATH_OUT = os.path.join(
            os.path.dirname(PRED_IMG),
            os.path.basename(PRED_IMG).split('.')[0] + '_true')
    else:
        PATH_OUT = PREFIX_OUT + '_true'
    gdf_true.to_csv(path_or_buf=PATH_OUT + '.txt', sep='\t',
                       lineterminator='\n', header=True)
    gdf_true.to_file(PATH_OUT + '.geojson', driver='GeoJSON')

    return gdf_pred, gdf_true



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process single images')
    parser.add_argument(
        'PRED_IMG', type=str,
        help=('PATH of predicted image (if use not as direct function...)'))
    parser.add_argument(
        'TRUE_IMG', type=str,
        help=('PATH of ground truth image (if use not as direct function...)'))
    parser.add_argument(
        '--AOI_PATH', type=str,
        help='path to AOI file if need clipping', default=None)
    parser.add_argument(
        '--EPSG', type=int, help='EPSG', default=32654)
    parser.add_argument(
        '--DICT_RELABEL', type=str, help='if need relabeling of classes',
        default='{}')
    parser.add_argument(
        '--MASK_TO_NAN_LST', type=str, help='if some classes should be set to None',
        default=None)


    args = parser.parse_args()

    main(**vars(args))




