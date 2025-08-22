'''
Evaluates true positives (TP), true negatives (TN), false positives (FP),
and false negatives (FN) per ground truth uncertainty/weight.

For certain categories (baydzherakhs) the ground truth data contains
label certainty levels. The metrics are calculated per this uncertainty
class to provide a better understanding  of model performance in
relation to label certainty/data quality.

'''

import os
import sys
import argparse
import pandas as pd
from rasterio.enums import Resampling
import xarray


# ===== import specific utils =====
import utils.geo_utils as geo_utils
import utils.xarray_utils as xarray_utils
import utils.conversion_utils as conversion_utils


def main(PRED_IMG=None, TRUE_IMG=None, CLASS_TO_EVAL=1, AOI_PATH=None,
         EPSG=32654, DICT_RELABEL=None, MASK_TO_NAN_LST=None,
         PREFIX_OUT=None, min_px_size=None):
    '''
    Input:
    PRED_IMG: str (path) or xarray.DataArray
        prediction raster either given as path (string) or
        xarray.DataArray image
    TRUE_IMG: str (path) or xarray.DataArray
        ground truth raster either given as path (string) or
        xarray.DataArray image
    CLASS_TO_EVAL: int
        class/category number for which the metrics should be calculated
    AOI_PATH: str (optional)
        Path to .geojson file with area of interest
    EPSG: int
        Coordinate system CRS EPSG code
    DICT_RELABEL: dict or None (optional)
        Dictionary specifying class relabelling {from: to}
    MASK_TO_NAN_LST: list or None (optional)
        List of class/category numbers which should be set to background
        (thus not be taken into account for stats)
    PREFIX_OUT: str
        Prefix for output file name
    min_px_size: int or None (optional)
        If set to int, then all shapes smaller than the specified number are
        removed and filled with nearest value

    Retruns

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
        pred_xr = pred_xr.assign_coords(band=('band', ['class_pred']))
    else:
        pred_xr = PRED_IMG.copy()

    # ----- get ground truth image -----
    if isinstance(TRUE_IMG, str):
        true_xr = geo_utils.read_rename_according_long_name(
            TRUE_IMG, chunk=None)
        # ignore weights in case they are given as second band
        #if true_xr.shape[0] > 1:
        #    true_xr = true_xr[:1, :, :]
        true_xr = geo_utils.check_for_missing_fill_val(true_xr)
        # change to integer if required
        if (true_xr.dtype.name.find('float') > -1
            or true_xr.dtype.name.find('int64') > -1):
            true_xr = geo_utils.convert_img_to_dtype(
                true_xr, dtype_out='uint8', nodata_out=0,
                replace_zero_with_nan=False)
        true_class = true_xr[:1, :, :]
        true_weight = true_xr[1:, :, :]

        # relabel classes where required
        if DICT_RELABEL is not None and DICT_RELABEL != 'None':
            if isinstance(DICT_RELABEL, str):
                DICT_RELABEL = eval(DICT_RELABEL)
            true_class = xarray_utils.relabel_xarray(
                true_class, DICT_RELABEL)

        # if required set some classes to 0
        if MASK_TO_NAN_LST is not None and MASK_TO_NAN_LST != 'None':
            if isinstance(MASK_TO_NAN_LST, str):
                MASK_TO_NAN_LST = eval(MASK_TO_NAN_LST)
            if  len(MASK_TO_NAN_LST) > 0:
                true_class = xarray_utils.set_label_to_nan(
                    true_class, MASK_TO_NAN_LST, fill_val=0)
    else:
        true_class = TRUE_IMG.copy()
        sys.exit('!!! this is not implemented !!!')

    if min_px_size is not None:
        pred_xr = geo_utils.remove_small_shapes_from_raster(
            pred_xr, min_px_size, 'class_pred', EPSG, plot_hist=True)

    # rename band name of true class
    true_class = true_class.assign_coords(band=('band', ['class_true']))

    # merge all images to later create dataframe
    # take true_class as reference
    # use here nearest resampling as all values are integers
    true_weight = true_weight.rio.reproject_match(
            true_class, Resampling=Resampling.nearest)
    pred_xr = pred_xr.rio.reproject_match(
            true_class, Resampling=Resampling.nearest)

    img_merged = xarray.concat(
        [true_class, true_weight, pred_xr], dim='band')

    # if required clip images
    if AOI_PATH is not None and AOI_PATH != 'None':
        AOI_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
            AOI_PATH, EPSG)

        img_merged = geo_utils.clip_to_aoi(
            img_merged, AOI_coords, EPSG, from_disk=False)

    # create dataframe
    df = conversion_utils.xarray_to_df(
        img_merged, drop_col='spatial_ref', bands=None, stack_dim='band')
    # remove all background (where true class is zero)
    df = df.query('class_true != 0')

    # assign class weighting
    df['class_weight'] = 1
    df.loc[df.quali <= 60, 'class_weight'] = 0
    df.loc[df.quali > 90, 'class_weight'] = 2

    df['TP'] = 0
    df['FN'] = 0
    df['TN'] = 0
    df['FP'] = 0
    TP_query = (df.class_true == CLASS_TO_EVAL) & (df.class_true == df.class_pred)
    FN_query = (df.class_true == CLASS_TO_EVAL) & (df.class_pred != df.class_true)
    TN_query = (df.class_true != CLASS_TO_EVAL) & (df.class_pred != CLASS_TO_EVAL)
    FP_query = (df.class_true != CLASS_TO_EVAL) & (df.class_pred == CLASS_TO_EVAL)
    df.loc[TP_query, 'TP'] = 1
    df.loc[FN_query, 'FN'] = 1
    df.loc[TN_query, 'TN'] = 1
    df.loc[FP_query, 'FP'] = 1

    fn = os.path.basename(PRED_IMG)
    df_cts = pd.DataFrame(
        columns=['TP_W0', 'TP_W1', 'TP_W2',
                 'FN_W0', 'FN_W1', 'FN_W2', 'TN', 'FP',
                 f'px_sum_cls{CLASS_TO_EVAL}',
                 f'px_sum_no_cls{CLASS_TO_EVAL}',
                 f'weight0_cts_cls{CLASS_TO_EVAL}',
                 f'weight1_cts_cls{CLASS_TO_EVAL}',
                 f'weight2_cts_cls{CLASS_TO_EVAL}'],
        index=[fn])
    for i in [0, 1, 2]:
        df_cts.loc[fn, f'TP_W{i}'] = df.loc[df.class_weight == i, 'TP'].sum()
        df_cts.loc[fn, f'FN_W{i}'] = df.loc[df.class_weight == i, 'FN'].sum()
    df_cts.loc[fn, 'TN'] = df.loc[:, 'TN'].sum()
    df_cts.loc[fn, 'FP'] = df.loc[:, 'FP'].sum()
    df_cts.loc[fn, 'px_sum'] = df.shape[0]
    df_cts.loc[fn, f'px_sum_cls{CLASS_TO_EVAL}'] = df.query(
        'class_true == @CLASS_TO_EVAL').shape[0]
    df_cts.loc[fn, f'px_sum_no_cls{CLASS_TO_EVAL}'] = df.query(
        'class_true != @CLASS_TO_EVAL').shape[0]
    df_cts.loc[fn, f'weight0_cts_cls{CLASS_TO_EVAL}'] = df.query(
        'class_true == @CLASS_TO_EVAL and class_weight == 0').shape[0]
    df_cts.loc[fn, f'weight1_cts_cls{CLASS_TO_EVAL}'] = df.query(
        'class_true == @CLASS_TO_EVAL and class_weight == 1').shape[0]
    df_cts.loc[fn, f'weight2_cts_cls{CLASS_TO_EVAL}'] = df.query(
        'class_true == @CLASS_TO_EVAL and class_weight == 2').shape[0]


    if PREFIX_OUT is None and isinstance(PRED_IMG, str):
        PATH_OUT = os.path.join(
            os.path.dirname(PRED_IMG),
            f"{os.path.basename(PRED_IMG).split('.')[0]}_TP_FN_class{CLASS_TO_EVAL}")
    else:
        PATH_OUT = f"{PREFIX_OUT}_TP_FN_class{CLASS_TO_EVAL}"
    df_cts.to_csv(path_or_buf=PATH_OUT + '.txt', sep='\t',
                  lineterminator='\n', header=True)

    return df_cts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate TP and FN compared to sample weighting')
    parser.add_argument(
        'PRED_IMG', type=str,
        help=('PATH of predicted image (if use not as direct function...)'))
    parser.add_argument(
        'TRUE_IMG', type=str,
        help=('PATH of ground truth image (if use not as direct function...)'))
    parser.add_argument(
        'CLASS_TO_EVAL', type=str,
        help=('which class to evaluate as class number (e.g. for baydzherakhs = 1)'))
    parser.add_argument(
        '--AOI_PATH', type=str,
        help='path to AOI file if need clipping', default=None)
    parser.add_argument(
        '--EPSG', type=int, help='EPSG', default=32654)
    parser.add_argument(
        '--DICT_RELABEL', type=str, help='if need relabeling of classes',
        default='{7: 3, 3: 7}')
    parser.add_argument(
        '--MASK_TO_NAN_LST', type=str, help='if some classes should be set to None',
        default=None)


    args = parser.parse_args()

    main(**vars(args))




