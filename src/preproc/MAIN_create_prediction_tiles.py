"""
===== Split prediction areas into tiles as input for CNN inference =====
===== (proc step 3) =====
Split greyscale imagery into prediction tiles with predefined window size
and overlap.

Note:
    - This script uses the same splitting method as in MAIN_create_training_tiles.py
      However, it is simpler as no rotation and tiling of the labels are
      required. Also there is only the prediction phase. Thus, it is not
      required to loop through different AOIs for training and validation
      splits.
    - If required GLCM features have to be calculated separately with
      MAIN_calc_texture_tiles.py

Possible TODOs:
    - Merge MAIN_create_training_tiles.py MAIN_create_prediction_tiles.py
      into one function.
    - Could also allow extracting the pre-calculated GLCMs here.
      Would need to read in the required imagery and then.
      This could simply merge then using training_data_prep_utils.merge_img_to_analyse().
      Could then split the image into the tiles and then save each band
      as a separate .tif file


--- Run script in command line
usage: MAIN_create_prediction_tiles.py [-h] [--prediction_area PREDICTION_AREA] [--PARAM_FILE PARAM_FILE] PATH_PROC PROJ_PARAM_FILE SCALE_TYPE


positional arguments:
  PATH_PROC             prefix to processing folder and proc file
                        (required, e.g. "./example/1_site_preproc/BLyaE_v1")
  PROJ_PARAM_FILE       file with project parameters
                        (required, e.g.: PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01)
  SCALE_TYPE            type of image scaling to be used (e.g. std4_8bit)

options:
  -h, --help            show this help message and exit
  --prediction_area PREDICTION_AREA
                        name of .geojson AOI-file to tile if "all" PARAM["AOI_full_area"] is taken.
                        Several areas can be provided by separating .geojson files with ":"
                        (optional, default: "all")
  --PARAM_FILE PARAM_FILE
                        name of task specific param file
                        (optional, default: PARAM03_extract_data)


--- for debugging use:
preproc/test/MAIN_create_prediction_tiles_test.py

"""

import os
import sys
import numpy as np
import datetime as dt
import logging
import argparse
import rasterio.enums

import memory_profiler

# ----------------- import custom utils -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import preproc.training_data_prep_utils as training_data_prep_utils
import utils.file_utils as file_utils
import utils.monitoring_utils as monitoring_utils
import utils.geo_utils as geo_utils
import param_settings.param_utils as param_utils


@monitoring_utils.conditional_profile
def main(inp):
    # %% -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # --- processing step identifier
    PROC_STEP = 3  # use PROC_STEP 3 since need window size definitions

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_all_params(inp, PROC_STEP)
    PARAM['LOG_FILE_SUFFIX'] = 'create_prediction_tiles'
    # ------------- define other sepcific paths -----------
    # folder to input data for tiling
    PARAM['PATH_INP_DATA'] = os.path.join(
        PARAM['PATH_PROC'], PARAM['DATA_FOLDER'])

    # is all the use full area for prediction
    #if PARAM['prediction_area'] == 'all':
    #    PARAM['prediction_area'] = PARAM['AOI_full_area']

    # folder to save prediction tiles
    PARAM['main_subfolder_save'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'],
                     PARAM['SUBFOLDER_PREFIX']))
    if not os.path.isdir(PARAM['main_subfolder_save']):
        os.mkdir(PARAM['main_subfolder_save'])

    # --------------  save parameter values -----------------
    param_file_name = (
        f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        PARAM['PATH_TRAIN'], param_file_name, {'all_param': PARAM})

    # ------------------------ setup logging --------------------------
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_TRAIN'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']}, Proc step {PROC_STEP}")


    # -------------------- AOI of prediction area ---------------------
    AOI_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
        os.path.join(PARAM['PATH_INP'], PARAM['AOI_extract_area']),
        PARAM['EPSG_TARGET'])

    # ---------- get data image (is greyscale image) ----------
    # GLCMs are calculated with in a separate script
    # MAIN_calc_texture_tiles.py
    data_img = geo_utils.read_rename_according_long_name(
        os.path.join(PARAM['PATH_INP_DATA'], PARAM['DATA_IMG']),
        mask_nan=False)

    # --- get extended AOI to later avoid edge effects in GLCM calculation
    # for prediction tiles
    fact_padding = 2  # here use factor 2 to ensure that have enough
    # coverage at edges. However, since do not apply rotation, factor 1
    # would be fine as well.
    AOI_scaled_gdf = geo_utils.scale_AOI(
        AOI_poly, data_img.rio.resolution()[0],
        -PARAM['WINDOW_PADDING_ADD'] * fact_padding,
        PARAM['EPSG_TARGET'])
    AOI_scaled_gdf.to_file(
        os.path.join(PARAM['main_subfolder_save'],
                     f"{PARAM['SUBFOLDER_PREFIX']}_AOI_scale.shp"))

    # ------ clip data with AOI
    data_img_orig = data_img.copy()
    # clip with extended AOI
    # !!! it is best to not use from_disk as this might introduce
    # inconsistencies at the borders
    data_img = geo_utils.clip_to_aoi_gdf(
        data_img, AOI_scaled_gdf, from_disk=False, drop_na=True)
    # reproject match would not be required here
    # however it makes the process much faster..
    data_img = data_img.rio.reproject_match(
        data_img_orig, Resampling=rasterio.enums.Resampling.nearest)
    del data_img_orig

    # ---- convert data_img to float
    # Use float values here such that all bands are consistent.
    # Will be converted back later.
    data_img = geo_utils.convert_img_to_dtype(
        data_img, dtype_out='float64', nodata_out=np.nan,
        replace_zero_with_nan=True)

    # ---------- SPLIT into sub images and export
    # ------- split non rotated img -------
    # for overlapping files loop thorugh
    start_x_idx = [0, PARAM['WINDOW_SHIFT_X']]
    start_y_idx = [0, PARAM['WINDOW_SHIFT_Y']]
    count = 0
    for i_x in start_x_idx:
        for i_y in start_y_idx:
            # -- split:
            # add additional coordinates levels x_coarse and x_fine
            # to create overlapping grid cooridnates. The are provided as 2D array
            # of dimenstion [x_coarse, x_fine] and [y_coarse, y_fine]
            # x_coarse (the rows) correspond to all x coordinates of a tile
            # and the tile number can e.g. be extracted with
            # img_split.x_coarse.values.tolist()
            img_split = data_img.isel(x=slice(i_x, None),
                                      y=slice(i_y, None)).coarsen(
                x=PARAM['WINDOW_SPLIT'][0],
                y=PARAM['WINDOW_SPLIT'][1], boundary="pad").construct(
                    x=("x_coarse", "x_fine"),
                    y=("y_coarse", "y_fine"))

            # -- export images
            # select each tile (using the coarse coord numbering),
            # transform it back to orig dtype and export it as a separate
            # .tif file
            training_data_prep_utils.export_split_data_tile(
                img_split, [], PARAM['FILE_PREFIX'][0],
                PARAM, PARAM['main_subfolder_save'], count,
                trim_sub_img=PARAM['TRIM_SUB_IMG'],
                out_type_data=PARAM['out_type_data'])
            count += 1

    logging.info(
        '\n---- Finalised splitting data into prediction tiles '
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        + '\n\n')

    # save time measure
    time_file = f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}"
    monitoring_utils.save_time_stats(prof, PARAM['PATH_TRAIN'], time_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create prediction tiles')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder and proc file'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('file with project parameters'))
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used (e.g. std4_8bit)')
    parser.add_argument(
        '--prediction_area', type=str,
        help='name of .geojson AOI-file to tile if "all" PARAM["AOI_full_area"] is taken. Several areas can be provided by separating .geojson files with ":"', default='all')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file',
        default='PARAM03_extract_data')

    args = parser.parse_args()

    # loop through different prediction areas
    label_area_lst = vars(args)['prediction_area'].split(':')
    inp_param = vars(args)
    for i in label_area_lst:
        inp_param['prediction_area'] = i
        main(vars(args))