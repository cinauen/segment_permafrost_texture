'''
==== Split training areas into tiles as input for CNN training ====
==== (proc step 3) ====
For specified AOIs extract labels and data (greyscale imagery)
and create tiles with predefined window size and overlap.

Note: GLCM features are either calculated later within the trainig loop
    or for offline augmentation in a separate step using
    MAIN_augment_calc_texture.py


--- Run script in command line
usage: MAIN_create_training_tiles.py [-h] [--PARAM_FILE PARAM_FILE] PATH_PROC PROJ_PARAM_FILE labelling_area SCALE_TYPE

Process single images

positional arguments:
  PATH_PROC             prefix to processing folder and proc file
                        (required, e.g. "./example/1_site_preproc/BLyaE_v1")
  PROJ_PARAM_FILE       file with project parameters
                        (required, e.g.: PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01)
  labelling_area        name of labelling area or several names separated by :
                        (required, e.g. A01:A02, refers to dict key in PARAM["LABEL_FILE_INP"])
  SCALE_TYPE            type of image scaling to be used (e.g. std4_8bit)

options:
  -h, --help            show this help message and exit
  --PARAM_FILE PARAM_FILE
                        name of task specific param file
                        (optional, default: PARAM03_extract_data)


--- for debugging use:
seg_preproc/test/MAIN_create_training_tiles_test.py

'''

import os
import sys
import numpy as np
import datetime as dt
import logging
import argparse

import memory_profiler

# ----------------- import custom utils -----------------
sys.path.insert(0,os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import preproc.training_data_prep_utils as training_data_prep_utils
import utils.file_utils as file_utils
import utils.monitoring_utils as monitoring_utils
import utils.geo_utils as geo_utils
import utils.plotting_utils as plotting_utils
import param_settings.param_utils as param_utils


@monitoring_utils.conditional_profile
def main(inp):
    # %% -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # --- processing step identifier
    PROC_STEP = 3

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_all_params(inp, PROC_STEP)
    PARAM['LOG_FILE_SUFFIX'] = 'create_training_tiles'
    # ------------- define other sepcific paths -----------
    # folder to input data for tiling
    PARAM['PATH_INP_DATA'] = os.path.join(
        PARAM['PATH_PROC'], PARAM['DATA_FOLDER'])
    # folder to save tiles of specific AOI set (e.g. A01)
    PARAM['main_subfolder_save'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'], PARAM['SUBFOLDER_PREFIX']))
    # create directory if non-existing
    if not os.path.isdir(PARAM['main_subfolder_save']):
        os.mkdir(PARAM['main_subfolder_save'])

    # --------------  save parameter values -----------------
    param_file_name = (
        f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        PARAM['PATH_TRAIN'], param_file_name, {'all_param': PARAM})

    rot_deg = PARAM['ROTATE_DEGREE'][PARAM['labelling_area']]

    # --------------- setup logging ---------------
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_TRAIN'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']}, Proc step {PROC_STEP}")

    # ----------------- get AOI of full labelling area ----------
    AOI_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
        os.path.join(PARAM['PATH_INP'], PARAM['AOI_extract_area']),
        PARAM['EPSG_TARGET'])

    # ----------------- get AOIs of sub-areas area ----------
    AOI_coords_p = {}
    AOI_poly_p = {}
    for e, i in enumerate(PARAM['SUB_AREAS']):
        AOI_coords_p[i], AOI_poly_p[i] = geo_utils.read_transl_geojson_AOI_coords_single(
            os.path.join(PARAM['PATH_INP'], PARAM['AOI_extract_sub_area'][e]),
            PARAM['EPSG_TARGET'])

    # ------ get data image (is greyscale image)
    data_img = geo_utils.read_rename_according_long_name(
        os.path.join(PARAM['PATH_INP_DATA'], PARAM['DATA_IMG']),
        mask_nan=False)

    # ------- scalre AOIs to avoid edge effects -------
    # Use enlarged AOI for clipping. This avoids later edge effects in
    # GLCM calculations at the outer labelling area boundaries
    fact_padding = 2  # add an additional factor is required to ensure
    # enough padding is use rotation
    AOI_scaled_gdf = geo_utils.scale_AOI(
        AOI_poly, data_img.rio.resolution()[0],
        -PARAM['WINDOW_PADDING_ADD'] * fact_padding,
        PARAM['EPSG_TARGET'])
    AOI_scaled_gdf.to_file(
        os.path.join(PARAM['main_subfolder_save'],
                     f"{PARAM['SUBFOLDER_PREFIX']}_AOI_scale.shp"))

    AOI_p_scaled_gdf = {}
    for e, i in enumerate(PARAM['SUB_AREAS']):
        AOI_p_scaled_gdf[i] = geo_utils.scale_AOI(
            AOI_poly_p[i], data_img.rio.resolution()[0],
            -PARAM['WINDOW_PADDING_ADD'] * fact_padding,
            PARAM['EPSG_TARGET'])

    # ------ clip data with AOI
    # !!! it is best to not use "from_disk" as this might introduce
    # inconsistencies at the borders such that class and data tiles
    # do not have same size
    data_img = geo_utils.clip_to_aoi_gdf(
        data_img, AOI_scaled_gdf, from_disk=False, drop_na=True)

    # use float values here such that all bands are consistent
    # and also due to rotation. Will be converted back to integer later.
    data_img = geo_utils.convert_img_to_dtype(
        data_img, dtype_out='float64', nodata_out=np.nan,
        replace_zero_with_nan=True)

    # --- get and preprocess class labels
    # reads, reproj_matches to ref img, relables background, clips to AOI
    # clipping is done with original AOI, not extended one (AOI_scaled_gdf)
    # used for grey scale) like this can use the zeros from the labelling
    # as mask to mask the edges of the training set after GLCM calculation
    class_img = training_data_prep_utils.prep_class_img(
        PARAM['PATH_LABELS'], PARAM['LABEL_FILE_INP'][PARAM['labelling_area']],
        AOI_coords, data_img)

    # to relabel classes could use:
    # (but relabeling is done later)
    # here also use just first band in case second band is quali
    class_img[0, :, :] = training_data_prep_utils.relabel_img(
        class_img[0, :, :], PARAM['TRAIN_RELABEL'])

    # ------ get class counts
    # DataFrame with percentage occurence per class
    df_count =  training_data_prep_utils.get_class_count(
        class_img[0, :, :], PARAM['LABEL_NAMING'][0],
        PARAM['LABEL_NAMING'][1], PARAM['SUBFOLDER_PREFIX'])

    # save class occurence to class_stats.txt file (for CNN weighting)
    training_data_prep_utils.read_append_df(
        os.path.join(PARAM['PATH_BASE'], '2_segment'),
        'class_stats.txt', df_count, ['class', 'data_set', 'class_num'])

    # ---- convert class_img to float
    # use float values here such that all bands are consistent
    # and also due to rotation. Will be converted back later
    class_img = geo_utils.convert_img_to_dtype(
        class_img, dtype_out='float64', nodata_out=np.nan,
        replace_zero_with_nan=True)

    # ----- merge all required images
    # no need to add any channels here for now
    add_channels_file = PARAM['ADD_CHANNEL_IMG']

    # !!! AOI_coords clipping is only done on ad_channels_file
    img_merged = training_data_prep_utils.merge_img_to_analyse(
        data_img, PARAM['PATH_EXPORT'], add_channels_file,
        PARAM['ADD_CHANNEL_BAND'], AOI_coords, resampling_type='bilinear',
        class_img=class_img)
    # get class shape for clipping later
    class_n_bands = class_img.band.values.shape[0]

    # plot
    plotting_utils.plot_xarray_imshow(
        img_merged, PARAM['main_subfolder_save'], 'img_merged.png')


    # ---- clip img merged for different SUB_AREAS:
    img_merged_clip = {}
    gdf_merged = {}
    for i_phase in PARAM['SUB_AREAS']:
        img_merged_clip[i_phase] = geo_utils.clip_to_aoi_gdf(
            img_merged, AOI_p_scaled_gdf[i_phase])

        # clip labels to original AOI.
        # like this can use nans from labels as mask to clip edges of
        # GLCM calculation use class dimension (class_n_bands) in case
        # if have quali also in class file
        class_p = geo_utils.clip_to_aoi(
            img_merged_clip[i_phase], AOI_coords_p[i_phase],
            from_disk=False, drop_na=False)
        img_merged_clip[i_phase].data[-class_n_bands:, :, :] = class_p.data[-class_n_bands:, :, :]

        # create input gdf
        gdf_merged[i_phase] = training_data_prep_utils.get_gdf_drop_missing_class(
            img_merged_clip[i_phase], drop_col_subset=['1', 'class'])

    # ---------- ROTATE image and keep resolution
    if (rot_deg is not None and rot_deg != 0):
        # rotate images
        subfolder_save_rot01 = os.path.normpath(
            f"{PARAM['main_subfolder_save']}_rot_{PARAM['rot_inperp']}")
        if not os.path.isdir(subfolder_save_rot01):
            os.mkdir(subfolder_save_rot01)

        img_rot_keep_resol = {}
        for i_phase in PARAM['SUB_AREAS']:
            img_rot_keep_resol[i_phase] = training_data_prep_utils.rotate_img(
                img_merged_clip[i_phase], gdf_merged[i_phase],
                rot_deg, PARAM['EPSG_TARGET'],
                [PARAM['rot_inperp']] + ['bilinear']*(len(img_merged_clip[i_phase].band.values)-(class_n_bands+1)) + ['nearest']*class_n_bands,
                AOI_coords=None, same_resolution=True)

            # plot
            plotting_utils.plot_xarray_imshow(
                img_rot_keep_resol[i_phase], subfolder_save_rot01,
                f"img_rotated_keep_resolution_{PARAM['rot_inperp']}_{i_phase}.png")


    # ---------- SPLIT into sub images and export
    # ------- split non rotated img -------
    # for overlapping files loop thorugh
    start_x_idx = [0, PARAM['WINDOW_SHIFT_X']]
    start_y_idx = [0, PARAM['WINDOW_SHIFT_Y']]

    for e_phase, i_phase in enumerate(PARAM['SUB_AREAS']):
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
                img_split = img_merged_clip[i_phase].isel(
                    x=slice(i_x, None), y=slice(i_y, None)).coarsen(
                        x=PARAM['WINDOW_SPLIT'][0],
                        y=PARAM['WINDOW_SPLIT'][1],
                            boundary="pad").construct(
                                x=("x_coarse", "x_fine"),
                                y=("y_coarse", "y_fine"))

                # -- export images
                # select each tile (using the coarse coord numbering),
                # transform it back to orig dtype and export it as a
                # separate .tif file
                training_data_prep_utils.export_split_img(
                    img_split, add_channels_file,
                    PARAM['FILE_PREFIX'][e_phase], PARAM,
                    PARAM['main_subfolder_save'], count,
                    PARAM['window_size'],
                    trim_sub_img=PARAM['TRIM_SUB_IMG'],
                    out_type_data=PARAM['out_type_data'],
                    num_class_band=class_n_bands)

                # create overview plot
                training_data_prep_utils.plot_chip_overview(
                    img_split, i_phase, count,
                    PARAM['main_subfolder_save'])

                count += 1


    # ------- split rotated img (keep symmetric resolution) -------
    if (rot_deg is not None and rot_deg != 0):
        # ----- split rotated img ------
        subfolder_save_rot01 = os.path.normpath(
            f"{PARAM['main_subfolder_save']}_rot_{PARAM['rot_inperp']}")

        # for overlapping files loop thorugh
        start_x_idx = [0, PARAM['WINDOW_SHIFT_X']]
        start_y_idx = [0, PARAM['WINDOW_SHIFT_Y']]

        for e_phase, i_phase in enumerate(PARAM['SUB_AREAS']):
            count = 0
            for i_x in start_x_idx:
                for i_y in start_y_idx:
                    # split
                    img_rot_keep_resol_split = img_rot_keep_resol[i_phase].isel(
                        x=slice(i_x, None), y=slice(i_y, None)).coarsen(
                            x=PARAM['WINDOW_SPLIT'][0],
                            y=PARAM['WINDOW_SPLIT'][1],
                            boundary="pad").construct(
                                x=("x_coarse", "x_fine"),
                                y=("y_coarse", "y_fine"))

                    # export images
                    training_data_prep_utils.export_split_img(
                        img_rot_keep_resol_split, add_channels_file,
                        PARAM['FILE_PREFIX'][e_phase], PARAM,
                        subfolder_save_rot01, count,
                        PARAM['window_size'],
                        trim_sub_img=PARAM['TRIM_SUB_IMG'],
                        out_type_data=PARAM['out_type_data'],
                        num_class_band=class_n_bands)

                    # create overview plot
                    training_data_prep_utils.plot_chip_overview(
                        img_rot_keep_resol_split, i_phase, count,
                        subfolder_save_rot01)

                    count += 1

    logging.info(
        '\n---- Finalised splitting data into training tiles '
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        + '\n\n')

    # save time measure
    time_file = f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}"
    monitoring_utils.save_time_stats(prof, PARAM['PATH_TRAIN'], time_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training tiles')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder and proc file'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('file with project parameters'))
    parser.add_argument(
        'labelling_area', type=str,
        help='name of labelling area or several names separated by : (e.g. A01:A02, refers to dict key in PARAM["LABEL_FILE_INP"])')
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used (e.g. std4_8bit)')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file',
        default='PARAM03_extract_data')

    args = parser.parse_args()

    # loop through different labelling areas
    label_area_lst = vars(args)['labelling_area'].split(':')
    inp_param = vars(args)
    for i in label_area_lst:
        inp_param['labelling_area'] = i
        main(vars(args))