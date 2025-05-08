
'''
== Create input for supervised classification or for test patch creation ==
== (proc step 3) ==

This script extracts training, validation and test data according the
specified training AOIs. The following data is extracted:
- labels (manual segmentation)
- greyscale data (can use intensity scaled image, pre-processed with MAIN_img_preproc.py)
- pre-calculated GLCM features (previously calculated with MAIN_calc_texture.py)

This script should be used to create the input for the Random Forest
training (ml_workflow) and to prepare the test patch data for as input for
CNN and Random Forest testing.
Note: the CNN training and prediction requires tiling and is therefore
    created with MAIN_create_training_tiles.py and
    MAIN_create_prediciton_tiles.py


--- Run script in command line
usage: MAIN_extract_input_untiled.py [-h] [--PARAM_FILE PARAM_FILE] PATH_PROC PROJ_PARAM_FILE labelling_area SCALE_TYPE

Process single images

positional arguments:
  PATH_PROC             prefix to processing folder and proc file
                        (required, e.g. "./docs/1_site_preproc/BLyaE_v1")
  PROJ_PARAM_FILE       file with project parameters
                        (required, e.g.: PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01)
  labelling_area        name of labelling area or several names separated by :
                        (required, e.g. A01:A02, referes to dict key in PARAM["LABEL_FILE_INP"])
  SCALE_TYPE            type of image scaling to be used (e.g. std4_8bit)

options:
  -h, --help            show this help message and exit
  --PARAM_FILE PARAM_FILE
                        name of task specific param file
                        (optional, default: PARAM03_extract_data)

--- for debugging use:
preproc/test/MAIN_extract_input_untiled_test.py

'''
import os
import sys
import pandas as pd
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
    # -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # --- processing step identifier
    PROC_STEP = 3
    # label (for file names) to be added to parameters
    inp['WINDOW_TYPE'] = 'untiled'  # where no window size wXXX since no
    # tiling is done

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_all_params(inp, PROC_STEP)
    PARAM['LOG_FILE_SUFFIX'] = 'extract_data'

    # ------------- define other sepcific paths -----------
    # folder to input data for data extraction
    PARAM['PATH_INP_DATA'] = os.path.join(
        PARAM['PATH_PROC'], PARAM['DATA_FOLDER'])
    # folder to save the extracted data for the specified AOI set (e.g. A01)
    PARAM['main_subfolder_save'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'],
                     PARAM['SUBFOLDER_PREFIX']))
    # create directory if non-existing
    if not os.path.isdir(PARAM['main_subfolder_save']):
        os.mkdir(PARAM['main_subfolder_save'])

    # --------------  save parameter values -----------------
    param_file_name = (
        f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        PARAM['PATH_TRAIN'], param_file_name, {'all_param': PARAM})

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
            os.path.join(PARAM['PATH_INP'],
                         PARAM['AOI_extract_sub_area'][e]),
            PARAM['EPSG_TARGET'])

    # ------ get reference and class image and merge all
    # get reference image to ensure that cooridnates are consistent
    data_img_orig_name = os.path.join(
        PARAM['PATH_INP_DATA'], PARAM['DATA_IMG'])
    data_img = geo_utils.read_rename_according_long_name(
        data_img_orig_name, mask_nan=False)

    # read metadta to check for available GLCM files
    if PARAM['META_FILE_TEX'] is not None:
        avail_files = pd.read_csv(
            os.path.join(PARAM['PATH_INP_DATA'], PARAM['META_FILE_TEX']),
            sep='\t', header=0, index_col=0).query(
                'proc_type == "tex" or proc_type == "calc"')
    else:
        # assign empty dict if do not use any texture files
        avail_files = {'file': [], 'name': []}

    LABEL_FILE_INP = PARAM['LABEL_FILE_INP'][PARAM['labelling_area']]

    # --- get and preprocess class labels
    # reads, reproj_matches to ref img, relables background, clips to AOI
    # clipping clips with AOIs
    class_img_orig_name = os.path.join(
        PARAM['PATH_LABELS'], LABEL_FILE_INP)
    class_img = training_data_prep_utils.prep_class_img(
        PARAM['PATH_LABELS'], LABEL_FILE_INP, AOI_coords, data_img)

    # to relabel classes could use:
    # (but relabeling is done later)
    # here also use just first band in case second band is quali
    class_img[0, :, :] = training_data_prep_utils.relabel_img(
        class_img[0, :, :], PARAM['TRAIN_RELABEL'])

    # ---- process labels and data for sub AOIs and save:
    class_img_p = {}
    for e_phase, i_phase in enumerate(PARAM['SUB_AREAS']):
        # create inputs for metadata
        meta = {'file_class': [], 'file_data': [],
                'x_bands': [], 'y_bands': [], 'data_img': [],
                'class_img': [], 'phase': []}
        meta['phase'].append(i_phase)

        # clip class image to sub AOI
        class_img_p[i_phase] = geo_utils.clip_to_aoi(
            class_img, AOI_coords_p[i_phase], from_disk=False)

        # derive class counts per sub AOI and update metadata
        df_count =  training_data_prep_utils.get_class_count(
            class_img_p[i_phase][0, :, :], PARAM['LABEL_NAMING'][0],
            PARAM['LABEL_NAMING'][1], PARAM['SUBFOLDER_PREFIX'])
        meta.update(
            {f"perc-class_{x}":y for x, y in
             zip(df_count.class_num.tolist(),
                 df_count.perc.tolist())})

        # clip data image to sub AOI
        data_img_p = geo_utils.clip_to_aoi(
            data_img, AOI_coords_p[i_phase], from_disk=False)
        # reproject match data image to class for exact grid match
        data_img_p = data_img_p.rio.reproject_match(
            class_img_p[i_phase],
            Resampling=rasterio.enums.Resampling.nearest)

        # create file name and save data
        file_name = f"{PARAM['FILE_PREFIX'][e_phase]}_data.tif"
        meta['file_data'].append(file_name)
        meta['data_img'].append(data_img_orig_name)
        meta['x_bands'].append(':'.join(data_img_p.band.values.tolist()))
        geo_utils.save_to_geotiff(
            data_img_p, PARAM['main_subfolder_save'],
            file_name, suffix='', add_crs=False)

        # create file name and save class
        file_name = f"{PARAM['FILE_PREFIX'][e_phase]}_seg.tif"
        meta['file_class'].append(file_name)
        meta['class_img'].append(class_img_orig_name)
        meta['y_bands'].append(
            ':'.join(class_img_p[i_phase].band.values.tolist()))
        geo_utils.save_to_geotiff(
            class_img_p[i_phase], PARAM['main_subfolder_save'],
            file_name, suffix='', add_crs=False)

        # save metadata
        df_meta = pd.DataFrame.from_dict(meta)
        meta_filename = os.path.join(
            PARAM['main_subfolder_save'],
            PARAM['FILE_PREFIX'][e_phase] + '_meta_data.txt')
        if os.path.isfile(meta_filename):
            df_meta_old = pd.read_csv(
                meta_filename, sep='\t', header=0, index_col=0)
            df_meta = pd.concat([df_meta_old, df_meta], axis=0)
            df_meta = df_meta.drop_duplicates(
                subset=['file_data', 'file_class'], keep='last')
        df_meta.to_csv(meta_filename, sep='\t', header=True)

    # ----- extract GLCM features for each sub-AOI
    # (one file per GLCM window size)
    for i_file, i_name in zip(avail_files['file'], avail_files['name']):
        glcm_img = geo_utils.read_rename_according_long_name(
            os.path.join(PARAM['PATH_INP_DATA'], i_file),
            mask_nan=False)
        for e_phase, i_phase in enumerate(PARAM['SUB_AREAS']):
            glcm_img_p = geo_utils.clip_to_aoi(
                glcm_img, AOI_coords_p[i_phase], from_disk=False)

            glcm_img_p = glcm_img_p.rio.reproject_match(
                class_img_p[i_phase],
                Resampling=rasterio.enums.Resampling.bilinear)

            file_name = f"{PARAM['FILE_PREFIX'][e_phase]}_data_{i_name}.tif"
            geo_utils.save_to_geotiff(
                glcm_img_p, PARAM['main_subfolder_save'],
                file_name, suffix='', add_crs=False)

    logging.info(
        '\n---- Finalised extracting data for AOIs'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '\n\n')

    # save time measure
    time_file = f"A_{PARAM['SUBFOLDER_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}"
    monitoring_utils.save_time_stats(prof, PARAM['PATH_TRAIN'], time_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process single images')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder and proc file'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('file with project parameters'))
    parser.add_argument(
        'labelling_area', type=str,
        help='name of labelling area or several names separated by : (e.g. test01:test02, referes to dict key in PARAM["LABEL_FILE_INP"])')
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
