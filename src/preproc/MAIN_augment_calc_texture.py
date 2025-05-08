"""
== Augment and re-calculate GLCM texture for each tile (proc step 4) ==
This script needs to be used as an additional pre-processing step if want
to do augmentation and recalculation of GLCM outside the training loop
(offline augmentation)

Offline augmentation has shown to be more prone to overfitting. However,
it can be useful for pre-tests as it is much faster than on the fly
augmentation since recalculating the GLCMs at every epoch (after
augmentation) takes a lot of computation time.

Thus, this script should be used to do the augmentation and GLCM
calculation prior to training. Afterwards can use MAIN_train_offl_aug.py
for training.


---- RUN script in command line:
usage: MAIN_augment_calc_texture.py [-h] [--PARAM_FILE PARAM_FILE] [--GPU_DEVICE GPU_DEVICE] [--N_JOBS N_JOBS] PATH_PROC PROJ_PARAM_FILE tiling_area SCALE_TYPE

positional arguments:
  PATH_PROC             Path to site/project processing folder
  PROJ_PARAM_FILE       Name of file with project parameters (.py file without file extension)
  tiling_area           name of tiling area or several names separated by ":" e.g. A01:A02. This relates to area name in subfolder.
  SCALE_TYPE            type of image scaling to be used (e.g. std4_8bit)

options:
  -h, --help            show this help message and exit
  --PARAM_FILE PARAM_FILE
                        Name of file with processing step specific parameters
                        (.py file without extension)
  --GPU_DEVICE GPU_DEVICE
                        GPU device number to use for texture calculation
  --N_JOBS N_JOBS       how many parallel jobs for augmentation and
                        texture calc (!!! needs to fit into GPU memory !!!)

--- for debugging use:
preproc/test/MAIN_augment_calc_texture_test.py

"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
import datetime as dt

import joblib

import memory_profiler

# ----------------- import custom utils -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import utils.file_utils as file_utils
import utils.monitoring_utils as monitoring_utils
import param_settings.param_utils as param_utils


@monitoring_utils.conditional_profile
def main(inp):

    # has to be initialized here due to GPU_device initialisation
    import cnn_workflow.cnn_workflow.custom_data_loader as custom_data_loader
    import cnn_workflow.cnn_workflow.augmentation_class as augmentation_class

    # -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # ---------- processing step identifier --------
    PROC_STEP = 4
    # phase is used as an identifier if need to extract labels
    # is phase is 'predict' then label file are not read in
    # Here just use 'augment'' since do not have any phases defined
    inp['PHASE'] = 'augment'
    phase_flag = inp['PHASE']

    # -----initialize all paramters
    PARAM = param_utils.initialize_all_params(inp, PROC_STEP)
    PARAM['LOG_FILE_SUFFIX'] = 'augment_tiles'


    PARAM['PATH_EXPORT'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'], PARAM['subfolder']))

    # --------------  save parameter values -----------------
    param_file_name = (
        f"A_{PARAM['FILE_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        PARAM['PATH_EXPORT'], param_file_name, {'all_param': PARAM})

    # -------- setup logging
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{PARAM['FILE_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_EXPORT'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']}, Proc step {PROC_STEP}")

    # ---- Get filenames of training/testing chips
    # (are read form metadata file)
    file_names = custom_data_loader.GetFilenames(
        PARAM['PATH_EXPORT'], PARAM['FILE_PREFIX'],
        x_bands=PARAM['X_BANDS'])

    if PARAM['load_phase_separate']:
        # -- read form metadata file, with specified AOI SUB_AREAS
        file_names.get_file_names_from_meta(
            PARAM['SUB_AREAS'], phase=phase_flag, query_text=None)
        # (query could be used to filter the tiles. However, this done
        #  when loading data into data_loader during training)
    else:
        # -- search filenames in folder
        file_names.get_file_names_glob(
            file_id=PARAM["FILE_PREFIX"], phase=phase_flag)

    # -- initialize augmentation class
    n_augment = PARAM['n_augment']
    tile_aug = augmentation_class.AugmentDataset(
        file_names.data_files[phase_flag], file_names.seg_files[phase_flag],
        proj_param_file=PARAM['PROJ_PARAM_FILE'],
        texture_param_file=PARAM['texture_param_file'],
        x_bands=file_names.x_bands, y_bands=file_names.y_bands,
        norm=PARAM['NORM'], augmentations_geom=PARAM['aug_geom'],
        augmentations_col=PARAM['aug_col'],
        band_lst_col_aug=PARAM['band_lst_col_aug'],
        n_augment=n_augment,
        augmentations_range=PARAM['aug_range'],
        aug_vers=PARAM['aug_vers'],
        tex_file_area_prefix_add=PARAM['tiling_area'],
        PATH_BASE=PARAM['PATH_BASE'], SCALE_TYPE=PARAM['SCALE_TYPE'])


    # =================== AUGMENT TILES =========================
    #tile_aug.augment_img(0)
    if n_augment > 0:
        # augment images
        mode = 'threads'  # 'processes' #'threads'
        n_files = len(file_names.data_files[phase_flag])
        joblib.Parallel(n_jobs=PARAM['N_JOBS'], prefer=mode, verbose=0)(
            joblib.delayed(tile_aug.augment_img)(i) for i in range(n_files))


    # ------ update filenames to save metadata with augmentation ------
    # --- create data frame with all subareas
    # original file names
    orig_names_data = file_names.data_files[phase_flag].copy()
    orig_names_seg = file_names.seg_files[phase_flag].copy()

    # original file name without file suffix
    orig_prefix_data = pd.Series(orig_names_data).apply(
        lambda x: os.path.basename(x).split('.')[0]).tolist() * (n_augment+1)

    # for numbering rows
    aug_num = [0] * len(orig_names_data)
    for i in range(1, n_augment+1):
        # row numbering
        aug_num += [i] * len(orig_names_data)

        # -- create names of augmentation files
        # suffix to be added to the end of the original file name
        suffix = (
            f"_aug{str(PARAM['aug_vers'])}-{i:02d}.tif")
        new_names_data = pd.Series(orig_names_data).apply(
            lambda x: x.split('.')[0] + suffix).tolist()
        new_names_seg = pd.Series(orig_names_seg).apply(
            lambda x: x.split('.')[0] + suffix).tolist()
        # add augmentation suffix to orig files
        file_names.data_files[phase_flag] += new_names_data
        file_names.seg_files[phase_flag] += new_names_seg

    # non-augmented and augmented datafile basename
    data_file_basename = [
        os.path.basename(x) for x in file_names.data_files[phase_flag]]
    seg_file_basename = [
        os.path.basename(x) for x in file_names.seg_files[phase_flag]]

    # create DataFrame with all metadata of all files (augmented and non-augmented)
    df_file_names = pd.DataFrame(
        np.array([data_file_basename, seg_file_basename,
                  orig_prefix_data]).T,
        columns=['file_data', 'file_class', 'orig_prefix_data'])
    df_file_names['prefix_data'] = df_file_names['file_data'].apply(
        lambda x: os.path.basename(x).split('.')[0])
    df_file_names['prefix_seg'] = df_file_names['file_class'].apply(
        lambda x: os.path.basename(x).split('.')[0])
    df_file_names['aug_num'] = aug_num

    # -add additional columns from metadata
    # (meta is only available if have been loading data from metafile)
    if PARAM['load_phase_separate']:
        col_add = [
            x for x in file_names.meta[phase_flag].columns
            if x.find('count') > -1]
        col_add += ['x_bands', 'y_bands']
        df_file_names.set_index('orig_prefix_data', inplace=True)
        file_names.meta[phase_flag].set_index(
            'orig_prefix_data', inplace=True)
        df_file_names.loc[:, col_add] = file_names.meta[phase_flag].loc[:, col_add]
        df_file_names.reset_index(inplace=True)
    # extract subarea name if any
    n_suffix = len(PARAM['FILE_PREFIX'].split('_'))
    df_file_names['sub_area'] = df_file_names['file_data'].apply(
        lambda x: x.split('_')[n_suffix-1][len(PARAM['tiling_area']):])
    sub_areas_saved = df_file_names['sub_area'].unique()

    # separately save metadata per SUB_AREA
    for i_sub in sub_areas_saved:
        fn_metadata = (
            f"{PARAM['FILE_PREFIX']}{i_sub}_meta_data_augment_Av{PARAM['aug_vers']}.txt")
        path_file = os.path.join(PARAM['PATH_EXPORT'], fn_metadata)

        df_file_names.query('sub_area == @i_sub').to_csv(
            path_file, sep='\t', header=True)

    # -- save augmentation parameters
    aug_param_file_name = (
        f"{PARAM['FILE_PREFIX']}_{phase_flag}_aug_param_Av{str(PARAM['aug_vers'])}.txt")
    key_exp = ['n_augment', 'aug_geom_probab', 'aug_col_probab',
               'band_lst_col_aug', 'aug_vers']
    file_utils.write_param_file(
        PARAM['PATH_EXPORT'], aug_param_file_name,
        {'all_param': {x: PARAM[x] for x in key_exp}})

    # =================== CALCULATE GLCMs =========================
    # ---- calculate GLCMs for orig and augmented images
    mode = 'threads'  # 'processes' #'threads'
    n_files = len(file_names.data_files[phase_flag])
    num_gpu = 0  # use here 0 as only one GPU visible
    joblib.Parallel(n_jobs=PARAM['N_JOBS'], prefer=mode, verbose=0)(
        joblib.delayed(tile_aug.run_recalc_texture_gpu)(i, num_gpu)
        for i in range(n_files))

    logging.info(
        '\n---- Finalised calculating texture tiles {}\n\n'.format(
            dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

    # ---- save time stats
    time_file = (
        f"A_{PARAM['FILE_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_{phase_flag}")
    monitoring_utils.save_time_stats(prof, PARAM['PATH_EXPORT'], time_file)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Augment and calculate texture of tiles')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('Path to site/project processing folder'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('Name of file with project parameters (.py file without file extension)'))
    parser.add_argument(
        'tiling_area', type=str,
        help='name of tiling area or several names separated by ":" e.g. A01:A02. This relates to area name in subfolder.')
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='Name of file with processing step specific parameters (.py file without extension)',
        default='PARAM04_augment_data')
    parser.add_argument(
        '--GPU_DEVICE', type=int,
        help='GPU device number to use for texture calculation',
        default=0)
    parser.add_argument(
        '--N_JOBS', type=int,
        help='how many parallel jobs for augmentation and texture calc (!!! needs to fit into GPU memory !!!)',
        default=20)

    args = parser.parse_args()

    tiling_area_lst = vars(args)['tiling_area'].split(':')
    inp_param = vars(args)

    # For GPU device initialization use environmental variable. This
    # ensures that only the specified device is used.
    # This is required because importing cucim (if installed) within
    # glcm_cupy seems to initialize all GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vars(args)['GPU_DEVICE'])
    import cupy
    with cupy.cuda.Device(0):  # use zero here since only one device visible
        for i in tiling_area_lst:
            # loop through different labelling areas
            inp_param['tiling_area'] = i
            main(vars(args))

