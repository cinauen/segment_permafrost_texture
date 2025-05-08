'''
===== Calculate texture for prediction tiles (proc step 4) =====

Textures are calcuated on the raw prediction tiles without any
augmentation.


--- Run script in command line
usage: MAIN_calc_texture_tiles.py [-h] [--PARAM_FILE PARAM_FILE] [--tiling_area TILING_AREA] [--GPU_DEVICE GPU_DEVICE] [--N_JOBS N_JOBS] PATH_PROC PROJ_PARAM_FILE SCALE_TYPE

positional arguments:
  PATH_PROC             Path to site/project processing folder
  PROJ_PARAM_FILE       file with project parameters
  SCALE_TYPE            type of image scaling to be used (e.g. std4_8bit)

options:
  -h, --help            show this help message and exit
  --PARAM_FILE PARAM_FILE
                        Name of file with project parameters
                        (.py file without file extension)
  --GPU_DEVICE GPU_DEVICE
                        GPU device number to use for texture calculation
  --tiling_area TILING_AREA
                        name of prediction or tiling area or several
                        names separated by ":". This relates to area
                        name in subfolder. Keep "all" for prediction files.
  --GPU_DEVICE GPU_DEVICE
                        GPU device number to use for texture calculation


--- for debugging use:
preproc/test/MAIN_create_prediction_tiles_test.py

'''

import os
import sys
import argparse
import importlib
import logging
import datetime as dt

from joblib import Parallel, delayed

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
    import texture_calc.FUNC_calc_texture_from_img as FUNC_calc_texture_from_img

    # -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # ---------- processing step identifier --------
    PROC_STEP = 4
    inp['PHASE'] = 'predict'

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_all_params(inp, PROC_STEP)
    PARAM['LOG_FILE_SUFFIX'] = 'calc_texture_tiles'


    PARAM['PATH_EXPORT'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'], PARAM['subfolder']))

    # ------------- add additional texture param ----------
    texture_param_module = importlib.import_module(
                f'.{PARAM["texture_param_file"]}', 'param_settings')
    texture_param_module.add_proc_param(PARAM)

    # --------------  save parameter values -----------------
    param_file_name = (
        f'A_{PARAM["FILE_PREFIX"]}_{PARAM["LOG_FILE_SUFFIX"]}_PARAM.txt')
    file_utils.write_param_file(
        PARAM['PATH_EXPORT'], param_file_name, {'all_param': PARAM})

    # -------- setup logging
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f'A_{PARAM["FILE_PREFIX"]}_{PARAM["LOG_FILE_SUFFIX"]}_error.log')
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_EXPORT'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']}, Proc step {PROC_STEP}")

    tiling_area = PARAM['tiling_area']


    # ---- Get filenames of tiles
    # (are read form metadata file)
    file_names = custom_data_loader.GetFilenames(
        PARAM['PATH_EXPORT'], PARAM['FILE_PREFIX'],
        x_bands=PARAM['X_BANDS'])

    if PARAM['load_phase_separate']:
        # -- read form metadata file, with specified AOI SUB_AREAS
        file_names.get_file_names_from_meta(
            PARAM['SUB_AREAS'], phase=PARAM['PHASE'], query_text=None)
        # (query could be used to filter the tiles. However, this done
        #  when loading data into data_loader during training)
    else:
        # -- search filenames in folder
        file_names.get_file_names_glob(
            file_id=PARAM["FILE_PREFIX"], phase=PARAM['PHASE'])


    # ======== option 1
    # -- to cacluate texture use directly FUNC_calc_texture_from_img
    # functions
    #FUNC_calc_texture_from_img.calc_texture_complete(
    #    file_names.data_files[PARAM['PHASE']][0],
    #    file_names.x_bands, PARAM)
    if True:
        # ---- calculate GLCMs for all tiles
        mode = 'threads'  # 'processes' #'threads'
        n_files = len(file_names.data_files[PARAM['PHASE']])
        Parallel(n_jobs=PARAM['N_JOBS'], prefer=mode, verbose=0)(
            delayed(FUNC_calc_texture_from_img.calc_texture_complete)(i_file, file_names.x_bands, PARAM)
                for i_file in file_names.data_files[PARAM['PHASE']])

    if False:
        # ======== option 1
        # -- to calculate texture use AugmentDataset class but just do not
        # apply any augmentation...
        tile_tex = augmentation_class.AugmentDataset(
            file_names.data_files[PARAM['PHASE']], None,
            proj_param_file=PARAM['PROJ_PARAM_FILE'],
            texture_param_file=PARAM['texture_param_file'],
            x_bands=file_names.x_bands, y_bands=None,
            norm=PARAM['NORM'],
            augmentations_geom=None,  # all augmentations set to None
            augmentations_col=None, band_lst_col_aug=None,
            n_augment=0,
            augmentations_range=None, aug_vers=None,
            tex_file_area_prefix_add=tiling_area)

        # ---- calculate GLCMs for orig and agmented images
        mode = 'threads'  # 'processes' #'threads'
        n_files = len(file_names.data_files[PARAM['PHASE']])
        num_gpu = 0  # use here 0 as only one GPU visible
        Parallel(n_jobs=PARAM['N_JOBS'], prefer=mode, verbose=0)(
            delayed(tile_tex.run_recalc_texture_gpu)(i, num_gpu)
                    for i in range(n_files))

    logging.info(
        '\n---- Finalised calculating texture tiles '
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        + '\n\n')

    # ---- save time stats
    time_file = (
        f'A_{PARAM["FILE_PREFIX"]}_{PARAM["LOG_FILE_SUFFIX"]}_{tiling_area}')
    monitoring_utils.save_time_stats(prof, PARAM['PATH_EXPORT'], time_file)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Calculate texture of tiles')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('Path to site/project processing folder'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('file with project parameters'))
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used (e.g. std4_8bit)')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='Name of file with project parameters (.py file without file extension)',
        default='PARAM04_augment_data')
    parser.add_argument(
        '--tiling_area', type=str,
        help='name of prediction or tiling area or several names separated by ":". This relates to area name in subfolder. Keep "all" for prediction files.',
        default='all')
    parser.add_argument(
        '--GPU_DEVICE', type=int,
        help='GPU device number to use for texture calculation',
        default=0)
    parser.add_argument(
        '--N_JOBS', type=int,
        help='how many parallel jobs for texture calc (!!! needs to fit into GPU memory !!!)',
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
