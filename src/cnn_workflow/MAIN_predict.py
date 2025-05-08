"""
================ Run prediction ===============
This script runs the prediction for a selected model and epoch.
The prediction can either be done based on the tiles (default) or on the
entire image at once.
For tiles the tile would have need to been created with:
- preproc/MAIN_create_prediction_tiles.py
- preproc/MAIN_calc_texture_tiles.py (if GLCM texture is required as input)

The prediction includes:
- Loading tiles or the full image (in dataloader):
- Preparing input features and labels for training (in dataloader):
    - apply optional standardisation
    - merge required bands into one torch tensor
- Initialize and load the trained model of the specified epoch
- Run prediction
    If run on tiles: tile predictions are saved in a temporary folder
    and tiles are afterwards merged
    Note on tile merging:
        Tiles were created with an overlap and outer edges of
        the predicted tiles are cut off due to edge effects. For the
        overlapping areas the probabilities of each class are averaged
        and classes are created afterwards (class of max probability)

The model to be used for the prediction is defined by:
- The area and imagery types (e.g. HEX, SPOT) it was trained on
    as defined in the project parameter file
    (e.g. /../example/2_segment/01_input/PARAM06_train_BLyaE_v1_HEX1979_A01_A02_set01a.py)
- the parameters used for feature preparation as defined in:
    param_settings/PARAM_inp_CNN_feature_prep_v01.txt
    (description see: docs/PARAM_options_feature_preparation_CNN.md)
- the training parameters as defined in:
    param_settings/PARAM_inp_CNN_train_v01.txt
    (description see: docs/PARAM_options_training_CNN.md)
- the model epoch, which must be provided as command line input
    (see below)

The main modules called in this file include:
- cnn_workflow/custom_data_loader.py: data loading and feature preparation
- cnn_workflow/custom_model.py: cnn model loading
- cnn_workflow/custom_train.py: run prediction and tile merging


--- Run script in command line
usage: MAIN_predict.py [-h] [--GPU_LST_STR GPU_LST_STR] [--CV_NUM CV_NUM] [--PATH_INP_BASE PATH_INP_BASE] PATH_PROC PATH_LOCAL PARAM_PREP_ID PARAM_TRAIN_ID EPOCH PARAM_FILE

positional arguments:
  PATH_PROC             path to processing folder (2_segment folder)
  PATH_LOCAL            path where to save temp files
  PARAM_PREP_ID         id of used merge and scaling parameters
                            e.g. v079 (PARAM_inp_CNN_feature_prep_v01.txt)
  PARAM_TRAIN_ID        id of used training parameters
                            e.g. t16onl (PARAM_inp_CNN_train_v01.txt)
  EPOCH                 epoch number
  PARAM_FILE            name of project parameter file

options:
  -h, --help            show this help message and exit
  --GPU_LST_STR GPU_LST_STR
                        list of GPUs (separated by ":")
  --CV_NUM CV_NUM       CV number if do not want to start at 0
  --PATH_INP_BASE PATH_INP_BASE
                        path to site pre processing folder as default (None)
                        the path to 1_site_preproc is used

--- for debugging use:
cnn_workflow/test/MAIN_predict_test.py
"""

import os
import sys
import numpy as np
import argparse
import tempfile
import logging
import datetime as dt

import torch

import memory_profiler
import matplotlib
matplotlib.use('qtagg')

# ----------------- import custom utils -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import cnn_workflow.custom_data_loader as custom_data_loader
import cnn_workflow.custom_train as custom_train
import cnn_workflow.custom_model as custom_model
import utils.file_utils as file_utils
import utils.monitoring_utils as monitoring_utils
import param_settings.param_utils as param_utils


@monitoring_utils.conditional_profile
def main(pca_calc_utils, inp, CV_NUM=0):
    # -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_segmentation_param(inp)
    PARAM['LOG_FILE_SUFFIX'] = 'prediction'

    # ----- file and folder settings
    # add cross-validation number to file prefix
    cv_suffix = f'cv{CV_NUM:02d}'
    PARAM['FILE_PREFIX_OUT'] = (
        f"{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}_{cv_suffix}")
    # where to get model and where to output predictions and tests
    folder_model_io = f"{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}_{cv_suffix}"
    # for train model input
    PARAM['PATH_TRAIN_INP'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'], folder_model_io))

    # --- set other aditional params -----
    # amount of classes
    n_classes = len(PARAM['CLASS_LABELS'][0])

    # --------------  save parameter values -----------------
    param_file_name = (
        f"A_{folder_model_io}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        os.path.join(PARAM['PATH_PROC'], '04_predict'),
        param_file_name, {'all_param': PARAM},
        close=True)

    # --------------- setup logging ---------------
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator
    # needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{folder_model_io}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_PROC'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']} CV{CV_NUM}")

    # ------ set device usage for torch -----
    device, PARAM['GPU_lst'] = custom_data_loader.set_device(
        PARAM['GPU_LST_STR'])

    # ----- Loop through all required predictions
    for i_pred_suffix, i_pred_inp in PARAM['PATH_PRED'].items():
        if isinstance(PARAM['FINAL_CLIP_AOI'], str):
            clip_AOI = PARAM['FINAL_CLIP_AOI']
        else:
            clip_AOI = PARAM['FINAL_CLIP_AOI'][i_pred_suffix]

        PARAM['PATH_PRED_OUT'] = os.path.join(
            PARAM['PATH_PROC'], '04_predict', i_pred_suffix,
            PARAM['model_folder_base'],
            f"{folder_model_io.split('_')[-1]}")

        if not os.path.isdir(PARAM['PATH_PRED_OUT']):
            os.makedirs(PARAM['PATH_PRED_OUT'])

        # ============== PREDICTION ===============
        # ---- Get filenames of prediction tile or whole image
        if PARAM['prediction_type'] == 'tiles':
            # (tiles are read form metadata file)
            path_search = os.path.join(PARAM['PATH_INP_BASE'], i_pred_inp[0])
            file_prefix = i_pred_inp[1]
            file_names = custom_data_loader.GetFilenames(
                path_search, file_prefix,
                x_bands=PARAM['X_BANDS'])
            # get file names from metadat file
            file_names.get_file_names_from_meta(
                '', phase='predict')
            files_pred_lst = file_names.data_files['predict']
        else:
            # if use single untiled image
            files_pred_lst = [
                os.path.join(PARAM['PATH_INP_BASE'], i_pred_inp[0])]

        file_id_lst = [i_pred_suffix]*len(files_pred_lst)

        # --- load prediction data and scale if required ---
        predict_ds = custom_data_loader.CustomDataset_ScaleMerge_pred(
            files_pred_lst,
            file_id_lst, PARAM['file_suffix_lst'],
            pca_calc_utils, PARAM['PATH_PROC'],
            x_bands=None,
            add_bands=PARAM['add_bands'],
            width_out=PARAM['window_size'], height_out=PARAM['window_size'],
            calc_pca=PARAM['calc_PCA'], PCA_components=PARAM['PCA_components'],
            standardize_add=PARAM['standardize_add'],
            norm_on_std=PARAM['norm_on_std'],
            norm_min_max_band1=PARAM['norm_min_max_band1'],
            take_log=PARAM['take_log'],
            gpu_no=PARAM['GPU_lst'][0],
            save_files_debug=False,  # if save files during debugging
            standardize_individual=PARAM['standardize_individual'],  # standardisation per individual tile
            if_std_band1=PARAM['if_std_band1'],  # if standardise grey scale band
            dl_phase='pred',
            debug_plot=False,
            norm_clip=PARAM['norm_clip'],  # if clip to 0 and 1 after nomalize (if normalize)
            exclude_greyscale=PARAM['exclude_greyscale'],
            aug_col=None,
            sensor_dict=PARAM['FILE_SENSOR_dict'],
            feature_add=PARAM['feature_add'])
        # stats file is defined per labelling area...
        predict_ds.get_stats_dict(
            PARAM['PATH_INP_BASE'], PARAM['STATS_FILE'],
            PARAM['PHASE_STATS_FILE'][CV_NUM])

        # ----- load data to pytorch
        predict_dl = torch.utils.data.DataLoader(
                predict_ds, batch_size=PARAM['N_BATCH'],
                num_workers=PARAM['N_WORKERS'], shuffle=False)
        n_channels = predict_ds.n_channels

        # -------- load model for predict
        # define model architecture for test and prediction
        # if would use last trained model then could aslo use unet from
        # above here - it migh be overfitted...
        model_pred = custom_model.initialize_model(
            PARAM['MODEL_ARCH'],  # 'smpUNet_test',   #
            PARAM['GPU_lst'], n_channels, n_classes, device,
            PARAM['use_batchnorm'], (np.nan, np.nan))

        # ----- pedict
        file_out_lst = [
            '_'.join(i_pred_inp[1].split('_')[:2]),
            PARAM['FILE_PREFIX_OUT'], PARAM['file_out_prefix_add'],
            f'ncla{n_classes}',
            ]

        file_out_prefix = file_utils.create_filename(file_out_lst)
        if 'old_naming' in PARAM.keys() and PARAM['old_naming'] == 1:
            # !!! TO BE REMOVED LATER !!!
            # allow naming from old code version
            file_prefix_inp = f"{PARAM['PARAM_TRAIN_ID']}_{cv_suffix}"
            file_inp_lst = [
                file_prefix_inp,
                PARAM['MODEL_ARCH'], 'nchannels', str(n_channels),
                'nclasses', str(n_classes)]
            file_out_suffix_short = file_utils.create_filename(
                [PARAM['FILE_PREFIX_OUT'], PARAM['file_out_prefix_add'],
                'nclasses', str(n_classes)])
        elif 'old_naming' in PARAM.keys() and PARAM['old_naming'] == 2:
            file_inp_lst = [
                PARAM['FILE_PREFIX_OUT'], PARAM['file_out_prefix_add'],
                'nclasses', str(n_classes)]
            file_out_suffix_short = file_utils.create_filename(
                [PARAM['FILE_PREFIX_OUT'], PARAM['file_out_prefix_add'],
                'nclasses', str(n_classes)])
        else:
            file_inp_lst = [
                PARAM['FILE_PREFIX_OUT'], PARAM['file_out_prefix_add'],
                f'ncla{n_classes}']
            file_out_suffix_short = file_utils.create_filename(
                [PARAM['FILE_PREFIX_OUT'], PARAM['file_out_prefix_add'],
                f'ncla{n_classes}'])
        file_inp_prefix = file_utils.create_filename(file_inp_lst)

        pred_c = custom_train.PredictRun(
            model_pred, predict_dl,
            PARAM['N_BATCH'], n_classes, PARAM['CLASS_LABELS'],
            PARAM['PATH_PRED_OUT'], file_out_prefix, device,
            file_inp_prefix, PARAM['EPSG_TARGET'],
            trained_model=None, model_path=PARAM['PATH_TRAIN_INP'],
            width_out=PARAM['window_size'], height_out=PARAM['window_size'],
            save_proba=PARAM['save_proba'])

        pred_c.read_model_metadata()
        pred_c.load_model(epoch=PARAM['EPOCH'])
        temp_prefix = os.path.join(PARAM['PATH_LOCAL'], 'temp_')
        with tempfile.TemporaryDirectory(prefix=temp_prefix) as tempdir:
            if PARAM['prediction_type'] == 'tiles':
                # Tiles are saved into temp folder to be merged later
                # The exported predictions include:
                # - the predicted classes
                # - the raw probablilities (no softmax) (if keep_proba_raw,
                #       this is required for tile averaging afterwards)
                # - the probabilities after using softmax (if keep_proba_softm)
                pred_c.out_path = tempdir
                pred_c.run_predict_proba(
                    parallel_run=False,
                    keep_proba_raw=True,  # will be averaged later this gives correct value
                    keep_proba_softm=False  # do not keep as average tiles later
                    )
                pred_c.out_path = PARAM['PATH_PRED_OUT']
                # ---- Merge tiles to cover the full area
                # this calculates the softmax probabilities after taking
                # the mean and uses argmax to get the classes
                # (also clips the predictions and creates .cog files)
                pred_c.merge_predicted_img(
                    AOI_path=PARAM['PATH_INP'],
                    AOI_file=clip_AOI)
                print('end merge')
            else:
                # Prediction on entire area at once
                pred_c.run_predict_proba(
                    keep_proba_raw=False,
                    keep_proba_softm=True,  # save softmax directly as no averaging used
                    file_suffix_out='_' + file_out_suffix_short)
                # no merging is required
                # here clip and save the predictions as cog
                pred_c.img_clip_cog(
                    PARAM['PATH_INP'],
                    clip_AOI)
        print('cleaned tempfiles')

    logging.info(
        '\n---- Finalised prediction {}\n\n'.format(
            dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

    # ---- save time stats
    time_file = '_'.join(
        ['A', file_out_prefix, PARAM['LOG_FILE_SUFFIX']])
    monitoring_utils.save_time_stats(
        prof, PARAM['PATH_PRED_OUT'], time_file)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('path to processing folder (2_segment folder)'))
    parser.add_argument(
        'PATH_LOCAL', type=str,
        help=('path where to save temp files'))
    parser.add_argument(
        'PARAM_PREP_ID', type=str,
        help=('id of used merge and scaling parameters e.g. v079 '
              + '(PARAM_inp_CNN_feature_prep_v01.txt)'))
    parser.add_argument(
        'PARAM_TRAIN_ID', type=str,
        help=('id of used training parameters e.g. t16onl '
              + '(PARAM_inp_CNN_train_v01.txt)'))
    parser.add_argument(
        'EPOCH', type=int, help='epoch number')
    parser.add_argument(
        'PARAM_FILE', type=str,
        help='name of project parameter file')
    parser.add_argument(
        '--GPU_LST_STR', type=str,
        help='list of GPUs (separated by ":")', default='0')
    parser.add_argument(
        '--CV_NUM', type=int,
        help='CV number if do not want to start at 0', default=0)
    parser.add_argument(
        '--PATH_INP_BASE', type=str,
        help=('path to site pre processing folder as default (None) '
              + 'the path to 1_site_preproc is used'),
        default='None')

    args = parser.parse_args()

    import utils.pca_calc_utils as pca_calc_utils
    main(pca_calc_utils, vars(args), vars(args)['CV_NUM'])

