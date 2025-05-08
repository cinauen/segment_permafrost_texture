"""
-- Parameter file for model prediction --
(for: cnn_workflow/MAIN_predict.py)

This parameter file defines the input data on which the prediction should be run.
Also it specifies which model should be used (model trained on a specific area).
The segmentation framework option (e.g. 'v079', 't16onl') as well as the epoch however are
defined as command line input.

In this example the model trained on the HEX1979 data from BLyaE used to predict the degradation
classes on the full area of BLyaE.

"""

import os
import numpy as np
from torch import nn
import torch


import param_settings.param_utils as param_utils


def get_param(PARAM):
    """
    The following are set on command line and are given by the PARAM input here:
    - PARAM['PARAM_TRAIN_ID']: key to define data preparation options (e.g. 'v079')
    - PARAM['PARAM_PREP_ID']: key to define training setup (e.g. 't16onl')
    - PARAM['EPOCH']: epoch number which to use for preditions (e.g. 44)
    - PARAM['GPU_LST_STR']: list of GPUs to use for training (separated by ':', e.g. '5:7')
    - ...
    """
    # if should create additional plots
    PARAM['extended_output'] = True

    # ---- Input parameter files specifying various setup options to be tested ---
    # file with different setups for inpu data preparation
    PARAM['file_merge_param'] = 'PARAM_inp_CNN_feature_prep_v01.txt'
    # file with different setups for training
    PARAM['file_train_param'] = 'PARAM_inp_CNN_train_v01.txt'

    # ------- Set input model to be used for prediction -------
    # folder to to input model (is also used as second subfolder for outptut prediction)
    PARAM['model_folder_base'] = 'BLyaE_v1_HEX1979_A02'
    # !!! CV_NUM and epoch are specified in function input
    # also the corresponding EPOCH is defined in function input
    # Path to model is:
    # f'...\2_segment\02_train\{PARAM['model_folder_base']}\{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}_{PARAM['PARAM_CV_SUFFIX']}'
    PARAM['file_out_prefix_add'] = ''  # only for output folder

    # ---------------- Set output options ----------------
    # - output folder for prediction
    # scale here corresponds to scaling of image to be predicted (not of model scaling)
    pred_folderHEX1979 = 'BLyaE_v1_pred_HEX1979_perc0-2_g0-3_8bit'
    # pred_folderSPOT2018 = 'BLyaE_v1_SPOT2018_std4_8bit'
    # full folder structure:
    # f'...\2_segment\04_predict\{pred_folderHEX}\{PARAM['model_folder_base']}\cvXX

    # target coordinate system
    PARAM['EPSG_TARGET'] = 32654

    # aoi to clip final prediction image after merge (no clipping is done if set to none)
    PARAM['FINAL_CLIP_AOI'] = 'BLyaE_prediction_AOI_32654_reduced.geojson'

    # ------------------ Prediction type ------------------
    # either can do prediction on tile (this is default that should be done)
    # otherwise can run prediction on an entire image
    PARAM['prediction_type'] = 'tiles'
    # !!! if want to save and merge probabilities !!! this is only important for tiles
    # otherwise probability is saved anyway
    PARAM['save_proba'] = True

    # ------------------ Prediction input ------------------
    if PARAM['prediction_type'] == 'tiles':
        # --- input for prediction on tiles --
        # path to folder with tiles to be predicted
        # is dict with one dict-item per perdiction imagery
        # item value is [path_to_tiles, prefix_of_tiles]
        # (prefix is used to search tiles and for output name)
        PARAM['PATH_PRED'] = {
            pred_folderHEX1979: [
                os.path.normpath(os.path.join(
                    'BLyaE_v1', '03_train_inp', f'BLyaE_HEX1979_all_perc0-2_g0-3_8bit_w298-298_v00')),
                'BLyaE_HEX1979_all'],
            #pred_folderSPOT2018: [
            #    os.path.normpath(os.path.join(
            #        'BLyaE_v1', '03_train_inp', 'BLyaE_SPOT2018_all_std4_8bit_w298-298_v00')),
            #    'BLyaE_HEX1979_all'],
            }
        PARAM['window_size'] = 256
    else:
        # -- input for prediction on whole image --
        # path to file to be predicted
        # is dict with one dict-item per perdiction imagery
        # item value is [path_to_file, prefix_of_file]
        # (prefix is used to search file and for output name)
        PARAM['PATH_PRED'] = {
            pred_folderHEX1979: [os.path.normpath(os.path.join(
                'BLyaE_v1', '02_pre_proc', 'BLyaE_HEX1979_scale_perc0-2_g0-3_8bit.tif')),
                'BLyaE_HEX1979'],
            #pred_folderSPOT2018: [os.path.normpath(os.path.join(
            #    'BLyaE_v1', '02_pre_proc', 'BLyaE_SPOT2018_scale_std4_8bit.tif'))],
            }
        # if predict on whole image. Dict input is path to file on which do prediction
        PARAM['window_size'] = None

    # specify sensor type. Is optionally used for adding sensor type as additional
    # input feature in training. Or for sensor specific augmentation e.g. FAD
    # (this is currently not included in this script version)
    PARAM['FILE_SENSOR_dict'] = {
        pred_folderHEX1979: 'HEX'
        }

    # --- Define stats-files for training-area-specific standardisation
    # If training area specific standardisation was used during training,
    # then same should be used during prediction.
    PARAM['STATS_FILE'] = {
        pred_folderHEX1979:
            {
             #'A01': os.path.normpath(
             #      'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
             'A02': os.path.normpath(
                    'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A02REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt')
            },
        #pred_folderSPOT2018:
        #    {'A01': os.path.normpath(
        #            'BLyaE_v1/02_pre_proc/BLyaE_SPOT2018_A01REPLACE_std4_8bit_P02_tex_stats_file.txt'),
        #     'A02': os.path.normpath(
        #            'BLyaE_v1/02_pre_proc/BLyaE_SPOT2018_A02REPLACE_std4_8bit_P02_tex_stats_file.txt')
        #    },
        }

    # ----- Define the correct stats values to be used depending on CV input
    PARAM['PHASE_STATS_FILE'] = [
        {  # -------------------- CV00 --------------------
        pred_folderHEX1979:
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
        #pred_folderSPOT2018:
        #    {
        #        'A01': ['train-01', 'train-02'],
        #        'A02': ['train-01', 'train-02']
        #        },
        },
        {  # -------------------- CV00 --------------------
        pred_folderHEX1979:
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
        #pred_folderSPOT2018:
        #    {
        #        'A01': ['train-01', 'train-03'],
        #        'A02': ['train-01', 'train-03']
        #        },
        },
        {  # -------------------- CV00 --------------------
        pred_folderHEX1979:
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
        #pred_folderSPOT2018:
        #    {
        #        'A01': ['train-02', 'train-03'],
        #        'A02': ['train-02', 'train-03']
        #        },
        },
        ]


    # Bands of input data file. This is used when opening tiles.
    # Which GLCM bands to use is specified in PARAM['file_merge_param'])
    PARAM['X_BANDS'] = None  # if set to None then all bands are used

    # ---------- Class label definitions ---------
    # this is still required as need amount of classes !
    PARAM['CLASS_LABELS'] = [
        [0, 1, 2, 3, 4, 5, 6],
        ['nan', 'baydherakhs', 'ridges_ponds', 'stable_areas',
         'gully_base', 'ponds', 'snow']]

    # ----------- Read parameters from list --------
    # input feature selection and preparation parameters taken from 1B_proc\2_segment\01_input
    param_utils.update_merge_param_from_file(PARAM)

    # training parameter taken from 1B_proc\2_segment\01_input
    param_utils.update_train_param_from_file(PARAM)


    # ----------- Performance options -----------
    # Overwrite the batch number to make prediction more efficient.
    # It does not impact prediction result.
    if PARAM['prediction_type'] == 'tiles':
        # when using tiles it is possible to use many workers
        # and batches due to the small tile size
        PARAM['N_BATCH'] = 20
        PARAM['N_WORKERS'] = 50
    else:
        # for the entile image prediciton batches and workers need to
        # be choosen such that it fits into memory
        PARAM['N_BATCH'] = 5
        PARAM['N_WORKERS'] = 8

    # ---- Selection of GLCM input features as sepcified in the PARAM['file_merge_param']
    PARAM['add_bands'] = [PARAM['merge_bands']]*len(PARAM['file_suffix_lst'])

    # ---- Normalization param
    if PARAM['if_norm_min_max_band1']:
        PARAM['norm_min_max_band1'] = [0, 2**PARAM['BIT_DEPTH'] - 1]  # if normalize grey scale band.
    else:
        PARAM['norm_min_max_band1'] = None

    return