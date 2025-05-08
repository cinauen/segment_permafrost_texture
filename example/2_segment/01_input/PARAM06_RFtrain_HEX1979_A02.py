"""
----- Parameter file for Random Forest supervside classification -------
(for: ml_workflow/MAIN_RF_classify.py and MAIN_RF_classify_CV.py)

This parameter file defines the input parameters for model training
testing and predicting and hyperparameter tuning and feature importance
testing
It is used with the script MAIN_RF_classify.py and
MAIN_RF_classify_CV.py (loops through all cross validations)

The processing steps is defined by command line input and the parameters
are then configured accordingly:
- PARAM['PROC_STEP'] = 'rt': train, test, predict
- PARAM['PROC_STEP'] = 'gs': hyperparameter tuning (grid search)
- PARAM['PROC_STEP'] = 'fi': feature importance (shap analysis)


--------- options for feature preparation -------
The specific options on how the input features are prepared, is defined
in the parameter file PARAM['file_merge_param']
(param_settings/PARAM_inp_ML_feature_prep_v01.txt).
This file contains combinatins of different setups which can
be selected by the PARAM_PREP_ID (must be provided as command line input).
The following options are defined:
- which bands to use
- what type of standardisation or normalisation to use
(parameter description: docs/PARAM_options_feature_preparation_ML.md)


--------- options for training -------
The training options are defined in the file PARAM['file_train_param']
(param_settings/PARAM_inp_ML_train_v01.txt).
This file contains combinatins of different setups which can
be selected by the PARAM_TRAIN_ID (must be provided as command line input).
The following options are defined:
- Random Forest input parameters
(parameter description: docs/PARAM_options_training_ML.md)


The areas (AOI patchs), which should be included in the training, validation and
testing are defined with:
- PARAM['PATH_PREFIX_SITE_LST'] (path data input)
- PARAM['PHASE_NAME'], PARAM['PHASE_META_FILE']: defines which exact AOI
    should be used per phase
    Note: as input for the machine learning classification the
        untiled data can be used

!!! Note on data usage in this example:
    This example uses a limited amount of input data in order to reduce
    computation times and data size. Only four four test patches are used.
    Furthermore, due to commercial restrictions, the SPOT input data cannot provided.
    Thus, this example uses the freely downloadable Hexagon (KH-9PC) data only.
"""

import os
import sys
import numpy as np

import param_settings.param_utils as param_utils


def get_param(PARAM):
    '''
    Following are set on cmd:
        PARAM['PROC_STEP']: processing step ('rt', 'gs', 'fi')
        PARAM['PARAM_TRAIN_ID']: training options ID (e.g. tML02)
        PARAM['PARAM_PREP_ID']: merge options ID (e.g. vML001)
        PARAM['GPU_LST_STR']: 7:5  # list og GPUS to use (separated by ':')
    '''
    # =====  test patch limit !!!!
    # to keep the example small we use here only a limited number of
    # fine-tuning and test patches
    patch_limit = 4

    # run processes in parallel
    PARAM['PARALLEL'] = True

    # ---- Input parameter files specifying various setup options to be tested.
    # file with different setups for inpu data preparation
    PARAM['file_merge_param'] = 'PARAM_inp_ML_feature_prep_v01.txt'
    # file with different setups for training
    PARAM['file_train_param'] = 'PARAM_inp_ML_train_v01.txt'

    # -------------- Definition of processing steps -------------
    if PARAM['PROC_STEP']== 'rt':  # train, test, predict
        PARAM['load_model'] = False  # if False then training is run
        PARAM['run_test'] = True  # run test on test patches after training
        PARAM['predict'] = True  #  run prediction after training
        PARAM['hyper_param_tune'] = False  # hyperparameter tuning to get best training parameters
        PARAM['calc_feature_importance'] = False  # calculate shape feature importance
        PARAM['LOG_FILE_SUFFIX'] = 'train_pred_ML_model'
    elif PARAM['PROC_STEP'] == 'gs':  # grid search for hyperparameter tuning
        PARAM['load_model'] = False  # if False then training is run
        PARAM['run_test'] = False  # run test on test patches after training
        PARAM['predict'] = False  #  run prediction after training
        PARAM['hyper_param_tune'] = True  # hyperparameter tuning to get best training parameters
        PARAM['calc_feature_importance'] = False  # calculate shap feature importance
        PARAM['LOG_FILE_SUFFIX'] = 'hyper_param_tuning'
    elif PARAM['PROC_STEP'] == 'fi':
        PARAM['load_model'] = False  # if False then training is run
        PARAM['run_test'] = False  # run test on test patches after training
        PARAM['predict'] = False  #  run prediction after training
        PARAM['hyper_param_tune'] = False  # hyperparameter tuning to get best training parameters
        PARAM['calc_feature_importance'] = True  # calculate shap feature importance
        # to select best feature input combination
        PARAM['LOG_FILE_SUFFIX'] = 'feature_importance_analysis'

    # if load model choose correct start count
    PARAM['start_count'] = 1  # is starting from 1

    # ----------- Set output properties -----------
    # -- output folder
    PARAM['model_folder_base'] = f"BLyaE_v1_HEX1979_A02_ML_{PARAM['PROC_STEP']}"
    # -- output file prefix
    PARAM['FILE_PREFIX'] = f"{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}"
    # --  target coord syst
    PARAM['EPSG_TARGET'] = 32654

    # -------------- Define input data ------------------
    # Name identifier of different input data sets
    # they are used as keys to extract parameters from dictionaries further
    PARAM['FILE_PREFIX_SITE_LST'] = [
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit',
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit',
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit',
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit',
        #'BLyaE_SPOT2018_test_std4_8bit',
        ]

    # Folders to input data
    PARAM['PATH_PREFIX_SITE'] = {
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': os.path.normpath(
        #    'BLyaE_v1/03_train_inp/BLyaE_HEX1979_A01_perc0-2_g0-3_8bit_Lv01_untiled_v00'),  # for training
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': os.path.normpath(
            'BLyaE_v1/03_train_inp/BLyaE_HEX1979_A02_perc0-2_g0-3_8bit_Lv01_untiled_v00'),  # for training
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': os.path.normpath(
            'BLyaE_v1/03_train_inp/BLyaE_HEX1979_test_perc0-2_g0-3_8bit_Lv01_untiled_v00'),  # for test
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': os.path.normpath(
            'FadN_v1/03_train_inp/FadN_HEX1980_test_histm_b0-5aRv1_8bit_Lv01_untiled_v00'),  # for test
        #'BLyaE_SPOT2018_test_std4_8bit': os.path.normpath(
        #    'BLyaE_v1/03_train_inp/BLyaE_SPOT2018_test_std4_8bit_Lv01_untiled_v00'),  # for test on other site
        }

    # ---- Info used for naming and to extract correct parameters ----
    PARAM['SCALE_TYPE_dict'] = {
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': 'perc0-2_g0-3_8bit',
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': 'perc0-2_g0-3_8bit',
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': 'perc0-2_g0-3_8bit',
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': 'histm_b0-5aRv1_8bit',
        # 'BLyaE_SPOT2018_test_std4_8bit': 'std4_8bit',
        }

    # Prefix of tile names (used to find tile names)
    # !!! if no sub area add '_' at end e.g. 'BLyaE_HEX1979_' !!!
    PARAM['FILE_SITE_dict'] = {
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': 'BLyaE_HEX1979_A01',
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': 'BLyaE_HEX1979_A02',
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': 'BLyaE_HEX1979_',  # for distributed test patches do not add any AOI name here
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': 'FadN_HEX1980_',
        # 'BLyaE_SPOT2018_test_std4_8bit': 'BLyaE_SPOT2018_'
        }
    PARAM['SENSOR_TYPE_dict'] = {
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': 0,
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': 0,
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': 0,
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': 0,
        # 'BLyaE_SPOT2018_test_std4_8bit': 1,
        }

    # --- define phases and choice of AOIs per phase
    # phase names
    # Note: For the test phase also add site and year to the PHASE_NAME
    #       as the test could also be run on test patches form other
    #       sites/years
    PARAM['PHASE_NAME']  = [
        'train', 'validate',
        'BLyaE_HEX1979_test',
        'FadN_HEX1980_test',
        #'BLyaE_SPOT2018_test'
        ]

    # PARAM['PHASE_META_FILE'] describes the patches to be used per
    # phase. The list contains one item per cross validation.
    # (if do not use in CV mode then just first item in list is used)
    # The dictionary key corresponds to the PARAM['FILE_PREFIX_SITE_LST']
    # and the data sets in PARAM['PATH_PREFIX_SITE_LST']
    # The dict values contrain the patch names per phase
    # Thus for first CV trainng is done on ['train-01', 'train-02']
    # and validation on ['train-03']
    # The sets patches are described in a separate dict item as they use
    # a different dataset.
    # for shuffle could use instead:
    # or for shuffle use ['train-01', 'train-02', 'train-03'], ['shuffle']]
    PARAM['PHASE_META_FILE'] = [
        {  # ------------------- CV00 --------------------------
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': [
        #    ['train-01', 'train-02'], ['train-03'], []],  # []],
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': [
            ['train-01', 'train-02'], ['train-03'], [], []],  # []],
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': [
            [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)], []],  # []],
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': [
           [], [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        # 'BLyaE_SPOT2018_test_std4_8bit': [
        #     [], [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
            },
        {  # ------------------- CV01 --------------------------
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': [
        #    ['train-01', 'train-03'], ['train-02'], []],  # []],
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': [
            ['train-01', 'train-03'], ['train-02'], [], []],  # []],
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': [
            [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)], []],  # []],
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': [
            [], [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        # 'BLyaE_SPOT2018_test_std4_8bit': [
        #     [], [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
            },
        {  # ------------------- CV02 --------------------------
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': [
        #    ['train-02', 'train-03'], ['train-01'], []],  # []],
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': [
            ['train-02', 'train-03'], ['train-01'], [], []],  # []],
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': [
            [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)], []],  # []],
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': [
            [], [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        # 'BLyaE_SPOT2018_test_std4_8bit': [
        #     [], [], [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
            },
        ]

    # dictionary defining which stats subareas to take per file_id
    PARAM['STATS_FILE'] = {
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit':
        #    {
        #     'A01': os.path.normpath(
        #            'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
        #     'A02': os.path.normpath(
        #            'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A02REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt')
        #    },
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit':
            {
            # 'A01': os.path.normpath(
            #        'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
             'A02': os.path.normpath(
                    'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A02REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt')
            },
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':
            {
            # 'A01': os.path.normpath(
            #        'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
             'A02': os.path.normpath(
                    'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A02REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt')
            },
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit':
            {
            # 'A01': os.path.normpath(
            #        'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
             'A02': os.path.normpath(
                    'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A02REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt')
            },
        #'BLyaE_SPOT2018_test_std4_8bit':
        #    {
        #     'A01': os.path.normpath(
        #            'BLyaE_v1/02_pre_proc/BLyaE_SPOT2018_A01REPLACE_std4_8bit_P02_tex_stats_file.txt'),
        #     'A02': os.path.normpath(
        #            'BLyaE_v1/02_pre_proc/BLyaE_SPOT2018_A02REPLACE_std4_8bit_P02_tex_stats_file.txt')
        #     }
        }

    # for the PCA stats, which stats subareas to take per file_id
    PARAM['STATS_FILE_PCA'] = {}
    PARAM['PHASE_STATS_FILE'] = [
        {  # ---------------- CV00 ------------------
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit':
        #    {
        #        'A01': ['train-01', 'train-02'],
        #        'A02': ['train-01', 'train-02']
        #        },
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit':
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit':
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
        #'BLyaE_SPOT2018_test_std4_8bit':
        #    {
        #        'A01': ['train-01', 'train-02'],
        #        'A02': ['train-01', 'train-02']
        #        },
        },
        {  # ---------------- CV01 ------------------
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit':
        #    {
        #        'A01': ['train-01', 'train-03'],
        #        'A02': ['train-01', 'train-03']
        #        },
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit':
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit':
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
        #'BLyaE_SPOT2018_test_std4_8bit':
        #    {
        #        'A01': ['train-01', 'train-03'],
        #        'A02': ['train-01', 'train-03']
        #        },
         },
         {  # ---------------- CV02 ------------------
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit':
        #    {
        #        'A01': ['train-02', 'train-03'],
        #        'A02': ['train-02', 'train-03']
        #        },
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit':
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit':
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
        #'BLyaE_SPOT2018_test_std4_8bit':
        #    {
        #        'A01': ['train-02', 'train-03'],
        #        'A02': ['train-02', 'train-03']
        #        },
         },
        ]

    # ---- specify class labels ---
    # Class labels need to be in incrementing order (range(n_classes)) since pytorch
    # uses indices for class labels
    # Dictionary to relabel classes if required (from : to)
    PARAM['DICT_RELABEL'] = {}

    # -- mask certain classes to nan (for testing)
    # Option to mask (neglect) certain labels by setting them to background "0".
    # Also the corresponding pixels in the imagery are set to nan.
    # This is done on the relabeled data.
    PARAM['MASK_TO_NAN_LST'] = []  # list the classes numbers
    # which sould be set to nan (AFTER relabeling)

    # Class number and corresponding label AFTER relabelling
    # !!! the indices need to be in order !!!! since pytorch
    # uses indices for class labels (otherwise would need to use
    # PARAM['DICT_RELABEL'])
    PARAM['CLASS_LABELS'] = [
        [0, 1, 2, 3, 4, 5, 6],
        ['nan', 'baydherakhs', 'ridges_ponds', 'stable_areas',
         'gully_base', 'ponds', 'snow']]

    # color scales per label
    PARAM['dict_assign'] = {
        1: ['#FFEA86', 'baydherakhs'],
        2: ['#882255', 'ridges_ponds'],
        3: ['#A7A3A6', 'undisturbed'],
        4: ['#2330D0', 'gully_base'],
        5: ['#A2F6FB', 'ponds'],
        6: ['#B157F5', 'snow']}

    # ------ evaluation ----
    # For which class to evaluate TP TN per patch weight
    # here use only class 1 (=baydzherakhs) since there are only patch certainty weights
    # available from this class
    PARAM['class_eval_weighted_TP_TN_lst'] = [1]

    # --- parameters for grid search and feature importance testing
    # general parameters for hyperparameter tuning and feature importance testing
    PARAM['param_dict'] = [
        {
        'tune_type': 'grid_search',  # 'grid_search' or 'randomized_search'
        'scoring': ['accuracy', 'recall_macro',
                    'jaccard_macro', 'f1_macro',
                    'jaccard_micro', 'f1_micro'],  # can be None, single string or list
        'refit_inp': 'jaccard_macro',  # or can be True (if scoring is not list)
        'gs_split': 'as_train',  # with "as train" use similar split as
        # for training (train and test set are merged for this).
        # other option is 'shuffle' or None. With None just default CV
        # with fold defined by cv_num below is used
        'cv_num': 3,
        'shap_sample_num': [5000, 2500]  # amount of samples to be used
        # for shap feature importance analysis
        },
        ]

    # parameter ranges to be tested in hyterparameter tuning
    # if set to None then default parameters are used
    # (see ml_classification_utils.set_classification_param_gs).
    # Otherwise could e.g. define
    # {'RandomForest': {'n_estimators': [500, ...]}}
    PARAM['algo_param_gs'] = {
        'RandomForest': {
            'n_estimators': [10, 100],  # cum 100 is default
            'max_depth': [5, 10],  # cuml: 16 is default
            }
        }

    # Bands of input data file. This is used when opening tiles.
    # Which GLCM bands to use is specified in PARAM['file_merge_param'])
    PARAM['X_BANDS'] = None  # if set to None then takes all bands from meta file
    # bands of in label files
    PARAM['Y_BANDS'] = None  # if set to None then takes all bands from meta file
    # (however weights are not taken into account for the random forest
    # training)

    PARAM['BIT_DEPTH'] = 8

    # ----- read parameters from list ---
    # input feature selection and preparation parameters taken from 1B_proc\2_segment\01_input
    param_utils.update_MLmerge_param_from_file(PARAM)

    # training parameter taken from 1B_proc\2_segment\01_input
    param_utils.update_MLtrain_param_from_file(PARAM)

    # ---- selection of GLCM input features as sepcified in the
    # merge paramter file (PARAM['file_merge_param']) and
    # defined by the band names (PARAM['merge_bands'])
    # and the GLCM window sizes and directions (PARAM['file_suffix_lst'])
    # Note: the greyscale band ['1'] is excluded here as it is contained
    #       per default
    PARAM['add_bands'] = [np.setdiff1d(PARAM['merge_bands'], ['1']).tolist()]*len(PARAM['file_suffix_lst'])

    # ---- normalization param
    if PARAM['if_norm_min_max_band1']:
        # if normalize grey scale band need to specify the min max
        # depending on bit depth
        PARAM['norm_min_max_band1'] = [0, 2**PARAM['BIT_DEPTH'] - 1]
    else:
        PARAM['norm_min_max_band1'] = None

    return


