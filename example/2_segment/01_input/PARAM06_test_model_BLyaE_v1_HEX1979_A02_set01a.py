"""
--------- Parameter file for model evaluation on the test patches --------
(for: cnn_workflow/MAIN_evaluate_model.py and
      cnn_workflow/MAIN_evaluate_model_CV.py)

This parameter file defines the input parameters for testing a model that was trained with:
- MAIN_train_incl_aug.py, MAIN_train_incl_aug_CV.py,
    MAIN_train_offl_aug.py, MAIN_train_offl_aug_CV.py

The model to be tested is selected based on:
- the model folder PARAM['model_folder_base']
- the merge options (provided as command line input): PARAM['PARAM_PREP_ID']
- the training options (provided as command line input): PARAM['PARAM_TRAIN_ID']
Note: all checkpoints are tested (thus epochs do not need to be specified)

The test patches which should be included in the testing are defined with:
- PARAM['PATH_PREFIX_SITE_LST'] (path to test patch input)
    Note: also test patches of sensors or sites wich were not included in the
        training can be selected (to test model transferability)
- PARAM['PHASE_NAME'], PARAM['PHASE_META_FILE']: defines which exact test patches
    should be used per test phase

Optional TODOs:
- automatically set bit depth according to input files !!!
    this is required for augmentation range if augment data and calc GLCM in training loop
    extract bit depth and other param from PARAM['PATH_PREFIX_SITE_LST']

!!! Note on data usage in this example:
    This example uses a limited amount of input data in order to reduce
    computation times and data size. Only four four test patches are used.
    Furthermore, due to commercial restrictions, the SPOT input data cannot provided.
    Thus, this example uses the freely downloadable Hexagon (KH-9PC) data only.
"""

import os
import numpy as np
from torch import nn
import torch

import param_settings.param_utils as param_utils
import cnn_workflow.cnn_workflow.custom_augmentation as custom_augmentation

def get_param(PARAM):
    """
    The following are set on command line and are given by the PARAM input here:
    - PARAM['PARAM_PREP_ID']: key to define data preparation options (e.g. 'v079')
    - PARAM['PARAM_TRAIN_ID']: key to define training setup (e.g. 't16onl')
    - PARAM['GPU_LST_STR']: list of GPUs to use for training (separated by ':', e.g. '5:7')
    - PARAM['GPU_num_GLCM']: GPU to use for calculating GLCM (GLCM calculation is run as subprocess, e.g. 6)
    Note: epoch number is not required as an input here as all checkpoints are tested
    """
    # =====  test patch limit !!!!
    # to keep the example small we use here only a limited number of
    # fine-tuning and test patches
    patch_limit = 4

    # -- if want additional plots
    PARAM['extended_output'] = True

    # ---- Input parameter files specifying various setup options to be tested.
    # file with different setups for inpu data preparation
    PARAM['file_merge_param'] = 'PARAM_inp_CNN_feature_prep_v01.txt'
    # file with different setups for training
    PARAM['file_train_param'] = 'PARAM_inp_CNN_train_v01.txt'

    # ---------------- Set output options ----------------
    # -- --output naming (prefixes and suffixes) -----
    # path to folder with model and for placing test output
    PARAM['model_folder_base'] = 'BLyaE_v1_HEX1979_A02'
    # prefix for model input and file output (for output phase name will be added)
    PARAM['FILE_PREFIX'] = f"{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}"
    PARAM['file_prefix_add'] = ''

    # target coordinate system
    PARAM['EPSG_TARGET'] = 32654  # target coord syst


    # --------- Define input data ---------
    # Name identifier of different input data sets
    # they are used as keys to extract parameters from dictionaries further
    PARAM['FILE_PREFIX_SITE_LST'] = [
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit',
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit',  # test on other site
        #  'BLyaE_SPOT2018_test_std4_8bit',  # if test should also be run on SPOT
        ]

    # paths to input test patches
    PARAM['PATH_PREFIX_SITE'] = {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': os.path.normpath(
            'BLyaE_v1/03_train_inp/BLyaE_HEX1979_test_perc0-2_g0-3_8bit_Lv01_untiled_v00'),
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': os.path.normpath(
            'FadN_v1/03_train_inp/FadN_HEX1980_test_histm_b0-5aRv1_8bit_Lv01_untiled_v00'),
        # 'BLyaE_SPOT2018_test_std4_8bit': os.path.normpath(
        #    'BLyaE_v1/03_train_inp/BLyaE_SPOT2018_test_std4_8bit_Lv01_untiled_v00'),
        }

    # --- Define stats-files for training-area-specific standardisation
    # Define the stats-files from which the values for training-area-specific
    # standardisation should be extracted. Separate values are used for
    # the Hexagon and the SPOT data. For the different training areas
    # (e.g. A01 and A02), the min max or average are used.The "REPLACE"
    # in the filename will be replaced by the sub-training area for the
    # specific training area per cross-validation run.
    PARAM['STATS_FILE'] = {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':
            {
             #'A01': os.path.normpath(
             #   'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
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
        # 'BLyaE_SPOT2018_test_std4_8bit':
        #     {'A01': os.path.normpath(
        #         'BLyaE_v1/02_pre_proc/BLyaE_SPOT2018_A01REPLACE_std4_8bit_P02_tex_stats_file.txt'),
        #      'A02': os.path.normpath(
        #         'BLyaE_v1/02_pre_proc/BLyaE_SPOT2018_A02REPLACE_std4_8bit_P02_tex_stats_file.txt')
        #     },
        }

    # ---- Info used for naming and to extract correct parameters ----
    # dictionary to extract scale type. Used for texture calculation within training
    PARAM['SCALE_TYPE_dict'] = {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': 'perc0-2_g0-3_8bit',
        # 'BLyaE_SPOT2018_test_std4_8bit': 'std4_8bit'
        }

    # Prefix of tile names (used to find tile names)
    # !!! if no sub area add '_' at end e.g. 'BLyaE_HEX1979_' !!!
    PARAM['FILE_SITE_dict'] = {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': 'BLyaE_HEX1979_',
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': 'FadN_HEX1980_',
        # 'BLyaE_SPOT2018_test_std4_8bit': 'BLyaE_SPOT2018_',
        }

    # specify sensor type. Is used for adding sensor type as additional
    # input feature in training. Or for sensor specific augmentation e.g. FAD
    # (this is currently not included in this script version)
    PARAM['FILE_SENSOR_dict'] = {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': 'HEX',
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': 'HEX',
        # 'BLyaE_SPOT2018_test_std4_8bit': 'SPOT',
        }

    # ------------ Define test phases -------------
    # here for test add also site and year to phase name as can also
    # use test patches form other sites/years
    PARAM['PHASE_NAME']  = ['BLyaE_HEX1979_test',
                            'FadN_HEX1980_test',
                            # 'BLyaE_SPOT2018_test',
                            ]
    n_phase = len(PARAM['PHASE_NAME'])

    # ---- Dictionary to define the correct test patches to use per CV and site --
    # if run in CV mode (with MAIN_evaluate_model_CV.py) the script
    # will cycle through the list with different combinations below.
    # If do not use in CV mode (MAIN_evaluate_model.py) then just first
    # list item is used. However, for test, the inputs for all CV runs are the same.
    # Thus all list items are the same below.
    PARAM['PHASE_META_FILE'] = [
        {  # -------------------- CV00 --------------------
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': [
            [f'test-{x:02d}' for x in range(1, patch_limit + 1)], []],  # []], # !!! for 'BLyaE_SPOT2018_test' would need to add a emtpy list item
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': [
           [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        # 'BLyaE_SPOT2018_test_std4_8bit': [
        #     [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        },
        {  # -------------------- CV01 --------------------
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': [
            [f'test-{x:02d}' for x in range(1, patch_limit + 1)], []],  # []],
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': [
           [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        # 'BLyaE_SPOT2018_test_std4_8bit': [
        #     [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        },
        {  # -------------------- CV02 --------------------
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': [
            [f'test-{x:02d}' for x in range(1, patch_limit + 1)], []],  # []],
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit': [
           [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        # 'BLyaE_SPOT2018_test_std4_8bit': [
        #     [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        },
        ]

    PARAM['PHASE_STATS_FILE'] = [  # which stats subareas to take per file_id
        {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':  # --- CV1
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit':
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
        # 'BLyaE_SPOT2018_test_std4_8bit':
        #     {
        #         'A01': ['train-01', 'train-02'],
        #         'A02': ['train-01', 'train-02']
        #         },
        },
        {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':  # --- CV2
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit':
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
        # 'BLyaE_SPOT2018_test_std4_8bit':
        #     {
        #         'A01': ['train-01', 'train-03'],
        #         'A02': ['train-01', 'train-03']
        #         },
        },
        {
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit':  # --- CV3
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit':
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
        # 'BLyaE_SPOT2018_test_std4_8bit':
        #     {
        #         'A01': ['train-02', 'train-03'],
        #         'A02': ['train-02', 'train-03']
        #         },
        },
        ]


    # Bit deth should be specified according to scaling
    # (TODO: could extract if it from PARAM['SCALE_TYPE_dict'] within data_loader,
    # thus, define depending on input file, e.g. in data loa
    PARAM['BIT_DEPTH'] = 8

    # Bands of input data file. This is used when opening tiles.
    # Which GLCM bands to use is specified in PARAM['file_merge_param'])
    PARAM['X_BANDS'] = None  # if set to None then takes all bands from meta file
    # bands of in label files
    PARAM['Y_BANDS'] = None  # if set to None then takes all bands from meta file

    # -------- Class label options -------
    # -- relabel classes is required
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

    # ------ evaluation ----
    # For which class to evaluate TP TN per patch weight
    # here use only class 1 (=baydzherakhs) since there are only patch certainty weights
    # available from this class
    PARAM['class_eval_weighted_TP_TN_lst'] = [1]

    # ---------- Read parameters from list --------
    # input feature selection and preparation parameters taken from 1B_proc\2_segment\01_input
    param_utils.update_merge_param_from_file(PARAM)

    # training parameter taken from 1B_proc\2_segment\01_input
    param_utils.update_train_param_from_file(PARAM)

    # ---------- Training options -----
    PARAM['window_size'] = None  # can be set to none as test patches have already
    # approx size of test patches
    PARAM['N_WORKERS'] = 6  # due to low amount of tiles and if use no GLCMs
    # can use higher number of workers
    PARAM['N_BATCH'] = 4 #!!!! adjust to test tile number

    # --- define model epochs for which test should be run:
    PARAM['SINGLE_EPOCH_LOAD'] = None  # rund for a single epoch
    # if single epoch is None then MIN_EPOCH is used (if not none)
    PARAM['MIN_EPOCH'] = 10  # use epochs bigger than (only those
    # with checkpoints) if min epoch is set to None then five epochs with
    # best metrics are taken

    # ------ Metadata file definition ---
    PARAM['meta_suffix_lst'] = ['meta_data']*n_phase

    # --- to query input data (metadata)
    # use only tiles where there are less than 50% NaNs
    PARAM['query_text'] = "`perc-class_0` <= 50"
    # !!! ` are required due to minus in header

    # If load image tiles based on metadata file.
    PARAM['load_phase_separate'] = True

    # ---- Selection of GLCM input features
    # as sepcified in the PARAM['file_merge_param']
    PARAM['add_bands'] = [
        PARAM['merge_bands']]*len(PARAM['file_suffix_lst'])


    # ---- Normalization param
    if PARAM['if_norm_min_max_band1']:
        PARAM['norm_min_max_band1'] = [0, 2**PARAM['BIT_DEPTH'] - 1]  # if normalize grey scale band.
    else:
        PARAM['norm_min_max_band1'] = None

    return