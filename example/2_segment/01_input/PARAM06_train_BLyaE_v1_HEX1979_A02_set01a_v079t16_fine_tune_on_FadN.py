"""
--------- Parameter file for fine tuning --------
(for: cnn_workflow/MAIN_train_incl_aug.py, MAIN_train_incl_aug_CV.py)

This parameter files defines which data (imagery type, site and training
or validation AOIs) to use for model fine tuning. According to this, it creates specific
output folder and file prefixes.

In this example the model trained on the HEX1979 data from BLyaE is fine-tuned with data from FadN
to improve the prediction on FadN.

The fine tuning is run with the same MAIN script as the training
(cnn_workflow/MAIN_train_incl_aug.py, MAIN_train_incl_aug_CV.py).
However, the pre-trained model is read in as specified by the PARAM['model_input']

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
        PARAM['PARAM_PREP_ID']: key to define data preparation options (e.g. 'v079')
        PARAM['PARAM_TRAIN_ID']: key to define training setup (e.g. 't16onl')
        PARAM['GPU_LST_STR']: list of GPUs to use for training (separated by ':', e.g. '5:7')
        PARAM['GPU_num_GLCM']: GPU to use for calculating GLCM (GLCM calculation is run as subprocess, e.g. 6)
    """

    # ===== fine-tuning and test patch limit !!!!
    # to keep the example small we use here only a limited number of
    # fine-tuning and test patches
    patch_limit = 4

    # -- If should create additional plots
    PARAM['extended_output'] = True

    # ---- Input parameter files specifying various setup options to be tested.
    # file with different setups for inpu data preparation
    PARAM['file_merge_param'] = 'PARAM_inp_CNN_feature_prep_v01.txt'
    # file with different setups for training
    PARAM['file_train_param'] = 'PARAM_inp_CNN_train_v01.txt'

    # ----- Set output naming (prefixes and suffixes) -----
    # output folder for model and training output
    # !!! v079t16 in the naming here refers to paramter set of input model
    PARAM['model_folder_base'] = 'BLyaE_v1_HEX1979_A02_v079t16_finet_FadN'
    # prefix for ouput files
    PARAM['FILE_PREFIX'] = f"{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}"
    PARAM['file_prefix_add'] = ''

    # ------- Define input model (pre-trained model) -------
    # The parameter sets (e.g. v079, t16onl) of the pre-trained model
    # can be choosen differently to the current training parameters.
    # With the follwoing dict, the correct foldernames are created automatically
    # for the specific CV_NUM (in param_utils.setup_model_input_properties()).
    # However, for non CV mode they can instead be provided directly with
    # with PARAM['model_input'] = {'PATH_PROC_inp': full_path_to_model_subfolder, 'file_prefix': model_file_prefix}
    PARAM['model_input'] = {
        'PATH_PROC_inp': None,
        'file_prefix': None,
        'PARAM_PREP_ID': 'v079',
        'PARAM_TRAIN_ID': 't16onl',
        'model_folder_base': 'BLyaE_v1_HEX1979_A02',
        'inp_epoch': [43, 72, 54],  # list if want to use different CV of pre trained model
        }

    # --------- Define input data ---------
    PARAM['adjust_file_num_to_batch'] = True  # if adjust amount of tiles
    # such that fill up all batches

    # Name identifier of different input data sets
    # they are used as keys to extract parameters from dictionaries further
    # Note: The data are read in according to the order of this list (per phase)
    #       Thus, in case tile numbers are adjusted to batch size
    #       PARAM['adjust_file_num_to_batch'] then important input data should
    #       be placed at the beginning to not being cut-off
    PARAM['FILE_PREFIX_SITE_LST'] = [
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft',  # fine-tuning set
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t',  # validation set
        # 'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit',  # train set kept in fine-tuning
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit',  # train set kept in fine-tuning
        ]

    # Folders to input data
    PARAM['PATH_PREFIX_SITE'] = {
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft': os.path.normpath(
            'FadN_v1/03_train_inp/FadN_HEX1980_fine_tune_histm_b0-5aRv1_8bit_Lv01_untiled_v00'),
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t': os.path.normpath(
            'FadN_v1/03_train_inp/FadN_HEX1980_test_histm_b0-5aRv1_8bit_Lv01_untiled_v00'),
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': os.path.normpath(
        #    'BLyaE_v1/03_train_inp/BLyaE_HEX1979_A01_perc0-2_g0-3_8bit_Lv01_w298-298_v00_rot_cubic'),
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': os.path.normpath(
            'BLyaE_v1/03_train_inp/BLyaE_HEX1979_A02_perc0-2_g0-3_8bit_Lv01_w298-298_v00'),
        }

    # --- Define stats-files for training-area-specific standardisation
    # Define the stats-files from which the values for training-area-specific
    # standardisation should be extracted. Separate values are used for
    # the Hexagon and the SPOT data. For the different training areas
    # (e.g. A01 and A02), the min max or average are used. The "REPLACE"
    # in the filename will be replaced by the sub-training area for the
    # specific training area per cross-validation run.
    PARAM['STATS_FILE'] = {
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft':
            {
            # 'A01': os.path.normpath(
            #        'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
             'A02': os.path.normpath(
                    'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A02REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt')
            },
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t':
            {
            # 'A01': os.path.normpath(
            #        'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A01REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt'),
             'A02': os.path.normpath(
                    'BLyaE_v1/02_pre_proc/BLyaE_HEX1979_A02REPLACE_perc0-2_g0-3_8bit_P02_tex_stats_file.txt')
            },
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
        }

    # ---- Info used for naming and to extract correct parameters ----
    # dictionary to extract scale type. Used for texture calculation within training
    PARAM['SCALE_TYPE_dict'] = {
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft': 'histm_b0-5aRv1_8bit',
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t': 'histm_b0-5aRv1_8bit',
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': 'perc0-2_g0-3_8bit',
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': 'perc0-2_g0-3_8bit',
        }
    # Prefix of tile names (used to find tile names)
    # !!! if no sub area add '_' at end e.g. 'BLyaE_HEX1979_' !!!
    PARAM['FILE_SITE_dict'] = {
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft': 'FadN_HEX1980_',
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t': 'FadN_HEX1980_',
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': 'BLyaE_HEX1979_A01',
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': 'BLyaE_HEX1979_A02',
        }

    # specify sensor type. Is used for adding sensor type as additional
    # input feature in training. Or for sensor specific augmentation e.g. FAD
    # (this is currently not included in this script version)
    PARAM['FILE_SENSOR_dict'] = {
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft': 'HEX',
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t': 'HEX',
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': 'HEX',
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': 'HEX',
        }

    # ------ Define phases and choice of AOIs per phase -----
    # define phases (here training and validation)
    # It is also possible to set 'test' here as well. However, this
    # would require an additional dictionary key (and PARAM['PATH_PREFIX_SITE']
    # to get the test patches input)
    PARAM['PHASE_NAME']  = ['train', 'validate']
    n_phase = len(PARAM['PHASE_NAME'])

    # ---- Dictionary to define the correct training and validation areas per CV --
    # here specify just one list item since do not run fine tuning in cross-validation
    # mode
    PARAM['PHASE_META_FILE'] = [
        {  # -------------------- CV00 --------------------
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft': [
           [f'fine_tune-{x:02d}' for x in range(1, patch_limit + 1)], []],
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t': [
           [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': [
        #   ['train-01', 'train-02'], ['train-03']],
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': [
           ['train-01', 'train-02'], ['train-03']],
           },
        {  # -------------------- CV01 --------------------
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft': [
           [f'fine_tune-{x:02d}' for x in range(1, patch_limit + 1)], []],
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t': [
           [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': [
        #   ['train-01', 'train-03'], ['train-02']],
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': [
           ['train-01', 'train-03'], ['train-02']],
           },
        {  # -------------------- CV02 --------------------
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft': [
           [f'fine_tune-{x:02d}' for x in range(1, patch_limit + 1)], []],
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t': [
           [], [f'test-{x:02d}' for x in range(1, patch_limit + 1)]],
        #'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': [
        #   ['train-02', 'train-03'], ['train-01']],
        'BLyaE_HEX1979_A02_perc0-2_g0-3_8bit': [
           ['train-02', 'train-03'], ['train-01']],
           },
        ]

    # -- Dictionary defining which stats sub-areas to take per file_id
    PARAM['PHASE_STATS_FILE'] = [  # whist stats subareas to take per file_id
        {  # -------------------- CV00 --------------------
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft':
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t':
            {
                #'A01': ['train-01', 'train-02'],
                'A02': ['train-01', 'train-02']
                },
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
        },
        {  # -------------------- CV01 --------------------
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft':
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t':
            {
                #'A01': ['train-01', 'train-03'],
                'A02': ['train-01', 'train-03']
                },
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
        },
        {  # -------------------- CV02 --------------------
        'FadN_HEX1980_histm_b0-5aRv1_8bit_ft':
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
        'FadN_HEX1980_histm_b0-5aRv1_8bit_t':
            {
                #'A01': ['train-02', 'train-03'],
                'A02': ['train-02', 'train-03']
                },
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
        },
        ]

    # Bit depth should be specified according to scaling
    # (TODO: could extract if it from PARAM['SCALE_TYPE_dict'] within data_loader,
    # thus, define depending on input file, e.g. in data loader with
    # int(scale_type.split('_')[-1].split('bit')[0]))
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

    # ---------- Read parameters from list --------
    # input feature selection and preparation parameters taken from 1B_proc\2_segment\01_input
    param_utils.update_merge_param_from_file(PARAM)

    # training parameter taken from 1B_proc\2_segment\01_input
    param_utils.update_train_param_from_file(PARAM)

    # ---------- Training options -----
    PARAM['window_size'] = 256  # or 128
    PARAM['N_EPOCHS'] = 50
    PARAM['N_WORKERS'] = 16  # 6
    PARAM['early_stopping'] = False  # if stop iterations after do not get any
    # improvements anymore (but anyway only models with imrovement are saved)
    PARAM['epoch_save_min'] = 2  # minimum epoch for which to save model

    # ------ Metadata file definition ---
    if 'onl' in PARAM['PARAM_TRAIN_ID'] or 'incl_aug' in PARAM['PARAM_TRAIN_ID']:
        # on-the-fly augmentation use non-augmentated input tiles
        PARAM['meta_suffix_lst'] = ['meta_data']*n_phase
    else:
        # if use offline augmentation use
        # (PARAM['aug_vers']specifies augmentation type and comes from
        # training parameter option file)
        PARAM['meta_suffix_lst'] = [
            'meta_data_augment_Av' + str(PARAM['aug_vers'])] + ['meta_data']*(n_phase - 1)

    # --- to query input data (metadata)
    # use only tiles where there are less than 50% NaNs
    PARAM['query_text'] = "`perc-class_0` <= 50"  # to query metadata
    # !!! ` are required due to minus in header

    # If load image tiles for different phases separayely.
    # Like this can make sure that there are no intersecting tiles between phase
    PARAM['load_phase_separate'] = True

    # -------- Augmentation parameters --------
    if 'off' in PARAM['PARAM_TRAIN_ID'] or 'offl_aug' in PARAM['PARAM_TRAIN_ID']:
        # for offline augment uses n amount augmentations (n_augment)
        # and orig dataset.
        # Here define which augmentation numbers to use (according number of augmentations)
        PARAM['augment_file_query'] = f'aug_num <= {PARAM["n_augment"]}'

    # run setup of aumgnetation parameters
    param_utils.get_augment_param(PARAM)

    if 'onl' in PARAM['PARAM_TRAIN_ID'] or 'incl_aug' in PARAM['PARAM_TRAIN_ID']:
        # parameters fro on the fly augmentation
        PARAM['BIT_DEPTH_aug_inp'] = 2**PARAM['BIT_DEPTH'] - 2
        # need to convert int from geyscale to float for color augmentation
        PARAM['aug_range'] = [
            custom_augmentation.convert_to_float(PARAM['BIT_DEPTH_aug_inp']),
            custom_augmentation.revert_to_int(PARAM['BIT_DEPTH_aug_inp'])]

    # ---- Selection of GLCM input features as sepcified in the
    # merge paramter file (PARAM['file_merge_param']) and
    # defined by the band names (PARAM['merge_bands'])
    # and the GLCM window sizes and directions (PARAM['file_suffix_lst'])
    PARAM['add_bands'] = [
        PARAM['merge_bands']]*len(PARAM['file_suffix_lst'])

    # -------- Normalization param ----
    if PARAM['if_norm_min_max_band1']:
        # if normalize grey scale band need to specify the min max depending on bit depth
        PARAM['norm_min_max_band1'] = [0, 2**PARAM['BIT_DEPTH'] - 1]
    else:
        PARAM['norm_min_max_band1'] = None

    return