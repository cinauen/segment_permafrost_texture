"""
================ Run test after training ===============\n
This script tests the trained model on the test patches.
No augmentation is required for this.
The test patch data and if required the GLCM features, must have been
extracted before with preproc/MAIN_extract_input_untiled.py.

The test includes:
- Loading tiles per batch (in dataloader):
    This includes non-augmented and augmented tiles and corresponding
    sample weights (weights are used for evaluation here) and labelling.
    The tiles are divided into train and validation according to the
    sub-areas (e.g. train: [train-01, train-02], validation: [train-03]
     as defined via PARAM['PHASE_META_FILE'])
- Preparing input features and labels for testing (in dataloader):
    - apply optional standardisation (same as did for training data)
    - merge required bands into one torch tensor
- Inside the testing loop:
    - predict test patch
    - run evaluation

The model to be tested must be selected by the:
 - feature preparation ID (PARAM_PREP_ID) as defined in:
    param_settings/PARAM_inp_CNN_feature_prep_v01.txt
    (description see: docs/PARAM_options_feature_preparation_CNN.md)
 - training option ID (PARAM_TRAIN_ID) as defined in:
    param_settings/PARAM_inp_CNN_train_v01.txt
    (description see: docs/PARAM_options_training_CNN.md)
 - the used training area and input imagery (e.g. HEX, SPOT) as defined in
    the project parameter file e.g.
    - /../example/2_segment/01_input/PARAM06_test_model_BLyaE_v1_HEX1979_A01_A02_set01a_on_BLyaE.py

The main modules called in this file include:
- cnn_workflow/custom_data_loader.py: data loading and feature preparation
- cnn_workflow/custom_model.py: CNN model initialization
- cnn_workflow/custom_train.py: load trained model and run test
- cnn_workflow/custom_metric.py: setup and calculation of evaluation metrics
- postproc/MAIN_extract_segmentation_properties.py: extracts true and
    predicted shapes as well as geometric properties for later statistical
    evaluation
- postproc/MAIN_extract_TP_TN_per_class_uncert.py: extracts true positives,
    true negatives etc. per class uncertainty for later statistical
    evaluation


--- Run script in command line
usage: MAIN_evaluate_model.py [-h] [--GPU_LST_STR GPU_LST_STR] [--CV_NUM CV_NUM] PATH_PROC PARAM_PREP_ID PARAM_TRAIN_ID PARAM_FILE

positional arguments:
  PATH_PROC             path to processing folder (2_segment folder)
  PARAM_PREP_ID         id of used merge and scaling parameters
                            e.g. v079 (PARAM_inp_CNN_feature_prep_v01.txt)
  PARAM_TRAIN_ID        id of used training parameters
                            e.g. t16onl (PARAM_inp_CNN_train_v01.txt)
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
cnn_workflow/test/MAIN_evaluate_model_test.py

"""

import logging
import os
import sys
import argparse
import datetime as dt

import torch
import memory_profiler


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
    PARAM['LOG_FILE_SUFFIX'] = 'run_test'

    # ----- file and folder settings
    # add cross-validation number to file prefix
    add_prefix = '_cv' + '{0:02d}'.format(CV_NUM)
    PARAM['FILE_PREFIX'] += add_prefix
    # create folder to save training outputs
    folder_model_output = (
        f"{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}_cv{'{0:02d}'.format(CV_NUM)}")
    PARAM['PATH_PROC'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'], folder_model_output))
    if not os.path.isdir(PARAM['PATH_PROC']):
        os.makedirs(PARAM['PATH_PROC'])


    # --- set other aditional params -----
    # amount of classes
    n_classes = len(PARAM['CLASS_LABELS'][0])


    # --------------  save parameter values -----------------
    param_file_name = (
        f"A_{folder_model_output}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        PARAM['PATH_TRAIN'], param_file_name, {'all_param': PARAM},
        close=True)

    # --------------- setup logging ---------------
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{folder_model_output}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_TRAIN'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']} CV{CV_NUM}")

    # ------ set device usage for torch -----
    device, PARAM['GPU_lst'] = custom_data_loader.set_device(
        PARAM['GPU_LST_STR'])

    # -------- Get filenames of test tiles -------
    # Initialize class to read filenames
    file_names = custom_data_loader.GetFilenames(
        PARAM['PATH_INP_BASE'], None,
        x_bands=PARAM['X_BANDS'], y_bands=PARAM['Y_BANDS'])

    # The file names are collected according to the test patch numbers
    # and per area and imagery as defined in PARAM['FILE_PREFIX_SITE_LST']
    # e.g. ['BLyaE_HEX1979_test_perc0-2_g0-3_8bit',
    #       'BLyaE_SPOT2018_test_std4_8bit']
    # PARAM['FILE_SITE_dict'] defines the phase names as list
    # (e.g. ['BLyakhE_HEX1979_test', 'BLyakhE_SPOT2018_test'])
    # PHASE_META_FILE specifies the test patches per phase as sublists
    # e.g. {'BLyaE_HEX1979_test_perc0-2_g0-3_8bit': [
    #        [f'test-{x:02d}' for x in range(1, 7)], []],
    #        ...}
    # whereas the first sublist belongs to 'BLyakhE_HEX1979_test' and
    # the second sublist remains here empty as it belongs to 'BLyakhE_SPOT2018_test'
    PHASE_META_FILE = PARAM['PHASE_META_FILE'][CV_NUM]
    for i_prefix in PARAM['FILE_PREFIX_SITE_LST']:
        # path to files
        i_site = PARAM['PATH_PREFIX_SITE'][i_prefix]
        file_names.path_inp = os.path.join(
            PARAM['PATH_INP_BASE'], i_site)
        file_names.file_prefix = PARAM['FILE_SITE_dict'][i_prefix]
        if PARAM['load_phase_separate']:
            # --- Use metadata file to load test patches per phase
            for e, i in enumerate(PARAM['PHASE_NAME']):
                if len(PHASE_META_FILE[i_prefix][e]) == 0:
                    continue
                file_names.get_file_names_from_meta(
                    PHASE_META_FILE[i_prefix][e],
                    phase=i, meta_suffix=PARAM['meta_suffix_lst'][e],
                    file_id=i_prefix,
                    query_text=PARAM['query_text'],
                    aug_query=None)
        else:
            # ---- Search files in folder
            for e, i in enumerate(PARAM['PHASE_NAME']):
                for i_search in PHASE_META_FILE[i_prefix][e]:
                    file_names.get_file_names_glob(
                        file_id=i_prefix, phase=i,
                        prefix=f"{PARAM['FILE_SITE_dict'][i_prefix]}{i_search}",
                        append_to_lst=True)

    # loop through all test phases
    for i_phase in PARAM['PHASE_NAME']:
        PARAM['PATH_PROC_OUT'] = os.path.join(PARAM['PATH_PROC'], i_phase)
        # create output path
        if not os.path.isdir(PARAM['PATH_PROC_OUT']):
            os.mkdir(PARAM['PATH_PROC_OUT'])

        # -------------------- DATA LOADER SETUP -------------------
        # --- load testing data (testing independent of train and validation) ---
        test_ds = custom_data_loader.CustomDataset_ScaleMerge_Test(
            file_names.data_files[i_phase], file_names.seg_files[i_phase],
            file_names.file_id[i_phase], PARAM['file_suffix_lst'],
            pca_calc_utils, PARAM['PATH_PROC'],
            x_bands=file_names.x_bands, y_bands=file_names.y_bands,
            add_bands=PARAM['add_bands'],
            width_out=PARAM['window_size'], height_out=PARAM['window_size'],
            calc_pca=PARAM['calc_PCA'], PCA_components=PARAM['PCA_components'],
            standardize_add=PARAM['standardize_add'],
            norm_on_std=PARAM['norm_on_std'],
            norm_min_max_band1=PARAM['norm_min_max_band1'],
            take_log=PARAM['take_log'],
            gpu_no=PARAM['GPU_lst'][0],
            save_files_debug=False,  # if save files during debugging
            dict_relabel=PARAM['DICT_RELABEL'],  # dictionary with classes for relabelling
            set_to_nan_lst=PARAM['MASK_TO_NAN_LST'],  # specific class number to be set to nan
            standardize_individual=PARAM['standardize_individual'],  # standardisation per individual tile
            if_std_band1=PARAM['if_std_band1'],  # if standardise grey scale band
            dl_phase=i_phase,
            debug_plot=False,
            norm_clip=PARAM['norm_clip'],  # if clip to 0 and 1 after nomalize (if normalize)
            exclude_greyscale=PARAM['exclude_greyscale'],
            sensor_dict=PARAM['FILE_SENSOR_dict'],
            feature_add=PARAM['feature_add'])

        # --- get stats from metadata, which were calculated per labelling area.
        test_ds.get_stats_dict(
            PARAM['PATH_INP_BASE'], PARAM['STATS_FILE'],
            PARAM['PHASE_STATS_FILE'][CV_NUM])

        # ----- load data to pytorch
        # (shuffle is not required for testing)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=PARAM['N_BATCH'],
            num_workers=PARAM['N_WORKERS'], shuffle=False)

        # ------ get ouput and input file prefix
        if 'old_naming' in PARAM.keys() and PARAM['old_naming'] == 1:
            # !!! TO BE REMOVED LATER !!!
            # allow naming from old code version
            n_channels = test_ds.n_channels
            file_prefix_inp = f"{PARAM['PARAM_TRAIN_ID']}_cv{CV_NUM:02d}"
            file_inp_lst = [
                file_prefix_inp,
                PARAM['MODEL_ARCH'], 'nchannels', str(n_channels),
                'nclasses', str(n_classes)]
            file_out_lst = [
                PARAM['FILE_PREFIX'], PARAM['file_prefix_add'],
                'nclasses', str(n_classes)]
        elif 'old_naming' in PARAM.keys() and PARAM['old_naming'] == 2:
            file_inp_lst = [
                PARAM['FILE_PREFIX'], PARAM['file_prefix_add'],
                'nclasses', str(n_classes)]
            file_out_lst = [
                PARAM['FILE_PREFIX'], PARAM['file_prefix_add'],
                'nclasses', str(n_classes)]
        else:
            file_inp_lst = [
                PARAM['FILE_PREFIX'], PARAM['file_prefix_add'],
                f'ncla{n_classes}']
            file_out_lst = [
                PARAM['FILE_PREFIX'], PARAM['file_prefix_add'],
                f'ncla{n_classes}']
        file_inp_prefix = file_utils.create_filename(file_inp_lst)
        file_out_prefix = file_utils.create_filename(file_out_lst)

        # ---- initialize a model
        n_channels = test_ds.n_channels
        model_test = custom_model.initialize_model(
            PARAM['MODEL_ARCH'], PARAM['GPU_lst'],
            n_channels, n_classes, device, PARAM['use_batchnorm'],
            (PARAM['window_size'], PARAM['window_size']))

        # ------------- initialize test
        test_c = custom_train.TestRun(
            model_test, test_dl,
            PARAM['N_BATCH'], n_classes, PARAM['CLASS_LABELS'],
            PARAM['PATH_PROC'], PARAM['PATH_PROC_OUT'],
            file_out_prefix, device, file_inp_prefix,
            trained_model=None, trained_epoch=None,
            ignore_index=PARAM['metrics_ignore_index'],
            save_files=True, EPSG_OUT=PARAM['EPSG_TARGET'],
            DICT_RELABEL=PARAM['DICT_RELABEL'],
            MASK_TO_NAN_LST=PARAM['MASK_TO_NAN_LST'],
            phase=i_phase,
            class_eval_weighted_TP_TN_lst=PARAM['class_eval_weighted_TP_TN_lst'])

        # ----- load trained models ----
        # read model according to metadata
        test_c.read_model_metadata()
        # define for which epochs want to load model
        if PARAM['SINGLE_EPOCH_LOAD'] is not None:
            # use specified single epoch
            model_load_lst = [PARAM['SINGLE_EPOCH_LOAD']]
        elif PARAM['MIN_EPOCH'] is not None:
            # use all checkpoints after minimum epoch
            model_checkp = test_c.model_meta.loc[test_c.model_meta.if_checkp, :]
            model_load_lst =  model_checkp.query(
                "epoch >= @PARAM['MIN_EPOCH']").index.tolist()
            if len(model_load_lst) == 0:
                model_load_lst = model_checkp.reset_index().nlargest(5, 'epoch').epoch.tolist()
        else:
            # use epochs with best five models
            model_load_lst = test_c.model_meta.nlargest(5, 'best_acc')

        # loop through selected epochs
        for i_model in model_load_lst:
            # ----- load trained model -------
            test_c.load_model(epoch=i_model)
            # ----- run test -----
            test_c.run_test()

    logging.info(
        '\n---- Finalised model testing {}\n\n'.format(
            dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

    # ---- save time stats
    time_file = '_'.join(
        ['A', folder_model_output, PARAM['LOG_FILE_SUFFIX']])
    monitoring_utils.save_time_stats(
        prof, PARAM['PATH_TRAIN'], time_file)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('path to processing folder (2_segment folder)'))
    parser.add_argument(
        'PARAM_PREP_ID', type=str,
        help=('id of used merge and scaling parameters e.g. v079 ' +
              '(PARAM_inp_CNN_feature_prep_v01.txt)'))
    parser.add_argument(
        'PARAM_TRAIN_ID', type=str,
        help=('id of used training parameters e.g. t16onl ' +
              '(PARAM_inp_CNN_train_v01.txt)'))
    parser.add_argument(
        'PARAM_FILE', type=str, help='name of project parameter file')
    parser.add_argument(
        '--GPU_LST_STR', type=str,
        help='list of GPUs (separated by ":")', default='0')
    parser.add_argument(
        '--CV_NUM', type=int,
        help=('CV number if do not want to start at 0'), default=0)
    parser.add_argument(
        '--PATH_INP_BASE', type=str,
        help=('path to site pre processing folder as default (None) '
             'the path to 1_site_preproc is used'),
        default='None')

    args = parser.parse_args()

    import utils.pca_calc_utils as pca_calc_utils
    main(pca_calc_utils, vars(args), CV_NUM=vars(args)['CV_NUM'])

    # to save all data to temp folder could use:
    # with tempfile.TemporaryDirectory(prefix=temp_folder) as tempdir:
    #     main(tempdir)


