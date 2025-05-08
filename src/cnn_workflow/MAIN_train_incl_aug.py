"""
======== Training and fine-tuning with on-the fly augmentation =========
This scripts runs the model training or fine-tuning.
As input it uses the tiles and labels prior created with
MAIN_create_training_tiles.py

The training includes at EVERY EPOCH:
 - Loading tiles per batch (in dataloader):
    Non-augmented greyscale tiles, sample weights and corresponding labelling.
    The tiles are divided into train and validation according to the
    sub-areas (e.g. train: [train-01, train-02], validation: [train-03]
     as defined via PARAM['PHASE_META_FILE'])
 - Preparing input features and labels for training (in dataloader):
    1. augmenting training tiles
    2. recalculate the GLCMs of the augmented tiles (as subprocess)
    3. apply optional standardisation
    4. merge required bands into one torch tensor
 - Inside the training loop (depending on the training options):
    - apply weights
    - save training and validation metric

For fine-tuning the model can be selected by providing the input model
properties and the input epoch.

The model is trained according to:
- The framework parameters including:
   - parameters for feature preparation as defined in:
     param_settings/PARAM_inp_CNN_feature_prep_v01.txt
     (description see: docs/PARAM_options_feature_preparation_CNN.md)
   - training parameters as defined in:
     param_settings/PARAM_inp_CNN_train_v01.txt
     (description see: docs/PARAM_options_training_CNN.md)
- The training site/area and the imagery type (e.g. HEX, SPOT) as
  specified in the project file e.g.
   - /../example/2_segment/01_input/PARAM06_train_BLyaE_v1_HEX1979_A01_A02_set01a.py
   - /../example/2_segment/01_input/PARAM06_train_BLyaE_v1_HEX1979_A01_A02_set01a_v079t16_finet_on_FadN.py

The main modules called in this file include:
 - cnn_workflow/custom_data_loader.py: data loading and feature preparation
 - cnn_workflow/custom_train.py: run training
 - cnn_workflow/custom_model.py: CNN model setup
 - cnn_workflow/custom_metric.py: setup and calculation of evaluation metrics
 - cnn_workflow/custom_augmentation.py: setup of augmentation chain


--- Run script in command line
usage: MAIN_train_incl_aug.py [-h] [--GPU_LST_STR GPU_LST_STR] [--GPU_num_GLCM GPU_NUM_GLCM] [--CV_NUM CV_NUM] PATH_PROC PATH_LOCAL PARAM_PREP_ID PARAM_TRAIN_ID PARAM_FILE

positional arguments:
  PATH_PROC             path to processing folder (2_segment folder)
  PATH_LOCAL            path where to save temp files
  PARAM_PREP_ID         id of used merge and scaling parameters
                            e.g. v079 (PARAM_inp_CNN_feature_prep_v01.txt)
  PARAM_TRAIN_ID        id of used training parameters
                            e.g. t16onl (PARAM_inp_CNN_train_v01.txt)
  PARAM_FILE            name of project parameter file

options:
  -h, --help            show this help message and exit
  --GPU_LST_STR GPU_LST_STR
                        list of GPUs (separated by ":")
  --GPU_num_GLCM GPU_NUM_GLCM
                        GPU number for GLCM calc
  --CV_NUM CV_NUM       CV number if do not want to start at 0
  --PATH_INP_BASE PATH_INP_BASE
                        path to site pre processing folder as default
                        (None) the path to 1_site_preproc is used


--- for debugging use:
cnn_workflow/test/MAIN_train_incl_aug_test.py

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

    # ---------initialize all paramters -------------
    PARAM = param_utils.initialize_segmentation_param(inp)
    PARAM['LOG_FILE_SUFFIX'] = 'training'

    if 'off' in PARAM['PARAM_TRAIN_ID'] or 'offl_aug' in PARAM['PARAM_TRAIN_ID']:
        sys.exit('!!! ERROR: need to use MAIN_train_off.py with offline augmentation')

    # --------- create folder to save training outputs --------
    folder_model_output = (
        f"{PARAM['PARAM_PREP_ID']}{PARAM['PARAM_TRAIN_ID']}_cv{'{0:02d}'.format(CV_NUM)}")
    PARAM['PATH_PROC'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'], folder_model_output))
    if not os.path.isdir(PARAM['PATH_PROC']):
        os.makedirs(PARAM['PATH_PROC'])

    # ---------------- set other additional params -----------------
    # amount of classes
    n_classes = len(PARAM['CLASS_LABELS'][0])
    # add cross-validation number to file prefix
    add_prefix = f'_cv{CV_NUM:02d}'
    PARAM['FILE_PREFIX'] += add_prefix

    # ------ in case of fine-tuning, load pre-trained model ------
    if 'model_input' in PARAM.keys() and PARAM['model_input'] is not None:
        param_utils.setup_model_input_properties(
            PARAM, CV_NUM, n_classes)

    # -------------------  save parameter values -------------------
    param_file_name = (
        f"A_{folder_model_output}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        PARAM['PATH_TRAIN'], param_file_name, {'all_param': PARAM},
        close=True)

    # ----------------------- setup logging -----------------------
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator
    # needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{folder_model_output}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_TRAIN'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']} CV{CV_NUM}")

    # ------ set device usage for torch -----
    device, PARAM['GPU_lst'] = custom_data_loader.set_device(
        PARAM['GPU_LST_STR'])

    # ---- Get filenames of training/validation tiles
    # (here use non-augmented tiles only as augmentation is done on the fly
    # during trainig)
    # Initialize class to read filenames
    file_names = custom_data_loader.GetFilenames(
        PARAM['PATH_INP_BASE'], None,
        x_bands=PARAM['X_BANDS'], y_bands=PARAM['Y_BANDS'])

    # The file names are collected for the speific CV number and per
    # training site and imagery (defined in PARAM['FILE_PREFIX_SITE_LST'])
    # PHASE_META_FILE specifies the sub-areas per phase ['train', 'validate'] as sublists e.g.
    # {'BLyaE_HEX1979_A01_perc0-2_g0-3_8bit': [
    #        ['train-01', 'train-02'], ['train-03']]}
    # It is used to assign the sub-areas for a CV run
    # to the specific phases
    # e.g. for ['train', 'validate'] --> [['train-01', 'train-02'], ['train-03']]
    # whereas the first sublist belongs to the training and the second
    # to the validation
    PHASE_META_FILE = PARAM['PHASE_META_FILE'][CV_NUM]
    for i_prefix in PARAM['FILE_PREFIX_SITE_LST']:
        # path to files
        # e.g. ./03_train_inp.BLyaE_HEX1979_A01_perc0-2_g0-3_8bit_Lv01_w298-298_v00_rot_cubic/
        i_site = PARAM['PATH_PREFIX_SITE'][i_prefix]
        file_names.path_inp = os.path.join(
            PARAM['PATH_INP_BASE'], i_site)
        # file name prefix e.g. BLyaE_HEX1979_A01
        file_names.file_prefix = PARAM['FILE_SITE_dict'][i_prefix]

        if PARAM['load_phase_separate']:
            # --------- get filenames accoding to sub-AOIs ----------
            # load the phases (training and validation) separately
            # according to the predefined AOIs e.g. "train-01".
            # This guarantees non-overlap between training and validation
            # tiles and no overlapping validation areas between
            # the different CV runs.

            # loop through the pases e.g. ['train', 'validate']
            for e, i in enumerate(PARAM['PHASE_NAME']):
                if len(PHASE_META_FILE[i_prefix][e]) == 0:
                    continue
                # For each phase, the filenames are collected from the
                # metadata file and according to the AOI combination as
                # defined in PHASE_META_FILE.
                file_names.get_file_names_from_meta(
                    PHASE_META_FILE[i_prefix][e],
                    phase=i, meta_suffix=PARAM['meta_suffix_lst'][e],
                    file_id=i_prefix, query_text=PARAM['query_text'])
                # Depending on the amount of augmentations multiply the
                # file occurances
                if i == 'train':  # !!! augment only for train
                    file_names.data_files[i] *= PARAM['n_augment']
                    file_names.seg_files[i] *= PARAM['n_augment']
                    file_names.file_id[i] *= PARAM['n_augment']
        else:
            # ---- Search files in folder and random split --------
            # use glob to search the file names in the input folder
            # (file_names.path_inp).
            # And use random split divide the tiles into training, validation
            # and test
            # (!!! however to ensure no leakage would need to
            # have tiles without overlap !!!)

            # find filenames
            file_names.get_file_names_glob(file_id=i_prefix, phase='all')

            # use random split
            # split into 0.5 train and 0.5 validation and 10 test patches
            # (test patches are removed from the validation set)
            # augmentation multiplies the traing set by n_augment
            file_names.split_train_validate(
                split_ratio=0.5, n_augment=PARAM['n_augment'],
                test_patch_num=10)

    # -- adjust file number to batch size
    if PARAM['adjust_file_num_to_batch']:
        for i_phase in file_names.data_files.keys():
            file_rem = len(file_names.data_files[i_phase])%PARAM['N_BATCH']
            if file_rem > 0:
                file_names.data_files[i_phase] = file_names.data_files[i_phase][:-file_rem]
                file_names.seg_files[i_phase] = file_names.seg_files[i_phase][:-file_rem]
                file_names.file_id[i_phase] = file_names.file_id[i_phase][:-file_rem]

    # -- get class counts and weighting from meta files
    class_sum_train, class_weights_train = custom_data_loader.get_class_counts(
        file_names.meta['train'], PARAM['DICT_RELABEL'],
        PARAM['CLASS_LABELS'][0],
        set_to_nan_index_inp=PARAM['MASK_TO_NAN_LST'],
        n_augment=PARAM['n_augment'])


    # -------------------- DATA LOADER SETUP -------------------
    # --- initialize data loader for training data ---
    train_ds = custom_data_loader.CustomDataset_augment_ScaleMerge(
        file_names.data_files['train'], file_names.seg_files['train'],
        file_names.file_id['train'], PARAM['file_suffix_lst'],
        pca_calc_utils, PARAM['PATH_PROC'], PARAM['SCALE_TYPE_dict'],
        PARAM['PATH_LOCAL'],
        x_bands=file_names.x_bands, y_bands=file_names.y_bands,
        add_bands=PARAM['add_bands'],
        width_out=PARAM['window_size'], height_out=PARAM['window_size'],
        calc_pca=PARAM['calc_PCA'], PCA_components=PARAM['PCA_components'],
        standardize_add=PARAM['standardize_add'],
        norm_on_std=PARAM['norm_on_std'],
        norm_min_max_band1=PARAM['norm_min_max_band1'],
        take_log=PARAM['take_log'],
        gpu_no=PARAM['GPU_lst'][0], gpu_no_GLCM=PARAM['GPU_num_GLCM'],
        augmentations_geom=PARAM['aug_geom'],  # geometric augmentation
        augmentations_col=PARAM['aug_col'],  # color augmentation
        band_lst_col_aug=PARAM['band_lst_col_aug'],
        augmentations_range=PARAM['aug_range'],  # change range to 0 - 1
        save_files_debug=False,  # if save files during debugging
        dict_relabel=PARAM['DICT_RELABEL'],  # dictionary with classes for relabelling
        set_to_nan_lst=PARAM['MASK_TO_NAN_LST'],  # specific class number to be set to nan
        standardize_individual=PARAM['standardize_individual'],  # standardisation per individual tile
        if_std_band1=PARAM['if_std_band1'],  # if standardise grey scale band
        dl_phase='train',  # which phase
        debug_plot=False, norm_clip=PARAM['norm_clip'],
        exclude_greyscale=PARAM['exclude_greyscale'],  # if do not include greyscale in trainig (only GLCMs)
        sensor_dict=PARAM['FILE_SENSOR_dict'],
        feature_add=PARAM['feature_add']  # add sensor number in training
        )

    # --- get stats from metadata, which were calculated per labelling area.
    train_ds.get_stats_dict(
        PARAM['PATH_INP_BASE'], PARAM['STATS_FILE'],
        PARAM['PHASE_STATS_FILE'][CV_NUM])

    # ---- options to plot input data ---
    # test and plot data examples
    train_ds.debug_plot=True
    for i in [0, 1]:
        for ii in range(0):
            x, y, w_sample = train_ds[i]
    train_ds.debug_plot=False

    # -- plot augmentations for test
    #custom_data_loader.plot_augmentations(
    #     train_ds, idx=0, n_augmentations=2)

    # -- get data from a single set
    # x, y = train_ds[0]
    # -- plot
    # train_ds.img_plot(0)

    # --- initialize data loader for validation data ---
    valid_ds = custom_data_loader.CustomDataset_augment_ScaleMerge(
        file_names.data_files['validate'], file_names.seg_files['validate'],
        file_names.file_id['validate'], PARAM['file_suffix_lst'],
        pca_calc_utils, PARAM['PATH_PROC'], PARAM['SCALE_TYPE_dict'],
        PARAM['PATH_LOCAL'],
        x_bands=file_names.x_bands, y_bands=file_names.y_bands,
        add_bands=PARAM['add_bands'],
        width_out=PARAM['window_size'], height_out=PARAM['window_size'],
        calc_pca=PARAM['calc_PCA'], PCA_components=PARAM['PCA_components'],
        standardize_add=PARAM['standardize_add'],
        norm_on_std=PARAM['norm_on_std'],
        norm_min_max_band1=PARAM['norm_min_max_band1'],
        take_log=PARAM['take_log'],
        gpu_no=PARAM['GPU_lst'][0], gpu_no_GLCM=PARAM['GPU_num_GLCM'],
        augmentations_geom=None, # do not augment validation data
        augmentations_col=None,  # do not augment validation data
        band_lst_col_aug=PARAM['band_lst_col_aug'],
        augmentations_range=PARAM['aug_range'],
        save_files_debug=False,
        dict_relabel=PARAM['DICT_RELABEL'],
        set_to_nan_lst=PARAM['MASK_TO_NAN_LST'],
        standardize_individual=PARAM['standardize_individual'],
        if_std_band1=PARAM['if_std_band1'], dl_phase='validate',
        debug_plot=False, norm_clip=PARAM['norm_clip'],
        exclude_greyscale=PARAM['exclude_greyscale'],
        sensor_dict=PARAM['FILE_SENSOR_dict'],
        feature_add=PARAM['feature_add'])

    # --- get stats from metadata, which were calculated per labelling area.
    valid_ds.get_stats_dict(PARAM['PATH_INP_BASE'], PARAM['STATS_FILE'],
                            PARAM['PHASE_STATS_FILE'][CV_NUM])

    # ----- load data to pytorch
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=PARAM['N_BATCH'],
        num_workers=PARAM['N_WORKERS'], shuffle=True)
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, batch_size=PARAM['N_BATCH'],
        num_workers=PARAM['N_WORKERS'], shuffle=True)

    # ------ get ouput file prefix
    file_out_lst = [
        PARAM['FILE_PREFIX'], PARAM['file_prefix_add'],
        f'ncla{n_classes}']
    file_out_prefix = file_utils.create_filename(file_out_lst)

    # --- initialize model
    n_channels = train_ds.n_channels
    unet = custom_model.initialize_model(
        PARAM['MODEL_ARCH'], PARAM['GPU_lst'], n_channels, n_classes,
        device, PARAM['use_batchnorm'],
        (PARAM['window_size'], PARAM['window_size']))

    # - check shape one pass
    # xb, yb = next(iter(train_dl))
    # print([xb.shape, yb.shape])
    # check shape pred, testing one pass
    # pred = unet(xb)
    # print(pred.shape)

    # ---- define the loss function and add class weights if required
    if PARAM['loss_weights'] == [1111]:
        class_weights = torch.FloatTensor(
            class_weights_train.values).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=PARAM['metrics_ignore_index'],
            reduction=PARAM['loss_reduction'])
    elif (PARAM['loss_weights'] is not None
          and PARAM['loss_weights'] != [9999]):
        # this could be used if specified specific class weights
        # in loss_weights parameter (is currently not used)
        class_weights = torch.FloatTensor(
            PARAM['loss_weights']).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=PARAM['metrics_ignore_index'],
            reduction=PARAM['loss_reduction'])
    else:
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=PARAM['metrics_ignore_index'],
            reduction=PARAM['loss_reduction'])

    # ---- define optimisation algorithm
    PARAM['otimizer'] = eval(PARAM['otimizer'])
    opt = PARAM['otimizer'](unet.parameters(), lr=PARAM['learning_rate'])

    # ------- initialize learning rate
    scheduler_lr = custom_train.initialize_lr(
        opt, PARAM['lr_scheduler'], PARAM['lr_gamma'],
        PARAM['lr_milestones'])

    # --- in case of fine-tuning, load pre-trained model ---
    if 'model_input' in PARAM.keys() and PARAM['model_input'] != None:
        unet_loader = custom_train.LoadModel(
            unet, PARAM['model_input']['PATH_PROC_inp'],
            PARAM['model_input']['file_prefix'], device)
        unet_loader.read_model_metadata()
        unet_loader.load_model(epoch=PARAM['model_input']['inp_epoch'][CV_NUM])

    # ---- initialize training class
    train_c = custom_train.TrainRun(
        unet, train_dl, valid_dl, loss_fn, opt, custom_train.acc_metric,
        PARAM['N_BATCH'], n_classes, PARAM['N_EPOCHS'], device,
        PARAM['CLASS_LABELS'], PARAM['PATH_PROC'], file_out_prefix,
        early_stopping=PARAM['early_stopping'],
        extended_output=PARAM['extended_output'],
        loss_weights=PARAM['loss_weights'],
        epoch_save_min=PARAM['epoch_save_min'],
        scheduler_lr=scheduler_lr,
        ignore_index=PARAM['metrics_ignore_index'],
        loss_reduction=PARAM['loss_reduction'],
        weight_calc_type=PARAM['weight_calc_type'])

    # ----- run training
    train_c.run_train()

    # ----- save metadata
    for i, i_suff in zip(PARAM['PHASE_NAME'], PARAM['meta_suffix_lst']):
        file_names.save_metadata(
            train_ds.x_bands + train_ds.y_bands,
            path_out=PARAM['PATH_PROC'], phase=i, meta_suffix=i_suff)

    logging.info(
        '\n---- Finalised training'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '\n\n')

    # ---- save time stats
    time_file = '_'.join(
        ['A', folder_model_output, PARAM['LOG_FILE_SUFFIX']])
    monitoring_utils.save_time_stats(
        prof, PARAM['PATH_TRAIN'], time_file)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train model with on-the-fly augmentation')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('path to processing folder (2_segment folder)'))
    parser.add_argument(
        'PATH_LOCAL', type=str,
        help=('path where to save temp files'))
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
        '--GPU_num_GLCM', type=int,
        help=('GPU number for GLCM calc'), default=0)
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


