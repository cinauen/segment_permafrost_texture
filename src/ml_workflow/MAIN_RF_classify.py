'''
=================== ML classification ====================
Depending on the defined processing step this model runs the following:
- PARAM['PROC_STEP'] = 'rt': train, test, predict
- PARAM['PROC_STEP'] = 'gs': hyperparameter tuning (grid search)
- PARAM['PROC_STEP'] = 'fi': feature importance (shap analysis)

Currently this script uses just the RandomForestClassifier using either
cuml (if GPUs available) or scikit-learn (if run on CPU).
Other machine learning algorithms could be included by adapting:
- ml_classification_utils.set_classification_param()
- ml_classification_utils.set_up_classification_algorithms()

As the machine learning training is pixel based, the input does not require
image tiles as input but can be converted to DataFrames.
The input data including the GLCMs is therefore taken from the untiled
images. These input must have been created prior to running this script
by using preproc/MAIN_extract_input_untiled.py

The main modules called in this file include:
 - cnn_workflow/custom_data_loader.py: default data loading and feature preparation
    (this is for comparability with CNN)
 - ml_workflow/ml_classification_utils.ClassifierML():
    run training, hyperparameter tuning and feature importance analysis
 - OPTIONAL ml_workflow/preproc_scikit.py: for testing do the preprocessing
    using scikit-learn or cuml (instead of using the data_loader)


--- Run script in command line
usage: MAIN_RF_classify.py [-h] [--GPU_LST_STR GPU_LST_STR] [--N_JOBS N_JOBS] [--CV_NUM CV_NUM] PATH_PROC PROC_STEP PARAM_PREP_ID PARAM_TRAIN_ID PARAM_FILE

positional arguments:
  PATH_PROC         path to processing folder (2_segment folder)
  PROC_STEP         name processing step:
                        "rt": for train test,
                        "gs": hyperparam tuning,
                        "fi": feature importance analysis
  PARAM_PREP_ID     id of merge and scaling parameters
                        e.g. vML001 (PARAM_inp_ML_feature_prep_v01.txt)
  PARAM_TRAIN_ID    id of training parameters
                        e.g. tML01 (PARAM_inp_ML_train_v01.txt)
  PARAM_FILE        name of parameter file

options:
  -h, --help            show this help message and exit
  --GPU_LST_STR GPU_LST_STR
                        list of GPUs (separated by ":")
  --N_JOBS N_JOBS       amount of jobs to run in parallel
                            (important if run on CPU)
  --CV_NUM CV_NUM       CV number if do not want to start at 0


--- for debugging use:
ml_workflow/test/MAIN_RF_classify_test.py

'''
import os
import sys
import gc
import datetime as dt
import numpy as np
import pandas as pd
import argparse

from joblib import Parallel, delayed, cpu_count
import memory_profiler

# ----------------- import custom utils -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import utils.file_utils as file_utils
import utils.monitoring_utils as monitoring_utils
import param_settings.param_utils as param_utils
import postproc.MAIN_extract_segmentation_properties as MAIN_extract_segmentation_properties
import postproc.MAIN_extract_TP_TN_per_class_uncert as MAIN_extract_TP_TN_per_class_uncert


@monitoring_utils.conditional_profile
def main(pca_calc_utils, inp, CV_NUM=0, client=None):
    import torch
    import cnn_workflow.cnn_workflow.custom_data_loader as custom_data_loader
    # ---- set up GPU usage and import additional modules using GPUs ---
    if torch.cuda.is_available():
        # --- if GPUs are available
        import cuml
        from cuml.common.device_selection import set_global_device_type
        # define output types
        cuml.set_global_output_type('numpy')
        # All operators supporting GPU execution will run on the GPU after
        # this configuration
        set_global_device_type("GPU")
        GPU_proc = True
        optimi_dict = {}
    else:
        GPU_proc = False
        optimi_dict = {'n_jobs':inp['N_JOBS']}

    import ml_workflow.ml_classification_utils as ml_classification_utils
    import ml_workflow.ml_preproc_scikit as ml_preproc_scikit

    # -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_segmentation_param(inp)

    # ----- create folder to save training outputs -
    folder_model_output = (
        f'{PARAM["PARAM_PREP_ID"]}{PARAM["PARAM_TRAIN_ID"]}_cv{"{0:02d}".format(CV_NUM)}')
    PARAM['PATH_PROC'] = os.path.normpath(
        os.path.join(PARAM['PATH_TRAIN'], folder_model_output))
    if not os.path.isdir(PARAM['PATH_PROC']):
        os.makedirs(PARAM['PATH_PROC'])

    # --- set other additional params -----
    #if PARAM['feature_add'] is None:
    #    PARAM['feature_add'] = []
    # add file prefix to contain cross validation number
    add_prefix = '_cv' + '{0:02d}'.format(CV_NUM)
    PARAM['FILE_PREFIX'] += add_prefix

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

    # ------------ set GPU device usage  ----
    GPU_lst = [int(x) for x in PARAM['GPU_LST_STR'].split(':')]
    if GPU_proc:
        monitoring_utils.get_GPU_memory_info(GPU_lst)

    # -------- Get filenames for training/testing ---------
    # Note: here use untiled images as tiling is not requird for
    #   pixel-based supervsed ML.
    file_name_tile = {}
    # prepare file metadata dict
    for e, i in enumerate(PARAM['PHASE_NAME']):
        file_name_tile[i] = {'data': [], 'seg': [], 'file_id': [],
                             'aoi_key': [], 'sensor': [], 'glcm': []}
    # The file names are collected for the speific CV number and per
    # training site and imagery (defined in PARAM['FILE_PREFIX_SITE_LST'])
    # PHASE_META_FILE specifies the sub-areas per phase ['train', 'validate']
    # as sublists e.g.
    # {'BLyakhE_HEX1979_A01_perc0-2_g0-3_8bit': [
    #        ['train-01', 'train-02'], ['train-03']]}
    # e.g. if pahses are ['train', 'validate'] then the first
    # sublist ['train-01', 'train-02'] would be used for training and
    # the second  ['train-03'] for validation
    PHASE_META_FILE = PARAM['PHASE_META_FILE'][CV_NUM]
    for i_prefix in PARAM['FILE_PREFIX_SITE_LST']:
        # path to files
        i_path = PARAM['PATH_PREFIX_SITE'][i_prefix]
        full_path = os.path.join(PARAM['PATH_INP_BASE'], i_path)
        for e, i in enumerate(PARAM['PHASE_NAME']):
            for ii in PHASE_META_FILE[i_prefix][e]:
                file_prefix = f"{PARAM['FILE_SITE_dict'][i_prefix]}{ii}"
                file_name_tile[i]['data'].append(
                    os.path.join(full_path, f"{file_prefix}_data.tif"))
                file_name_tile[i]['glcm'].append(
                    [os.path.join(full_path, f"{file_prefix}_data_{x}.tif")
                     for x in PARAM['file_suffix_lst']])
                file_name_tile[i]['seg'].append(
                    os.path.join(full_path, f"{file_prefix}_seg.tif"))
                file_name_tile[i]['aoi_key'].append(file_prefix)
                file_name_tile[i]['file_id'].append(i_prefix)
                file_name_tile[i]['sensor'].append(
                    PARAM['SENSOR_TYPE_dict'][i_prefix])

    # ------- merge files and create DataFrames as training input ---
    # Note: for cuml or scikit-learn input can be DataFrames or numpy
    #    arrays not images
    # Steps for merging include:
    # 1) open files
    # 2) if required renumber classes and remove specific classes (if MASK_NAN..)
    # 2) convert to float (class and greyscale).
    #    Note: match_project of the data to the labels is not required
    #          since this has been done in MAIN_extract_input_untiled.py
    # 3) merge arrays
    # 4) create DataFrame
    gdf_merge = {}
    feature_lst = {}
    class_lst = {}
    img_class_dict = {}
    # loop through phases
    for e, i in enumerate(PARAM['PHASE_NAME']):
        img_class_dict[i] = {}
        if not 'scikit' in PARAM['PARAM_PREP_ID']:
            # ------------ Read in data, merges and scales -------------
            # -- using custom_data_loader.py in same way as for CNN --
            # This ensures that separate stats are used for different
            # sensors, sites/AOIs and years (as specified with
            # PARAM['STATS_FILE'] and PARAM['PHASE_META_FILE']).
            # Note: "sensor" is added per default but will be selected
            # optionally further below with ml_classification_utils.prepare_df_sep()
            train_ds = custom_data_loader.CustomDataset_ScaleMerge_ML(
                file_name_tile[i]['data'], file_name_tile[i]['seg'],
                file_name_tile[i]['file_id'], PARAM['file_suffix_lst'],
                file_name_tile[i]['aoi_key'], file_name_tile[i]['sensor'],
                pca_calc_utils,
                PARAM['PATH_PROC'], PARAM['merge_bands'],
                x_bands=PARAM['X_BANDS'], y_bands=PARAM['Y_BANDS'],
                add_bands=PARAM['add_bands'],
                calc_pca=PARAM['calc_PCA'], PCA_components=PARAM['PCA_components'],
                standardize_add=PARAM['standardize_add'],
                norm_on_std=PARAM['norm_on_std'],
                norm_min_max_band1=PARAM['norm_min_max_band1'],
                take_log=PARAM['take_log'],
                gpu_no=GPU_lst[0], save_files_debug=True,
                dict_relabel=PARAM['DICT_RELABEL'],
                set_to_nan_lst=PARAM['MASK_TO_NAN_LST'],
                standardize_individual=PARAM['standardize_individual'],
                if_std_band1=PARAM['if_std_band1'],
                debug_plot=True, dl_phase=i, norm_clip=PARAM['norm_clip'])
            train_ds.get_stats_dict(
                PARAM['PATH_INP_BASE'], PARAM['STATS_FILE'],
                PARAM['PHASE_STATS_FILE'][CV_NUM])

            # get all items from dataloader
            n_jobs = get_n_jobs([file_name_tile[i]])
            w = Parallel(n_jobs=n_jobs, verbose=0)(delayed(
                train_ds.__getitem__)(k)
                for k in range(len(file_name_tile[i]['data'])))
        else:
            # --- Read in and preprocess with standard scikit learn ----
            # --- (for tsting and comparison) ----
            # !!! This is for testing as it does not standardise based
            # on mean and std from specific training areas per sensor or
            # site (as specified with PARAM['STATS_FILE']). Instead it
            # standardises based on the stats of the FULL merged training
            # data (self.X).
            # To get sensor and area specific standardisation, would need
            # to use the optopn above. However, the option
            # here allows band specific log or exp (with PARAM['log_dict_scikit'])
            # which is otherwise not implemented.

            # Here data is merged only (preprocessing is done later with
            # class_proc.preprocess_data())
            if PARAM['PARALLEL']:
                w = Parallel(n_jobs=n_jobs, verbose=0)(delayed(
                    ml_preproc_scikit.read_merge_extract)(
                        *k, PARAM['merge_bands'],
                        dict_relabel=PARAM['DICT_RELABEL'],
                        mask_lst=PARAM['MASK_TO_NAN_LST']) for k in zip(
                            file_name_tile[i]['data'],
                            file_name_tile[i]['glcm'],
                            file_name_tile[i]['seg'],
                            file_name_tile[i]['aoi_key'],
                            file_name_tile[i]['sensor']))
            else:
                w = []
                for k in zip(file_name_tile[i]['data'],
                             file_name_tile[i]['glcm'],
                            file_name_tile[i]['seg'],
                            file_name_tile[i]['aoi_key'],
                            file_name_tile[i]['sensor']):
                    w.append(ml_preproc_scikit.read_merge_extract(
                        *k, PARAM['merge_bands'],
                        dict_relabel=PARAM['DICT_RELABEL'],
                        mask_lst=PARAM['MASK_TO_NAN_LST']))

        gdf_merge[i] = pd.concat(list(zip(*w))[0], axis=0)
        [img_class_dict[i].update(x) for x in list(zip(*w))[1]]
        feature_lst[i] = list(zip(*w))[2]
        class_lst[i] = list(zip(*w))[3]
        gc.collect()

    # -------- create input for classification
    df_train, X_train, Y_train = ml_classification_utils.prepare_df_sep(
        gdf_merge['train'],
        feature_lst['train'][0].tolist() + PARAM['feature_add'],
        class_lst['train'][0], fmt_df=True, GPU_proc=GPU_proc)
    df_test, X_test, Y_test = ml_classification_utils.prepare_df_sep(
        gdf_merge['validate'],
        feature_lst['validate'][0].tolist() + PARAM['feature_add'],
        class_lst['validate'][0], fmt_df=True, GPU_proc=GPU_proc)

    # -------- check correlations of input features
    file_name = os.path.join(
        PARAM['PATH_PROC'], PARAM['FILE_PREFIX'] + '_correlation')
    dist_link = ml_classification_utils.get_feature_correlation(
        X_train.loc[:, feature_lst['train'][0]], file_name,
        feature_lst['train'][0], GPU_proc=GPU_proc)

    file_name = os.path.join(
        PARAM['PATH_PROC'], PARAM['FILE_PREFIX'] + '_correlation_group.txt')
    selected_feature_names = ml_classification_utils.extract_correlated_groups(
        dist_link, feature_lst['train'][0], correl_threshold=0.75,
        path_export=file_name)

    # ---------------- setup classifier -------------
    class_proc = ml_classification_utils.ClassifierML(
        X_train, Y_train, X_test, Y_test,
        feature_lst['train'][0].tolist() + PARAM['feature_add'],
        class_lst['train'][0],
        PARAM['PATH_PROC'], PARAM['FILE_PREFIX'], 'RandomForest',
        PARAM['CLASS_LABELS'][0][1:], PARAM['CLASS_LABELS'][1][1:],
        PARAM['dict_assign'], PARAM['PHASE_NAME'],
        PARAM['EPSG_TARGET'], preproc_type=PARAM['preproc_scikit'],
        n_jobs=PARAM['N_JOBS'],
        train_count=PARAM['start_count'],
        algo_param_gs=PARAM['algo_param_gs'],
        classifier_optimi=optimi_dict,
        GPU_proc=GPU_proc)

    # ---- load model (instead of training) ----
    if PARAM['load_model']:
        # this loads preprocessing pipeline and trained classifier
        class_proc.load_model_pkl(count=PARAM['start_count'])

    if 'scikit' in PARAM['PARAM_PREP_ID']:
        # This is not required if do preprocessing with
        # CustomDataset_ScaleMerge_ML() when reading in and directly
        # scaling data (which is advantageous if want to scale sensors
        # differently).

        # for single sensor and if want to use log_dict, the option below
        # can be used (here log allows more specific options)
        # --- preprocessing needs to be run separately even inf load model
        if GPU_proc:
            with cuml.using_output_type('cudf'):
            # self.X and self.X_test are adjusted !!!
                class_proc.preprocess_data(log_dict=PARAM['log_dict_scikit'])
        else:
            class_proc.preprocess_data(log_dict=PARAM['log_dict_scikit'])

    # ------------ convert to float32 -----
    # make sure that X, X_test, Y, Y_test are float32 or int32
    # (required or cuml)
    class_proc.convert_type()

    # ------- distribute input across workers ---
    # this is useful for Gridsearch where otherwise ther is not enough
    # memory on a single GPU
    if GPU_proc:
        workers = client.scheduler_info()["workers"]
        n_partitions = len(workers)
        class_proc.X, class_proc.Y = ml_classification_utils.distribute(
            class_proc.X, class_proc.Y, n_partitions, client, workers,
            GPU_proc)
        class_proc.X_test, class_proc.Y_test = ml_classification_utils.distribute(
            class_proc.X_test, class_proc.Y_test, n_partitions, client, workers,
            GPU_proc)

    if PARAM['hyper_param_tune'] and not PARAM['load_model']:
        # hyperparameter tuning
        for e_param, i_param in enumerate(PARAM['param_dict']):
            class_proc.run_full_hyper_tuning_wf(i_param)
    else:
        # training and prediction
        if not PARAM['load_model']:
            PARAM['param_inp'] = {'RandomForest': dict(
                n_estimators=int(PARAM['n_estimators']),
                max_depth=int(PARAM['max_depth']),
                max_samples=PARAM['max_samples'],
                max_features=PARAM['max_features'],
                bootstrap=PARAM['bootstrap'])}
            class_proc.resetup_with_param(PARAM['param_inp'])
            class_proc.train_all()

            # save model
            class_proc.save_model_pkl('retrain')
            # save model trained on GPU to be usable on CPU for prediction later
            class_proc.save_model_cpu()

    if PARAM['predict']:
        class_proc.run_full_prediction(
            gdf_merge['train'].loc[:, ['x', 'y', 'aoi_key']], df_train,
            gdf_merge['validate'].loc[:, ['x', 'y', 'aoi_key']],
            df_test)

        # -------- save metrics and info --------
        out_suffix = 'train_validate'  # '_'.join(class_proc.pred_files_df.keys()) + '_pred'
        class_proc.summarize_save_metrics(out_suffix=out_suffix)

        # ------ save prediction comparison -----
        pred_files_df = class_proc.pred_files_df['validate'][class_proc.train_count]
        for i_aoi, i_file in zip(file_name_tile['validate']['aoi_key'],
                                file_name_tile['validate']['seg']):
            PRED_INP = os.path.join(
                PARAM['PATH_PROC'], pred_files_df[i_aoi][0])
            PREFIX_OUT = os.path.join(
                PARAM['PATH_PROC'], pred_files_df[i_aoi][1])
            gdf_pred, gdf_true = MAIN_extract_segmentation_properties.main(
                PRED_IMG=PRED_INP + '.tif', TRUE_IMG=i_file,
                AOI_PATH=None, EPSG=PARAM['EPSG_TARGET'],
                DICT_RELABEL=PARAM['DICT_RELABEL'],
                MASK_TO_NAN_LST=PARAM['MASK_TO_NAN_LST'],
                PREFIX_OUT=PREFIX_OUT,
                additionally_save_with_min_px_size=3)

    # ------- calculate feature importance
    if PARAM['calc_feature_importance']:
        for e_param, i_param in enumerate(PARAM['param_dict']):
            class_proc.run_feature_importance(
                i_param, selected_feature_names)

    # ---- run test on additional patches
    # change epxort path
    if PARAM['run_test']:
        # --- create input for test
        for i_phase in PARAM['PHASE_NAME'][2:]:
            # prepare new data from test sets
            class_proc.path_export = os.path.join(
                PARAM['PATH_PROC'], i_phase)
            if not os.path.isdir(class_proc.path_export):
                os.mkdir(class_proc.path_export)
            df_test_new, X_test_new, Y_test_new = ml_classification_utils.prepare_df_sep(
                gdf_merge[i_phase],
                feature_lst[i_phase][0].tolist() + PARAM['feature_add'],
                class_lst[i_phase][0], fmt_df=True, GPU_proc=GPU_proc)

            if not 'scikit' in PARAM['PARAM_PREP_ID']:
                # processing not required here since was already done
                # with data loader
                class_proc.X_test = X_test_new
                class_proc.Y_test = Y_test_new
            else:
                # if did not use dataloader need here to add and
                # preprocess test input
                # (this replaces self.X_test and self.Y_test)
                # --- preprocessing needs to be run separately even inf
                # load model
                with cuml.using_output_type('cudf'):
                    # self.X_test and self.Y_test are adjusted !!!
                    class_proc.update_preprocess_test_data(
                        X_test_new, y_test_new=Y_test_new,
                        log_dict=PARAM['log_dict_scikit'], phase_suffix=i_phase)

            # ----- convert to float32 -----
            # make sure that X, X_test, Y, Y_test are float32 or int32
            # (required or cuml)
            class_proc.convert_type()

            # --- distribute across workers ---
            # if use client with GPUs
            if GPU_proc:
                workers = client.scheduler_info()["workers"]
                n_partitions = len(workers)
                class_proc.X_test, class_proc.Y_test = ml_classification_utils.distribute(
                    class_proc.X_test, class_proc.Y_test, n_partitions,
                    client, workers, GPU_proc)

            # -- predict and evaluate
            class_proc.predict_all(
                gdf_merge[i_phase].loc[:, ['x', 'y', 'aoi_key']],
                df_test_new, phase=i_phase)

            # ------ save prediction comparison -----
            pred_files_df = class_proc.pred_files_df[i_phase][class_proc.train_count]
            for i_aoi, i_file in zip(file_name_tile[i_phase]['aoi_key'],
                                     file_name_tile[i_phase]['seg']):
                PRED_INP = os.path.join(
                    PARAM['PATH_PROC'], i_phase, pred_files_df[i_aoi][0])
                PREFIX_OUT = os.path.join(
                    PARAM['PATH_PROC'], i_phase, pred_files_df[i_aoi][1])
                gdf_pred, gdf_true = MAIN_extract_segmentation_properties.main(
                    PRED_IMG=PRED_INP + '.tif',
                    TRUE_IMG=i_file,
                    AOI_PATH=None, EPSG=PARAM['EPSG_TARGET'],
                    DICT_RELABEL=PARAM['DICT_RELABEL'],
                    MASK_TO_NAN_LST=PARAM['MASK_TO_NAN_LST'],
                    PREFIX_OUT=PREFIX_OUT,
                    additionally_save_with_min_px_size=3)
                for i_class_eval in PARAM['class_eval_weighted_TP_TN_lst']:
                    df_cts = MAIN_extract_TP_TN_per_class_uncert.main(
                        PRED_IMG=PRED_INP + '.tif',
                        TRUE_IMG=i_file,
                        CLASS_TO_EVAL=i_class_eval,
                        AOI_PATH=None, EPSG=PARAM['EPSG_TARGET'],
                        DICT_RELABEL=PARAM['DICT_RELABEL'],
                        MASK_TO_NAN_LST=PARAM['MASK_TO_NAN_LST'],
                        PREFIX_OUT=PREFIX_OUT, min_px_size=None)

            # -------- save metrics and info --------
            out_suffix = 'train_validate_test'  #'_'.join(class_proc.pred_files_df.keys()) + '_pred'
            class_proc.summarize_save_metrics(out_suffix=out_suffix)

    if GPU_proc:
        monitoring_utils.get_GPU_memory_info(GPU_lst)

        def collect():
            gc.collect()

        gc.collect()
        if GPU_proc:
            client.run(collect)
            try:
                print('restart cluster')  # ensures removeind al data on cluster
                client.restart()  # deletes all remaining copies on client
            except:
                pass

        monitoring_utils.get_GPU_memory_info(GPU_lst)
    print('CV end: ' + add_prefix)
    print('====== END ' + dt.datetime.now().strftime('%Y-%m-%d_%H%M') +'=====')
    # ---- save time stats
    time_file = '_'.join(
        ['A', PARAM['FILE_PREFIX'], 'time_stats'])
    monitoring_utils.save_time_stats(
        prof, PARAM['PATH_PROC'], time_file)
    return


def get_n_jobs(loop_items):
    n_inp = [len(x) for x in loop_items]
    out = min(int(cpu_count()/10), np.prod(n_inp))
    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('path to processing folder (2_segment folder)'))
    parser.add_argument(
        'PROC_STEP', type=str,
        help=('name processing step: "rt": for train test, '
              + '"gs": hyperparam tuning, "fi": feature importance analysis'))
    parser.add_argument(
        'PARAM_PREP_ID', type=str,
        help=('id of merge and scaling parameters e.g. vML001 ' +
              '(PARAM_inp_ML_feature_prep_v01.txt)'))
    parser.add_argument(
        'PARAM_TRAIN_ID', type=str,
        help=('id of training parameters e.g. tML01 ' +
              '(PARAM_inp_ML_train_v01.txt)'))
    parser.add_argument(
        'PARAM_FILE', type=str,
        help='name of parameter file')
    parser.add_argument(
        '--GPU_LST_STR', type=str,
        help='list of GPUs (separated by ":")', default='0')
    parser.add_argument(
        '--N_JOBS', type=int,
        help='amount of jobs to run in parallel (important if run on CPU)', default=6)
    parser.add_argument(
        '--CV_NUM', type=int,
        help=('CV number if do not want to start at 0'), default=0)


    args = parser.parse_args()

    # set up GPU devices
    inp_GPU = vars(args)['GPU_LST_STR'].split(':')
    GPUs = ','.join([str(x) for x in inp_GPU])
    # !!! CUDA_VISIBLE_DEVICES needs to be set before import torch
    # otherwise torch seems to select GPU 0 in addition
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

    import torch
    if torch.cuda.is_available():
        # set up client for GPU usage
        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster
        # setup CUDA cluster and limit GPU device memory usage to 95%
        with LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=GPUs, device_memory_limit=0.95) as cluster, Client(cluster) as client:
            import utils.pca_calc_utils as pca_calc_utils
            main(pca_calc_utils, vars(args), CV_NUM=vars(args)['CV_NUM'],
                client=client)
    else:
        # set up for CPU usage
        import utils.pca_calc_utils as pca_calc_utils
        main(pca_calc_utils, vars(args), CV_NUM=vars(args)['CV_NUM'],
                client=None)




