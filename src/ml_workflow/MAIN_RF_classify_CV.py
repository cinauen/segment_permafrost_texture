"""
====== By looping through all cross-validation options: ======
=================== ML classification ====================
(run with: ml_workflow/MAIN_RF_classify.py)

--- Run script in command line
usage: MAIN_RF_classify_CV.py [-h] [--GPU_LST_STR GPU_LST_STR] [--N_JOBS N_JOBS] PATH_PROC PROC_STEP PARAM_PREP_ID PARAM_TRAIN_ID PARAM_FILE

positional arguments:
  PATH_PROC         path to processing folder (2_segment folder)
  PROC_STEP         name processing step:
                        "rt": for train test
                        "gs": hyperparam tuning
                        "fi": feature importance analysis
  PARAM_PREP_ID     id for merge and scaling parameters
                        e.g. vML001 (PARAM_inp_ML_feature_prep_v01.txt)
  PARAM_TRAIN_ID    id of training parameters
                        e.g. tML01 (PARAM_inp_ML_train_v01.txt)
  PARAM_FILE        name of parameter file

options:
  -h, --help        show this help message and exit
  --GPU_LST_STR GPU_LST_STR
                    list of GPUs (separated by ":")
  --N_JOBS N_JOBS   amount of jobs to run in parallel
"""

import os
import sys
import importlib
import argparse


# ----------------- import custom utils -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def main(PARAM_inp):
    """
    Runs training by looping through all cross-validation options
    """
    # add path for project parameter file
    PARAM = PARAM_inp.copy()
    PARAM['PATH_INP'] = os.path.normpath(
        os.path.join(PARAM['PATH_PROC'], '01_input'))
    sys.path.append(PARAM['PATH_INP'])

    # pre-read parameter file
    # import processing step specific parameters
    param_module = importlib.import_module(PARAM['PARAM_FILE'])
    param_module.get_param(PARAM)

    for e_cv in range(len(PARAM['PHASE_META_FILE'])):
        # set up GPU device usage
        inp_GPU = PARAM['GPU_LST_STR'].split(':')
        GPUs = ','.join([str(x) for x in inp_GPU])
        # !!! CUDA_VISIBLE_DEVICES needs to be set before import torch
        # otherwise torch seems to select GPU 0 in addition
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

        import torch
        if torch.cuda.is_available():
            # set up client for GPU usage
            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client, wait
            # setup CUDA cluster and limit GPU device memory usage to 95%
            with LocalCUDACluster(
                CUDA_VISIBLE_DEVICES=GPUs, device_memory_limit=0.95) as cluster, Client(cluster) as client:
                import MAIN_RF_classify
                import utils.pca_calc_utils as pca_calc_utils
                MAIN_RF_classify.main(
                    pca_calc_utils, PARAM_inp, CV_NUM=e_cv, client=client)
        else:
            import utils.pca_calc_utils as pca_calc_utils
            import MAIN_RF_classify
            MAIN_RF_classify.main(
                pca_calc_utils, PARAM_inp, CV_NUM=e_cv, client=None)

    return


if __name__ == "__main__":
    # general inputs required for MAIN_classify
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
        help=('id for merge and scaling parameters e.g. vML001 ' +
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
        help='list of GPUs (separated by ":")', default='7')
    parser.add_argument(
        '--N_JOBS', type=int,
        help='amount of jobs to run in parallel', default=7)

    args = parser.parse_args()

    main(vars(args))

