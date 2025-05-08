"""
================ Test function for RF classification  ===========\n
======================== (proc step 06) =========\n
"""
import os
import sys
import argparse

# ----------------- PATHS & GLOBAL PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)


# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(path_proc_inp, '2_segment')

PARAM_FILE = 'PARAM06_RFtrain_HEX1979_A02'
PARAM_PREP_ID = 'vML080'
PARAM_TRAIN_ID = 'tML99'
GPU_LST_STR = '99'
N_JOBS = '20'
PROC_STEP = 'rt'
CV_NUM = '1'


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
        help='list of GPUs (separated by ":")', default='0')
    parser.add_argument(
        '--N_JOBS', type=int,
        help='amount of jobs to run in parallel (important if run on CPU)', default=6)
    parser.add_argument(
        '--CV_NUM', type=int,
        help=('CV number if do not want to start at 0'), default=0)

    args = parser.parse_args(
        ['--GPU_LST_STR', GPU_LST_STR,
         '--N_JOBS', N_JOBS,
         '--CV_NUM', CV_NUM,
         PATH_PROC, PROC_STEP, PARAM_PREP_ID, PARAM_TRAIN_ID,
         PARAM_FILE])

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
            #import MAIN_RF_classify
            import utils.pca_calc_utils as pca_calc_utils
            import MAIN_RF_classify
            MAIN_RF_classify.main(
                pca_calc_utils, vars(args), CV_NUM=vars(args)['CV_NUM'],
                client=client)
    else:
        # set up for CPU usage
        import utils.pca_calc_utils as pca_calc_utils
        import MAIN_RF_classify
        MAIN_RF_classify.main(
            pca_calc_utils, vars(args), CV_NUM=vars(args)['CV_NUM'],
            client=None)

