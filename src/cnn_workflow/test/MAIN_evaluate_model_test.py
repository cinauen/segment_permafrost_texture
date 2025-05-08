"""
== Test function for testing the trained models in the test patches ==\n
== (proc step 06) ==\n
"""

import os
import sys
import argparse

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)

from MAIN_evaluate_model import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()


PATH_PROC = os.path.join(
    path_proc_inp, '2_segment')

PARAM_PREP_ID = 'v079'
PARAM_TRAIN_ID = 't16off'
PARAM_FILE = 'PARAM06_test_model_BLyaE_v1_HEX1979_A02_set01a'
GPU_LST_STR = '5'
CV_NUM = 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train model with on the fly augmentation')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to base processing'))
    parser.add_argument(
        'PARAM_PREP_ID', type=str,
        help=('id for merge and scaling parameters e.g. v01 (from train_prep, pred_prep)'))
    parser.add_argument(
        'PARAM_TRAIN_ID', type=str,
        help=('id for training parameters e.g. r01'))
    parser.add_argument(
        'PARAM_FILE', type=str, help='name of project parameter file')
    parser.add_argument(
        '--GPU_LST_STR', type=str,
        help='list of GPUs (separated by ":")', default='0')
    parser.add_argument(
        '--CV_NUM', type=int,
        help=('CV number if do not want to start at 0'), default=0)

    args = parser.parse_args(
        ['--GPU_LST_STR', GPU_LST_STR,
         '--CV_NUM', CV_NUM,
         PATH_PROC, PARAM_PREP_ID,
         PARAM_TRAIN_ID, PARAM_FILE])

    import utils.pca_calc_utils as pca_calc_utils
    main(pca_calc_utils, vars(args), CV_NUM=CV_NUM)

