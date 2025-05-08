'''

'''

import os
import sys
import argparse


# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)
from MAIN_train_incl_aug_CV import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(path_proc_inp, '2_segment')
PATH_LOCAL = path_temp_inp  # temp folder

PARAM_FILE = 'PARAM06_train_BLyaE_v1_HEX1979_A01_A02_set01a'
PARAM_PREP_ID = 'v079'
PARAM_TRAIN_ID = 't16onl'
GPU_LST_STR = '1:2'
GPU_num_GLCM = '2'


if __name__ == "__main__":

    # general inputs required for MAIN_train_incl_aug
    parser = argparse.ArgumentParser(description='train model with on the fly augmentation')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('path to processing folder (2_segment folder)'))
    parser.add_argument(
        'PATH_LOCAL', type=str,
        help=('path where to save temp files'))
    parser.add_argument(
        'PARAM_PREP_ID', type=str,
        help=('id for merge and scaling parameters e.g. v079 ' +
              '(from train_prep, pred_prep)'))
    parser.add_argument(
        'PARAM_TRAIN_ID', type=str,
        help=('id for training parameters e.g. t16onl'))
    parser.add_argument(
        'PARAM_FILE', type=str, help='name of framework parameter file')
    parser.add_argument(
        '--GPU_LST_STR', type=str,
        help='list of GPUs (separated by ":")', default='0')
    parser.add_argument(
        '--GPU_num_GLCM', type=int,
        help=('GPU number for GLCM calc'), default=0)

    args = parser.parse_args(
        ['--GPU_LST_STR', GPU_LST_STR,
         '--GPU_num_GLCM', GPU_num_GLCM,
         PATH_PROC, PATH_LOCAL, PARAM_PREP_ID,
         PARAM_TRAIN_ID, PARAM_FILE])

    import utils.pca_calc_utils as pca_calc_utils
    main(pca_calc_utils, vars(args))
