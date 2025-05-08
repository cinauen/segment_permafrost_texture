'''
===== Test function for image pre-processing (proc step 01) =====

'''
import os
import sys
import argparse

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)
from MAIN_img_preproc import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(
    path_proc_inp, '1_site_preproc', 'FadN_v1')

# set other input
PROJ_PARAM_FILE = 'PROJ_PARAM_FadN_HEX1980_v01'
PARAM_FILE = 'PARAM01_img_preproc'
PARALLEL_PROC = '0'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process single images')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder (required, e.g. ".docs/1_site_preproc/BLyaE_v1")'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('name of project/site specific parameter file (required, e.g.: PROJ_PARAM_BLyaE_HEX1979_v01)'))
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file (optional, default: PARAM01_img_preproc)',
        default='PARAM01_img_preproc')
    parser.add_argument(
        '--PARALLEL_PROC', type=int,
        help='if want to use parallel processing (optional, default: 1)',
        default=1)

    args = parser.parse_args(
        ['--PARAM_FILE', PARAM_FILE,
         '--PARALLEL_PROC', PARALLEL_PROC,
         PATH_PROC, PROJ_PARAM_FILE])

    main(vars(args))
