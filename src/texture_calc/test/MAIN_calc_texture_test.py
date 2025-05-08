'''
======== Test function for texture calculation on full image ========
=====================  (proc step 02) =====================
'''

import os
import sys
import argparse

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)
from MAIN_calc_texture import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(path_proc_inp, '1_site_preproc', 'BLyaE_v1')

# set other input
PROJ_PARAM_FILE = 'PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01'
PARAM_FILE = 'PARAM02_calc_texture'
GPU_DEVICE = '7'
SCALE_TYPE = 'perc02_g03_8bit'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process single images')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder and proc file'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('file with project parameters'))
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used (e.g. std4_8bit)')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file',
        default='PARAM02_calc_texture')
    parser.add_argument(
        '--GPU_DEVICE', type=int,
        help='GPU device number', default=0)

    args = parser.parse_args(
        ['--PARAM_FILE', PARAM_FILE,
         '--GPU_DEVICE', GPU_DEVICE,
         PATH_PROC, PROJ_PARAM_FILE, SCALE_TYPE])

    # For GPU device initialization use environmental variable. This
    # ensures that only the specified device is used.
    # This is required because importing cucim (if installed) within
    # glcm_cupy seems to initialize all GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vars(args)['GPU_DEVICE'])
    import cupy
    with cupy.cuda.Device(0):  # use zero here since only one device visible
        main(vars(args))
