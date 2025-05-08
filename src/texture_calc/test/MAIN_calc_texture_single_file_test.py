'''
===== Test function to calculate texture fr a specified raster file ====
'''

import os
import sys
import argparse
import tempfile

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)

from MAIN_calc_texture_single_file import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(path_proc_inp, '1_site_preproc', 'BLyaE_v1')

# set other input
DATA_FILE = os.path.join(
    PATH_PROC,
    '03_train_inp/BLyaE_HEX1979_A01_perc0-2_g0-3_8bit_Lv01_w298-298_v00_rot_cubic/BLyaE_HEX1979_A02_train-01_00_00-00_data.tif')
FILE_SUFFIX_LST = 'a0-1-2-3_r02_norm_C01:a0-1-2-3_r05_norm_C01:r05_calc_std'
SCALE_TYPE = 'std4_8bit'
PARAM_FILE = 'PARAM06_calc_texture_train'
GPU_NUM = '7'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process single images')
    parser.add_argument(
        'DATA_FILE', type=str,
        help=('file grey scale imag'))
    parser.add_argument(
        'FILE_SUFFIX_LST', type=str,
        help=('string with list (split :) of GLCM suffixes e.g. a0-1-2-3_r02_norm_C01:a0-1-2-3_r05_norm_C01:r05_calc_std'))
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help=('scale type string e.g. perc_8bit'))
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of parameter file', default='PARAM06_calc_texture_train')
    parser.add_argument(
        '--GPU_NUM', type=int,
        help=('GPU nuber of GLCM calc'), default=7)
    parser.add_argument(
        '--PATH_OUT', type=str,
        help='output path', default='None')

    temp_path_prefix = os.path.join(os.path.dirname(DATA_FILE), 'temp_')
    with tempfile.TemporaryDirectory(prefix=temp_path_prefix) as tempdir:
        args = parser.parse_args(
            ['--PARAM_FILE', PARAM_FILE,
            '--GPU_NUM', GPU_NUM,
            '--PATH_OUT', tempdir,
            DATA_FILE, FILE_SUFFIX_LST, SCALE_TYPE])

        os.environ["CUDA_VISIBLE_DEVICES"] = str(vars(args)['GPU_NUM'])
        import cupy
        with cupy.cuda.Device(0):  # use zero here since only one device visible
            main(vars(args))
            print('test')

