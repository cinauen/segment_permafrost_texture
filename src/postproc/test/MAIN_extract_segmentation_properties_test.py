'''

'''


import os
import sys
import argparse

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)

from MAIN_extract_segmentation_properties import *

BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = path_proc_inp

PRED_IMG = os.path.join(
    PATH_PROC,
    '2_segment/02_train/BLyaE_v1_HEX1979_A02_ML_train/vML006_tML01_cv00',
    'BLyaE_HEX1979_A01_train-03_tML01_cv00_test_pred_count01.tif')
TRUE_IMG = os.path.join(
    PATH_PROC,
    '1_texture_proc/BLyaE_v1/03_train_inp/BLyaE_HEX1979_A02_perc0-2_g0-3_8bit_Lv01_ML_v00',
    'BLyakhE_HEX1979_A01_train-03_seg.tif')

AOI_PATH = 'None'
EPSG = '32654'
MASK_TO_NAN_LST = 'None'
DICT_RELABEL = '{}'  # {from: to}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process single images')
    parser.add_argument(
        'PRED_IMG', type=str,
        help=('PATH of predicted image (if use not as direct function...)'))
    parser.add_argument(
        'TRUE_IMG', type=str,
        help=('PATH of ground truth image (if use not as direct function...)'))
    parser.add_argument(
        '--AOI_PATH', type=str,
        help='path to AOI file if need clipping', default=None)
    parser.add_argument(
        '--EPSG', type=int, help='EPSG', default=32654)
    parser.add_argument(
        '--DICT_RELABEL', type=str,
        help='if need relabeling of classes',
        default='{}')
    parser.add_argument(
        '--MASK_TO_NAN_LST', type=str,
        help='if some classes should be set to None',
        default=None)

    args = parser.parse_args(
        ['--AOI_PATH', AOI_PATH,
         '--EPSG', EPSG,
         '--DICT_RELABEL', DICT_RELABEL,
         '--MASK_TO_NAN_LST', MASK_TO_NAN_LST,
         PRED_IMG, TRUE_IMG])

    main(**vars(args))




