'''
=== Test function to split imagery into trainig tiles (proc step 03) ====

'''

import os
import sys
import argparse

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)
from MAIN_create_training_tiles import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(path_proc_inp, '1_site_preproc', 'BLyaE_v1')

# set other input
PROJ_PARAM_FILE = 'PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01'
PARAM_FILE = 'PARAM03_extract_data'
labelling_area = 'A02'  # training are for which to create training tiles
SCALE_TYPE = 'perc02_g03_8bit'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training tiles')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder and proc file'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('file with project parameters'))
    parser.add_argument(
        'labelling_area', type=str,
        help='name of labelling area or several names separated by : (e.g. A01:A02, refers to dict key in PARAM["LABEL_FILE_INP"])')
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used (e.g. std4_8bit)')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file',
        default='PARAM03_extract_data')

    args = parser.parse_args(
        ['--PARAM_FILE', PARAM_FILE,
         PATH_PROC, PROJ_PARAM_FILE, labelling_area, SCALE_TYPE])

    # loop through different labelling areas
    label_area_lst = vars(args)['labelling_area'].split(':')
    inp_param = vars(args)
    for i in label_area_lst:
        inp_param['labelling_area'] = i
        main(vars(args))
