'''
== Test function to split imagery into prediction tiles for CNN inference ==
== (proc step 03) ==

'''

import os
import sys
import argparse

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)
from MAIN_create_prediction_tiles import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(
    path_proc_inp, '1_site_preproc', 'BLyaE_v1')

# set other input
PROJ_PARAM_FILE = 'PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01'
prediction_area = 'all'
PARAM_FILE = 'PARAM03_extract_data'
SCALE_TYPE = 'perc02_g03_8bit'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create prediction tiles')
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
        '--prediction_area', type=str,
        help='name of .geojson AOI-file to tile if "all" PARAM["AOI_full_area"] is taken. Several areas can be provided by separating .geojson files with ":"', default='all')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file',
        default='PARAM03_extract_data')


    args = parser.parse_args(
        ['--prediction_area', prediction_area,
         '--PARAM_FILE', PARAM_FILE,
         PATH_PROC, PROJ_PARAM_FILE, SCALE_TYPE])

    # loop through different prediction areas
    label_area_lst = vars(args)['prediction_area'].split(':')
    inp_param = vars(args)
    for i in label_area_lst:
        inp_param['prediction_area'] = i
        main(vars(args))

