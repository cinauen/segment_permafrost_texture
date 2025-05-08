'''

'''

import os
import sys
import argparse

# ----------------- PATHS & INPUT PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)
from MAIN_augment_calc_texture import *

# set paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_PROC = os.path.join(
    path_proc_inp, '1_site_preproc', 'BLyaE_v1')

# set other input
PROJ_PARAM_FILE = 'PROJ_PARAM_BLyaE_HEX1979_v01'
PARAM_FILE = 'PARAM04_augment_data'
GPU_DEVICE = '5'
tiling_area = 'A02'  # 'A01:A02'
N_JOBS = '20'  # how many parallel jobs !!! needs to fit into GPU memory !!!
SCALE_TYPE = 'perc0-2_g0-3_8bit'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Augment and calculate texture of tiles')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('Path to site/project processing folder'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('Name of file with project parameters (.py file without file extension)'))
    parser.add_argument(
        'tiling_area', type=str,
        help='name of tiling area or several names separated by ":" e.g. A01:A02. This relates to area name in subfolder.')
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used (e.g. std4_8bit)')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='Name of file with processing step specific parameters (.py file without extension)',
        default='PARAM04_augment_data')
    parser.add_argument(
        '--GPU_DEVICE', type=int,
        help='GPU device number to use for texture calculation',
        default=0)
    parser.add_argument(
        '--N_JOBS', type=int,
        help='how many parallel jobs for texture calc (!!! needs to fit into GPU memory !!!)',
        default=20)


    args = parser.parse_args(
        ['--PARAM_FILE', PARAM_FILE,
         '--GPU_DEVICE', GPU_DEVICE,
         '--N_JOBS', N_JOBS,
         PATH_PROC, PROJ_PARAM_FILE, tiling_area, SCALE_TYPE])

    tiling_area_lst = vars(args)['tiling_area'].split(':')
    inp_param = vars(args)

    # For GPU device initialization use environmental variable. This
    # ensures that only the specified device is used.
    # This is required because importing cucim (if installed) within
    # glcm_cupy seems to initialize all GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vars(args)['GPU_DEVICE'])
    import cupy
    with cupy.cuda.Device(0):  # use zero here since only one device visible
        for i in tiling_area_lst:
            # loop through different labelling areas
            inp_param['tiling_area'] = i
            main(vars(args))

