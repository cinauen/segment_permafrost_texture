'''
===== Calculate texture for a specified raster files =====

!!! This script is called as subprocess in data loader for online
augmentation to recalculate GLCM after augmentation !!!!
(cnn_workflow/cnn_workflow/custom_data_loader.CustomDataset_augment_ScaleMerge())

No pre-processing is done within this function. Thus, all files need to
have correct dtypes (int with correct bit depth) and nans must have been
replaced with nan previousely.

'''
import copy
import os
import sys
import numpy as np
import argparse

import memory_profiler


# ----------------- PATHS & GLOBAL PARAM -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import utils.file_utils as file_utils
import utils.geo_utils as geo_utils
import param_settings.param_utils as param_utils


def main(inp):
    '''
    subfiles are saved as tif files.
    These are e.g. a0-1-2-3_r05_norm_C01
    and for cross_std r05_calc_std (created from non saved
        a0_r05, a1_r05, a2_r05, a3_r05)

    # required parameters:
    --- function input
        PARAM['DATA_FILE'] (from metadata files): file-path to input data file
        --> dir name will be used as location of output files

        PARAM['FILE_SUFFIX_LST'] (from PARAM_inp_merge05.txt):
            a0-1-2-3_r02_norm_C01:a0-1-2-3_r05_norm_C01:a0-1-2-3_r10_norm_C01:r05_calc_std

    --- function input (taken from proj param file)
        PARAM['GPU_NUM']  #
        PARAM['SCALE_TYPE']

    --- in texture param file:
        PARAM['x_band_GLCM']  # x_band for input to calculate GLCM
        PARAM['PADDING_CONST']
        PARAM['BIN_FROM']
        PARAM['BIN_TO']
        PARAM['TEX_SUFFIX']
        PARAM['cross_calc']

        PARAM['TEX_PARAM_inp'] = {
            'r01': [[0, 45, 90, 135], 3, 3],
            'r02': [[0, 45, 90, 135], 5, 5],
            ...}

        PARAM['img_cross_calc_inp'] = {
            'r01': [[[x], 3, 3] for x in [0, 45, 90, 135]],
            'r02': [[[x], 5, 5] for x in [0, 45, 90, 135]],
            ...}
    '''
    import texture_calc.FUNC_calc_texture_from_img as FUNC_calc_texture_from_img
    # --- add input parameters
    PARAM = {}
    for key, val in inp.items():
        PARAM[key] = val
    if PARAM['PATH_OUT'] == 'None':
        PARAM['PATH_OUT'] = os.path.dirname(PARAM['DATA_FILE'])

    param_utils.add_proc_param(PARAM)

    # DEBUG is per default set to False as this main function is run
    # as a subprocess within the data loader...
    if PARAM['DEBUG']:
        # redirect memory profiling output to console
        # @profile decorator needs to be running
        sys.stdout = memory_profiler.LogFile('log')

        # create log file and initialize errors to be written to console
        PARAM['LOG_FILE_SUFFIX'] = 'calc_texture_in_train_loop'
        log_file_name =f"A_{PARAM['LOG_FILE_SUFFIX']}_error.log"
        file_utils.init_logging(
            log_file=os.path.join(PARAM['PATH_OUT'], log_file_name),
            logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']}")

    # read raster file form path
    x_img = geo_utils.read_rename_according_long_name(
        PARAM['DATA_FILE'], mask_nan=False)

    # select band for GLCM calculation
    x_img = x_img.sel(band=PARAM['x_band_GLCM'])

    # create numpy array with correct dimension for GLCM calculation input
    x_np = x_img.values
    x_np = np.moveaxis(x_np, 0, -1)

    PARAM['FILE_SUFFIX_LST'] = PARAM['FILE_SUFFIX_LST'].split(':')

    # prepare texture parameters (windows and directions) based on
    # FILE_SUFFIX_LST
    tex_search = [
        x.split('_norm')[0][-3:] for x in PARAM['FILE_SUFFIX_LST']
        if x.find('norm') > -1]
    PARAM['TEX_PARAM'] = [
        PARAM['TEX_PARAM_inp'][x_val] for x_val in tex_search]

    # prepare cross stats parameters (windows and directions) based on
    # FILE_SUFFIX_LST
    cross_calc_search = [
        x.split('_calc')[0][-3:] for x in PARAM['FILE_SUFFIX_LST']
        if x.find('calc') > -1]
    PARAM['img_cross_calc'] = {
        x: PARAM['img_cross_calc_inp'][x] for x in cross_calc_search}

    PARAM['EPSG_TARGET'] = None  # is not required

    # filename and path for output
    file_prefix = os.path.basename(PARAM['DATA_FILE']).split('.')[0]

    # calculate texture
    with cupy.cuda.Device(0):  # use zero here since set only one device to be visible
        FUNC_calc_texture_from_img.calc_texture(
            x_img, PARAM['PATH_OUT'], file_prefix, copy.deepcopy(PARAM),
            img_np=x_np)

        FUNC_calc_texture_from_img.texture_cross_calc(
            x_img, PARAM['PATH_OUT'], file_prefix, copy.deepcopy(PARAM),
            img_np=x_np)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process single images')
    parser.add_argument(
        'DATA_FILE', type=str,
        help=('path to input data file, basedir will be used as save location'))
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
        help=('GPU device number for GLCM calc'), default=7)
    parser.add_argument(
        '--PATH_OUT', type=str,
        help=('output path'), default='None')

    args = parser.parse_args()
    # there seems to be an issue with the GPU initialisation
    # therefore set here CUDA_VISIBLE_DEVICES to make sure that
    # correct GPU is used
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vars(args)['GPU_NUM'])

    import cupy
    with cupy.cuda.Device(0):  # use zero here since only one device visible
        main(vars(args))
