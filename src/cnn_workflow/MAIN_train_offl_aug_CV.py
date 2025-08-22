"""
====== By looping through all cross-validation options: ======
============ Training after affline augmentation ===============
(training is run with: cnn_workflow/MAIN_train_offl_aug.py)

For detailed function description see: cnn_workflow/MAIN_train_offl_aug.py

--- Run script in command line
usage: MAIN_train_offl_aug_CV.py [-h] [--GPU_LST_STR GPU_LST_STR] [--GPU_num_GLCM GPU_NUM_GLCM] PATH_PROC PARAM_PREP_ID PARAM_TRAIN_ID PARAM_FILE


positional arguments:
  PATH_PROC             path to processing folder (2_segment folder)
  PARAM_PREP_ID         id for merge and scaling parameters e.g. v079 (from train_prep, pred_prep)
  PARAM_TRAIN_ID        id for training parameters e.g. t16onl
  PARAM_FILE            name of framework parameter file

options:
  -h, --help            show this help message and exit
  --GPU_LST_STR GPU_LST_STR
                        list of GPUs (separated by ":")
  --PATH_INP_BASE PATH_INP_BASE
                        path to site pre processing folder as default
                        (None) the path to 1_site_preproc is used

"""

import os
import sys
import importlib
import argparse


# ----------------- import custom utils -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import MAIN_train_offl_aug


def main(pca_calc_utils, PARAM_inp):
    """
    Runs training by looping through all cross-validation options
    """
    # add path for project parameter file
    PARAM = PARAM_inp.copy()
    PARAM['PATH_INP'] = os.path.normpath(
        os.path.join(PARAM['PATH_PROC'], '01_input'))
    sys.path.append(PARAM['PATH_INP'])

    # pre-read parameter file
    # import processing step specific parameters
    param_module = importlib.import_module(PARAM['PARAM_FILE'])
    param_module.get_param(PARAM)

    for e_cv in range(len(PARAM['PHASE_META_FILE'])):
        MAIN_train_offl_aug.main(pca_calc_utils, PARAM_inp, e_cv)

    return


if __name__ == "__main__":

    # general inputs required for MAIN_train_incl_aug
    parser = argparse.ArgumentParser(description='Train model after offline augmentation, loop through CV')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('path to processing folder (2_segment folder)'))
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
        help='list of GPUs (separated by ":")', default='7')
    parser.add_argument(
        '--PATH_INP_BASE', type=str,
        help=('path to site pre processing folder as default (None) '
             'the path to 1_site_preproc is used'),
        default='None')

    args = parser.parse_args()

    import utils.pca_calc_utils as pca_calc_utils
    main(pca_calc_utils, vars(args))

