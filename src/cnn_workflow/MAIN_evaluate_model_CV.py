'''
====== By looping through all cross-validation options: ======
============ Evaluate model on test patches ===============
(evaluation is run with: cnn_workflow/MAIN_evaluate_model.py)

--- Run script in command line
usage: MAIN_evaluate_model_CV.py [-h] [--GPU_LST_STR GPU_LST_STR] PATH_PROC PARAM_PREP_ID PARAM_TRAIN_ID PARAM_FILE

positional arguments:
  PATH_PROC             path to processing folder (2_segment folder)
  PARAM_PREP_ID         id for merge and scaling parameters e.g. v079 (PARAM_inp_CNN_feature_prep_v01.txt)
  PARAM_TRAIN_ID        id for training parameters e.g. t16onl (PARAM_inp_CNN_train_v01.txt)
  PARAM_FILE            name of framework parameter file

options:
  -h, --help            show this help message and exit
  --GPU_LST_STR GPU_LST_STR
                        list of GPUs (separated by ":")
  --PATH_INP_BASE PATH_INP_BASE
                        path to site pre processing folder as default (None)
                        the path to 1_site_preproc is used

'''

import os
import sys
import argparse

# ----------------- PATHS & GLOBAL PARAM -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


import cnn_workflow.MAIN_evaluate_model as MAIN_evaluate_model


def main(pca_calc_utils, PARAM_inp):

    # PATH FOR PARAM FILE
    PARAM = PARAM_inp.copy()
    PARAM['PATH_INP'] = os.path.normpath(
        os.path.join(PARAM['PATH_PROC'], '01_input'))

    # pre read param file
    # read parameters
    sys.path.append(PARAM['PATH_INP'])
    exec('import ' + PARAM['PARAM_FILE'])
    eval(PARAM['PARAM_FILE']).get_param(PARAM)

    for e_cv in range(len(PARAM['PHASE_META_FILE'])):
        MAIN_evaluate_model.main(pca_calc_utils, PARAM_inp, e_cv)

    return



if __name__ == "__main__":

    # general inputs required for MAIN_train_incl_aug
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('path to processing folder (2_segment folder)'))
    parser.add_argument(
        'PARAM_PREP_ID', type=str,
        help=('id for merge and scaling parameters e.g. v079 ' +
              '(PARAM_inp_CNN_feature_prep_v01.txt)'))
    parser.add_argument(
        'PARAM_TRAIN_ID', type=str,
        help=('id for training parameters e.g. t16onl ' +
              '(PARAM_inp_CNN_train_v01.txt)'))
    parser.add_argument(
        'PARAM_FILE', type=str,
        help='name of framework parameter file')
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



