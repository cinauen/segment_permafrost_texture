'''
This module contains general utils to handle the parameters used for
the segmentation workflow
'''

import sys
import os
import importlib
import pandas as pd

import utils.file_utils as file_utils


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(file_path, f"{module_name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize_all_params(funct_inp, PROC_STEP):
    # --- add input parameters
    PARAM = {}
    for key, val in funct_inp.items():
        if val != 'None':
            PARAM[key] = val

    # --- add site/project specific parameters
    # cmd line input parameters have priority over site/project
    # specific parameters
    PARAM = path_param_setup_proj(PARAM, PROC_STEP)
    # --- add processing step dependent, fixed parameters
    add_proc_param(PARAM)
    return PARAM


def initialize_segmentation_param(funct_inp):
    '''
    prepare input parameters for segmentation
    '''
    # add cmd line input parameters
    PARAM = {}
    for key, val in funct_inp.items():
        if val != 'None':
            PARAM[key] = val

    # path for framework specific parameter file
    PARAM['PATH_INP'] = os.path.normpath(
        os.path.join(PARAM['PATH_PROC'], '01_input'))
    # add path
    sys.path.append(PARAM['PATH_INP'])
    # import processing step specific parameters
    param_module = importlib.import_module(
        PARAM['PARAM_FILE'])
    param_module.get_param(PARAM)

    # path to site specific folders
    if 'PATH_INP_BASE' not in PARAM.keys():
        PARAM['PATH_INP_BASE'] = os.path.normpath(
            os.path.join(PARAM['PATH_PROC'], '..', '1_site_preproc'))

    # for training outputs
    if 'model_folder_base' in PARAM.keys():
        PARAM['PATH_TRAIN'] = os.path.normpath(
            os.path.join(PARAM['PATH_PROC'], '02_train',
                        PARAM['model_folder_base']))
    return PARAM


def path_param_setup_proj(PARAM_inp, proc_step):
    '''
    Function to setup the project/site specific paramters, which are
    used for creating the project/site specific training inputs,
    which is all done within the "1_site_preproc" folder.

    Parameter hierarchy is:
    - processing-step specific parameters (are added in other function
      add_proc_param())
    - PARAM_inp: parameters from cmd line input (but inputs which are
      "None" are ignored)
    - site/project specific parameters (are overwritten by
      PARAM_inp from cmd line input)

    Steps:
    1) Input PARAM (from function input) are first used to get all paths
       e.g. to parameter files
    2) site/project specfic parameters are taken from
       PARAM_inp['PROJ_PARAM_FILE']
    3) site/project specfic parameters are updated with parameters from
       cmd line input
    * processing-step specific parameters (static parameters) are
       afterwards added in function add_proc_param()
    '''
    # --- 1) get all paths
    PARAM_inp['PATH_BASE'] = os.path.abspath(
        os.path.join(PARAM_inp['PATH_PROC'], '../..'))
    PARAM_inp['PATH_INP'] = os.path.normpath(
        os.path.join(PARAM_inp['PATH_PROC'], '01_input'))
    PARAM_inp['PATH_EXPORT'] = os.path.normpath(
        os.path.join(PARAM_inp['PATH_PROC'], '02_pre_proc'))
    PARAM_inp['PATH_LABELS'] = os.path.normpath(
        os.path.join(PARAM_inp['PATH_PROC'], '00_labelling'))
    PARAM_inp['PATH_TRAIN'] = os.path.normpath(
        os.path.join(PARAM_inp['PATH_PROC'], '03_train_inp'))

    # retrief GPU device numbers to use
    if 'GPU_LST_STR' in PARAM_inp.keys():
        PARAM_inp['GPU_lst'] = [
            int(x) for x in PARAM_inp['GPU_LST_STR'].split(':')]

    # check that input path exists
    if not os.path.isdir(PARAM_inp['PATH_INP']):
        raise FileNotFoundError(
            f'Non-existing path: {PARAM_inp["PATH_INP"]}')

    # --- 2) get site/project specific parameters
    # add parameter file path
    sys.path.append(PARAM_inp['PATH_INP'])
    # import and get site/project specific parameters
    proj_param_module = importlib.import_module(
        PARAM_inp['PROJ_PARAM_FILE'])
    PARAM = proj_param_module.get_proj_param(
        proc_step, PARAM_inp['PATH_BASE'], PARAM_inp['SCALE_TYPE'])
    # if cmd line input is not 'None' then prefer input from cmd line
    PARAM.update({x:y for x, y in PARAM_inp.items() if y != 'None'})
    return PARAM


def add_proc_param(PARAM):
    '''
    get processing-step specific parameters
    '''
    proc_param_module = importlib.import_module(
        f'.{PARAM["PARAM_FILE"]}', 'param_settings')
    proc_param_module.add_proc_param(PARAM)
    return


def update_window_param(PARAM):
    '''
    Create window parameters for tiling in
    projcessing steps 3 (create trainnig tiles), 4, 5
    '''
    if 'window_size' not in PARAM.keys() or PARAM['window_size'] is None:
        # !!! this is for the case where use extract data for supervised ML
        # or for test patches and no tiling is required
        return

    # Additional padding is added to avoid edge effects for calculartig GLCMs
    # The padding is later clipped in the custom data loader
    PARAM['WINDOW_PADDING_ADD'] = (20+1)*2  # should be max radius + step x2 for both sides
    PARAM['WINDOW_SPLIT'] = [
        PARAM['window_size']*1 + PARAM['WINDOW_PADDING_ADD'],
        PARAM['window_size']*1 + PARAM['WINDOW_PADDING_ADD']]
    PARAM['WINDOW_SHIFT_X'] = int(
        PARAM['WINDOW_SPLIT'][0]*PARAM['window_shift_factor'])
    PARAM['WINDOW_SHIFT_Y'] = int(
        PARAM['WINDOW_SPLIT'][1]*PARAM['window_shift_factor'])

    PARAM['WINDOW_TYPE'] =  'w' + '-'.join([str(x) for x in PARAM['WINDOW_SPLIT']])
    return


def read_param_from_file(path, file_name, param_extract, col_list=None,
                         convert_lst=None, convert_val_lst=None,
                         convert_eval_lst=None):
    """
    Reads parameters from a text file.

    Parameters:
    ----------
    - path: str
        The directory containing the parameter file.
    - file_name: str
        The name of the parameter file.
        (e.g. data preparation parameters:
            PARAM_inp_CNNmerge_v01.txt/PARAM_inp_MLmerge_v01.txt
         e.g. training parameters:
            PARAM_inp_CNNtrain_v01.txt/PARAM_inp_MLtrain_v01.txt)
    - param_extract: str
        The key in the to extract the parameter.
        (e.g. v079 according to PARAM_PREP_ID or
         t16onl according to PARAM_TRAIN_ID or)
    - col_list: list, optional
        List of columns to split and clean.
    - convert_lst: list, optional
        List of columns to convert list items to int or float.
    - convert_val_lst: list, optional
        List of columns to convert values to specific types.
    - convert_eval_lst: list, optional
        List of columns to evaluate and convert.

    Returns:
    ----------
    - dict: A dictionary containing the processed parameters.

    """
    # read parameter file
    path_inp = os.path.join(path, file_name)
    param_df = pd.read_csv(
        path_inp, delimiter='\t', header=0, index_col=0)
    param = param_df.loc[param_extract, :].to_dict()

    if col_list is not None:
        # split parameter into list (according to ":" separation)
        param_new = {
            x: str(param[x]).split(':')
            if (str(param[x]) != 'None' and str(param[x]) != 'nan')
            else None for x in col_list}
        # remove empy items
        param_new = {x: [yy for yy in y if yy != '']
                     if y is not None else None
                     for x, y in param_new.items()}
        param.update(param_new)

    if convert_lst is not None:
        # convert specified parameters to int or float
        param_new = {
            x:[int(xx) if type(eval(xx))==int else float(xx) for xx in param[x]]
            if param[x] is not None else None for x in convert_lst}
        param.update(param_new)

    if convert_val_lst is not None:
        # check type and convert
        def check_val(inp):
            if type(eval(inp))==int:
                out = int(inp)
            elif type(eval(inp))==float:
                out = float(inp)
            elif type(eval(inp))==bool:
                out = bool(inp)
            elif inp == 'None':
                out = None
            else:
                out = inp
            return out
        param_new = {x: check_val(str(param[x]))
                     if (str(param[x]) != 'None' and str(param[x]) != 'nan')
                     else None for x in convert_val_lst}
        param.update(param_new)

    if convert_eval_lst is not None:
        param_new = {x: eval(str(param[x]))
                     if (str(param[x]) != 'None' and str(param[x]) != 'nan')
                     else None for x in convert_eval_lst}
        param.update(param_new)
    return param


def update_merge_param_from_file(PARAM, path_inp=None):
    """
    Get merge parameters
    Parameters are converted using read_param_from_file()
    """
    lst_inp = ['file_suffix_lst', 'merge_bands',
               'feature_add']
    if path_inp is None:
        path_inp = os.path.dirname(__file__)
    param_read = read_param_from_file(
        path_inp, PARAM['file_merge_param'],
        PARAM['PARAM_PREP_ID'],
        col_list=lst_inp)
    PARAM.update(param_read)
    return


def update_train_param_from_file(PARAM, path_inp=None):
    """
    For train parameters: Specifies which parameter values to convert
        with read_param_from_file()
    """
    lst_inp = ['loss_weights',
               'aug_geom_probab', 'aug_col_probab',
               'aug_col_param',
               'band_lst_col_aug', 'lr_scheduler',
               'lr_gamma', 'lr_milestones']
    list_conv = ['loss_weights',
                 'aug_geom_probab', 'aug_col_probab',
                 'aug_col_param',
                 'band_lst_col_aug',
                 'lr_gamma', 'lr_milestones']
    list_conv_val = []
    if path_inp is None:
        path_inp = os.path.dirname(__file__)
    param_read = read_param_from_file(
        path_inp, PARAM['file_train_param'],
        PARAM['PARAM_TRAIN_ID'],
        col_list=lst_inp, convert_lst=list_conv,
        convert_val_lst=list_conv_val)
    PARAM.update(param_read)
    return


def update_MLmerge_param_from_file(PARAM, path_inp=None):

    if path_inp is None:
        path_inp = os.path.dirname(__file__)

    lst_inp = ['file_suffix_lst', 'merge_bands',
               'feature_add', 'preproc_scikit']
    list_conv = []
    list_conv_val = []
    list_eval = ['log_dict_scikit']
    param_read = read_param_from_file(
        path_inp, PARAM['file_merge_param'],
        PARAM['PARAM_PREP_ID'],
        col_list=lst_inp, convert_lst=list_conv,
        convert_val_lst=list_conv_val,
        convert_eval_lst=list_eval)
    PARAM.update(param_read)
    return


def update_MLtrain_param_from_file(PARAM, path_inp=None):

    if path_inp is None:
        path_inp = os.path.dirname(__file__)

    lst_inp = []
    list_conv = []
    list_conv_val = ['n_estimators', 'max_depth', 'max_samples',
                     'max_features', 'bootstrap']
    list_eval = []
    param_read = read_param_from_file(
        path_inp, PARAM['file_train_param'],
        PARAM['PARAM_TRAIN_ID'],
        col_list=lst_inp, convert_lst=list_conv,
        convert_val_lst=list_conv_val,
        convert_eval_lst=list_eval)
    PARAM.update(param_read)
    return


def get_augment_param(PARAM):
    ''''''
    import cnn_workflow.cnn_workflow.custom_augmentation as custom_augmentation
    if PARAM['aug_geom_probab'] is not None:
        PARAM['aug_geom'] = custom_augmentation.get_train_augment_geom01(
            p1=PARAM['aug_geom_probab'][0],
            p2=PARAM['aug_geom_probab'][1])
    else:
        PARAM['aug_geom'] = None  # custom_augmentation.get_train_augment_geom01()

    if PARAM['aug_col_probab'] is not None and len(PARAM['aug_col_probab']) == 2:
        PARAM['aug_col'] = custom_augmentation.get_train_augment_col_grey(
            p1=PARAM['aug_col_probab'][0], p2=PARAM['aug_col_probab'][1],
            b_limit=PARAM['aug_col_param'][0],
            c_limit=PARAM['aug_col_param'][1],
            gamma_min=PARAM['aug_col_param'][2],
            gamma_max=PARAM['aug_col_param'][3])

    elif PARAM['aug_col_probab'] is not None and len(PARAM['aug_col_probab']) > 2:
        PARAM['aug_col'] = custom_augmentation.get_train_augment_col_grey_advanced(
            p1=PARAM['aug_col_probab'][0],
            p2=PARAM['aug_col_probab'][1],
            p3=PARAM['aug_col_probab'][2],
            b_limit=PARAM['aug_col_param'][0],
            c_limit=PARAM['aug_col_param'][1],
            gamma_min=PARAM['aug_col_param'][2],
            gamma_max=PARAM['aug_col_param'][3])
    else:
        PARAM['aug_col'] = None
    return


def setup_model_input_properties(PARAM, CV_NUM, n_classes):
    '''
    Function to create folder and prefix of pre-trained model
    in accordance with the cross-validation and the used parameter setups

    Automatic folder and prefix generation can be avoided by directly
    define them in the project parameter file (i.e.
    PARAM['model_input']['PATH_PROC_inp'],
    PARAM['model_input']['file_prefix'])

    The log file suffix is adapted to fine-tuining
    '''
    if PARAM['model_input']['PATH_PROC_inp'] is None:
        # ------- path to pre-trained model to be fine-tuned -------
        sub_folder_model_inp = (
            f"{PARAM['model_input']['PARAM_PREP_ID']}{PARAM['model_input']['PARAM_TRAIN_ID']}_cv{CV_NUM:02d}")

        PARAM['model_input']['PATH_PROC_inp'] = os.path.normpath(
            os.path.join(PARAM['PATH_INP'], '..', '02_train',
                        PARAM['model_input']['model_folder_base'],
                        sub_folder_model_inp))

    # -------- model file input prefixes
    if PARAM['model_input']['file_prefix'] is None:
        file_inp_lst = [
            sub_folder_model_inp, PARAM['file_prefix_add'],
            f'ncla{n_classes}']
        PARAM['model_input']['file_prefix'] = file_utils.create_filename(
            file_inp_lst)

    # adapt log suffix to fine tune
    PARAM['LOG_FILE_SUFFIX'] = 'fine_tune'
    return
