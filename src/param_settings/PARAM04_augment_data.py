'''
Parameters for augmenting and/or calcuate GLCM texture for offline
augmentation or for prediction data (for CNN)

'''

import param_settings.param_utils as param_utils
import cnn_workflow.cnn_workflow.custom_augmentation as custom_augmentation

def add_proc_param(PARAM):
    # update to incorporate cmd input
    param_utils.update_window_param(PARAM)

    window_type =  PARAM['WINDOW_TYPE']
    if PARAM['PHASE'] == 'predict':
        tiling_area = PARAM['tiling_area']
        label_version = ''  # empty since there are no labels for prediction
    else:
        tiling_area = PARAM['tiling_area']
        label_version = f'{PARAM["LABEL_VERSION"][tiling_area.split("-")[0]]}'

    scale_type = PARAM['SCALE_TYPE']
    set_version = PARAM['SPLIT_SET_VERSION']

    # here SUB_AREAS are used to define separate areas through which can
    # loop during cross-validation: e.g. with ['train-01', 'train-02', 'train-03']
    # could do cross-validation using always one area for validation
    # and two areas for training
    if tiling_area not in PARAM['SUB_AREAS_dict'].keys():
        # if do not have any sub areas (e.g  for creating test patches
        # or for prediction tiles)
        PARAM['SUB_AREAS'] = ['']
    else:
        PARAM['SUB_AREAS'] = PARAM['SUB_AREAS_dict'][tiling_area]

    PARAM['FILE_PREFIX'] = PARAM['PROJ_PREFIX'] + '_' + tiling_area
    # e.g. 'BLyakhE_SPOT2018_A01' this will be used for file prefix

    #PARAM['GPU_no'] = [PARAM['GPU_lst'][0]]  # numbering starts from 0
    # several device IDs are used for torch.nn.DataParallel
    # when setting up model

    PARAM['NORM'] = False  # best keep as False here

    # define input forder where input tiles and output (augmented tiles or
    # GLCM tiles) is saved
    rot_index = f'rot_{PARAM["rot_inperp"]}' if PARAM['ROTATE_DEGREE'][tiling_area] != 0 else ''
    name_prefix = [PARAM['PROJ_PREFIX'], tiling_area, scale_type,
                   label_version, window_type, set_version, rot_index]
    PARAM['subfolder'] = '_'.join([x for x in name_prefix if x != ''])

    PARAM['X_BANDS'] = None  # keep (with None all bands are taken)

    # ------- setup augmentations for training data (thus only in
    # connection with labelling area)
    if PARAM['PHASE'] != 'predict':
        bit_depth = int(scale_type.split('_')[-1].split('bit')[0])
        param_utils.get_augment_param(PARAM)
        PARAM['BIT_DEPTH_aug_inp'] = 2**bit_depth - 2
        # need to convert int from geyscale to float for color augmentation
        PARAM['aug_range'] = [
            custom_augmentation.convert_to_float(
                PARAM['BIT_DEPTH_aug_inp']),
            custom_augmentation.revert_to_int(
                PARAM['BIT_DEPTH_aug_inp'])]
        # other parameters for offline augmentation are set in the
        # project param file such as PARAM['n_augment'], PARAM['aug_geom'],
        # PARAM['aug_col'], PARAM['band_lst_col_aug']


    # With load_phase_separate = True list of separate SUB_AOIs are taken into
    # account if provided (files are read according to metadata file)
    # If False then files are searched with glob
    # Here both options give same result if all SUB_AREAS are reads in
    PARAM['load_phase_separate'] = True  # can be kept as true if
    # metadata files were generated

    PARAM['texture_param_file'] = 'PARAM02_calc_texture'


    return PARAM