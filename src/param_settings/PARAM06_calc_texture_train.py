'''
Input params for texture calculation of single tiles during training
used in:
 - texture_calc/MAIN_calc_texture_single_file.py
 - and dufing recalculating GLCMs with on-the-fly augmentation in
    custom_data_loader.CustomDataset_augment_ScaleMerge()

'''

import numpy as np

def add_proc_param(PARAM):
    '''
    get BIT_DEPTH from project specific train param file
    PARAM['BIT_DEPTH'] = int(scale_type.split('_')[-1].split('bit')[0])

    '''
    scale_type = PARAM['SCALE_TYPE']

    PARAM['PADDING_CONST'] = np.nan  # has to be np.nan do not change

    PARAM['BIT_DEPTH'] = int(scale_type.split('_')[-1].split('bit')[0])
    PARAM['BIN_FROM'] = 2**PARAM['BIT_DEPTH']
    PARAM['BIN_TO'] = 2**4
    PARAM['x_band_GLCM'] = ['1']
    # (for texture calc inpuit will always be converted to 4 bit: PARAM['BIN_TO'])
    PARAM['TEX_SUFFIX'] = ''

    PARAM['PARALLEL'] = True  # only used for cross calc here parallel is faster
    PARAM['DEBUG'] = False

    # !!! COMMMNENT CROSS STATS calc for whole area:
    # !!! texture for single direction files is needed to be contained in PARAM['TEX_PARAM']
    # !!! since they are saved on file and then read in again (due to large file size)

    # params are sublists with
    #   [list with direction sin degree, window_width,
    #    window_height]
    # for each sublist a texture set will be calcualted
    # (window width and height should be same and should be odd numbers)
    PARAM['cross_calc'] = ['std']

    # if calculate GLCM for individual chip then do not save each individual texture direction
    # thus calculate this only in PARAM['img_cross_calc']
    PARAM['TEX_PARAM_inp'] = {
        'r01': [[0, 45, 90, 135], 3, 3],
        'r02': [[0, 45, 90, 135], 5, 5],
        'r05': [[0, 45, 90, 135], 11, 11],
        'r10': [[0, 45, 90, 135], 21, 21],
        'r20': [[0, 45, 90, 135], 41, 41]
        }

    PARAM['cross_calc'] = ['std']
    PARAM['img_cross_calc_inp'] = {
        # 'r01': [[[x], 3, 3] for x in [0, 45, 90, 135]],
        # 'r02': [[[x], 5, 5] for x in [0, 45, 90, 135]],
        'r05': [[[x], 11, 11] for x in [0, 45, 90, 135]],
        # 'r10': [[[x], 21, 21] for x in [0, 45, 90, 135]],
        # 'r20': [[[x], 41, 41] for x in [0, 45, 90, 135]]
        }


    return PARAM