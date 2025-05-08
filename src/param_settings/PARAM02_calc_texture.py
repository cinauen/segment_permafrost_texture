'''
Input params for texture calculation


'''

import numpy as np

def add_proc_param(PARAM):
    scale_type = PARAM['SCALE_TYPE']
    PARAM['FILE_INP'] = f'{PARAM["PROJ_PREFIX"]}_scale_{scale_type}.tif'
    # e.g. 'BLyakhE_HEX1979_scale_std_8bit.tif'

    PARAM['PADDING_CONST'] = np.nan  # has to be np.nan do not change
    PARAM['SAVE_PREPROC_IMG'] = True  # if want to save input image after
    # preprocessing e.g. if needed to clip image to smaller area

    PARAM['BIT_DEPTH'] = int(scale_type.split('_')[-1].split('bit')[0])
    PARAM['BIN_FROM'] = 2**PARAM['BIT_DEPTH']
    PARAM['BIN_TO'] = 2**4
    # (for texture calc input will always be converted to 4 bit: PARAM['BIN_TO'])
    PARAM['TEX_SUFFIX'] = ''

    PARAM['PARALLEL'] = True  # only used for cross calc here parallel is faster

    # -------------- TEXTURE cacluation parameters ------------
    # PARAM['TEX_PARAM'] are sublists with
    #   [list with directions in degree, window_width, window_height]
    # for each sublist a texture set will be calcualted
    # (window width and height should be same and should be odd numbers)

    # calculate standard deviation (std) of different texture directions
    PARAM['cross_calc'] = ['std']
    if PARAM['PROC_STEP'] == 2:
        # ------- params to calculate texture for full area ---
        # !!! COMMMNENT CROSS STATS calc for whole area:
        # !!! texture of single direction files is need to be contained in PARAM['TEX_PARAM']
        # !!! since they are saved to file and then read in again (due to large file size)
        PARAM['FILE_PREFIX'] = f"{PARAM['PROJ_PREFIX']}_{PARAM['AOI_TEX_SUFFIX']}_{scale_type}"  # output prefix

        # for proc_step 2 when calcualte texture for full area,
        # texture sets are saved to individual files such that not need
        # to keep all in memeory
        PARAM['TEX_PARAM'] = [
            [[0, 45, 90, 135], 3, 3],
            [[0, 45, 90, 135], 5, 5],
            [[0, 45, 90, 135], 11, 11],
            [[0, 45, 90, 135], 21, 21],
            [[0, 45, 90, 135], 41, 41],
            [[0], 11, 11],
            [[45], 11, 11],
            [[90], 11, 11],
            [[135], 11, 11],
            ]

        # from which files to calculate cross statistics
        # r05 corresponds to radius 5 (window size 11)
        PARAM['img_cross_calc'] = {
            'r05': ['a0_r05_norm_C01', 'a1_r05_norm_C01',
                    'a2_r05_norm_C01', 'a3_r05_norm_C01'],
            }
    else:
        # ---- prarms to calculate texture for individual tiles (e.g. for offline
        # augmentation or for prediction tiles)
        PARAM['FILE_PREFIX'] = PARAM['PROJ_PREFIX'] + '_' + PARAM['tiling_area'] # output prefix
        # if calculate GLCM for individual tile, then do not save each
        # individual texture direction. Thus the individual directions
        # required for cross calc are set in PARAM['img_cross_calc']
        PARAM['TEX_PARAM'] = [
            [[0, 45, 90, 135], 3, 3],
            [[0, 45, 90, 135], 5, 5],
            [[0, 45, 90, 135], 11, 11],
            [[0, 45, 90, 135], 21, 21],
            [[0, 45, 90, 135], 41, 41],
            ]

        # r05 corresponds to radius 5 (window size 11)
        PARAM['img_cross_calc'] = {
            'r05': [[[x], 11, 11] for x in [0, 45, 90, 135]],
            }
    return PARAM