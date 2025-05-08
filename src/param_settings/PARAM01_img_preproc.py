"""
Input params image preprocessing (proc step 01)

"""
import numpy as np

def add_proc_param(PARAM):

    # output file prefix (e.g. 'BLyakhE_HEX1979')
    PARAM['FILE_PREFIX'] = PARAM['PROJ_PREFIX']

    # input files
    PARAM['FILE_INP'] = PARAM['GREYSCALE_FILE_INP']
    PARAM['AOI_clip'] = PARAM['AOI_full_area']

    # target coord syst
    PARAM['EPSG_TARGET'] = PARAM['EPSG_TARGET']

    # suffix for param error log file
    PARAM['LOG_FILE_SUFFIX'] = 'img_pre_proc'

    # set of scaling types to test to use
    # with nodata_out = None then nodata value of input is used
    PARAM['SCALING_DICT'] = {
        'scale_8bit': [
            dict(img_np_key='scale_perc0-2_g0-3_8bit',
                nodata_out=None, how='scale',
                to_bit_num=2**8, perc=[0.2, 99.8], gamma=0.3),
            dict(img_np_key='scale_perc0-2_g0-4_8bit',
                nodata_out=None, how='scale',
                to_bit_num=2**8, perc=[0.2, 99.8], gamma=0.4),
            dict(img_np_key='scale_std3_8bit',
                nodata_out=None, how='scale',
                to_bit_num=2**8, std_fact=3),
            dict(img_np_key='scale_std4_8bit',
                nodata_out=None, how='scale',
                to_bit_num=2**8, std_fact=4),
            dict(img_np_key='scale_std5_8bit',
                nodata_out=None, how='scale',
                to_bit_num=2**8, std_fact=5),
            dict(img_np_key='scale_min_max_8bit',
                nodata_out=None, how='scale',
                to_bit_num=2**8, perc=[0, 100]),
        ],
        #'scale_16bit': [
        #    dict(img_np_key='scale_std4_16bit',
        #        nodata_out=None, how='scale',
        #        to_bit_num=2**16, std_fact=4),
        #    dict(img_np_key='scale_min_max_16bit',
        #        nodata_out=None, how='scale',
        #        to_bit_num=2**16, perc=[0, 100]),
        #]
        }

    # set of parameters for histogram matching
    # !!! use placeholder {{}} to add reference file id afterwards
    PARAM['HIST_MATCH_DICT'] = {
        'hist-match_8bit': [
            dict(img_key=f'scale_histm_scik{{}}_8bit',
                how='scikit', to_bit_num=2**8,
                img_key_inp='scale_min_max_8bit'),
            dict(img_key=f'scale_histm_inp_raw_scik{{}}_8bit',
                how='scikit', to_bit_num=2**8,
                img_key_inp='raw'),
            dict(img_key=f'scale_histm_inp_std4_scik{{}}_8bit',
                how='scikit', to_bit_num=2**8,
                img_key_inp='scale_std4_8bit'),
            #dict(img_key='scale_histm_blend0-1_8bit',
            #     how='albu', to_bit_num=2**8, hist_blend=0.1,
            #     img_key_inp='scale_min_max_8bit'),

            dict(img_key=f'scale_histm_b0-0a{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.0,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to=255, discard_nodata=False),
            dict(img_key=f'scale_histm_b0-2a{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.2,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to=255, discard_nodata=False),
            dict(img_key=f'scale_histm_b0-3a{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.3,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to=255, discard_nodata=False),
            dict(img_key=f'scale_histm_b0-5a{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.5,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to=255, discard_nodata=False),
            dict(img_key=f'scale_histm_b0-8a{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.8,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to=255, discard_nodata=False),

            dict(img_key=f'scale_histm_b0-0b{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.0,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to='mean', discard_nodata=False),
            dict(img_key=f'scale_histm_b0-2b{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.2,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to='mean', discard_nodata=False),
            dict(img_key=f'scale_histm_b0-3b{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.3,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to='mean', discard_nodata=False),
            dict(img_key=f'scale_histm_b0-5b{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.5,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to='mean', discard_nodata=False),
            dict(img_key=f'scale_histm_b0-8b{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.8,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to='mean', discard_nodata=False),
            #dict(img_key='scale_histm_b1-0_8bit',
            #     how='albu', to_bit_num=2**8, hist_blend=1.0,
            #     img_key_inp='scale_min_max_8bit'),
            # raw should be same as min_max
            #dict(img_key='scale_histm_inp_raw_blend0-5_8bit',
            #     how='albu', to_bit_num=2**8, hist_blend=0.5,
            #     img_key_inp='raw'),
            #dict(img_key='scale_histm_inp_raw_blend0-8_8bit',
            #     how='albu', to_bit_num=2**8, hist_blend=0.8,
            #     img_key_inp='raw'),
            #dict(img_key='scale_histm_inp_raw_blend0-3_8bit',
            #     how='albu', to_bit_num=2**8, hist_blend=0.3,
            #     img_key_inp='raw'),
            dict(img_key=f'scale_histm_b0-5c{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.5,
                 img_key_inp='scale_min_max_8bit',
                 set_mask_to=np.nan, discard_nodata=False),
            dict(img_key=f'scale_histm_inp_std5_b0-5c{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.5,
                 img_key_inp='scale_std5_8bit',
                 set_mask_to=np.nan, discard_nodata=False),
            dict(img_key=f'scale_histm_inp_std4_b0-5c{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.5,
                 img_key_inp='scale_std4_8bit',
                 set_mask_to=np.nan, discard_nodata=False),
            dict(img_key=f'scale_histm_inp_g03_b0-5c{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.5,
                 img_key_inp='scale_perc0-2_g0-3_8bit',
                 set_mask_to=np.nan, discard_nodata=False),
            dict(img_key=f'scale_histm_inp_std3_b0-5c{{}}_8bit',
                 how='albu', to_bit_num=2**8, hist_blend=0.5,
                 img_key_inp='scale_std3_8bit',
                 set_mask_to=np.nan, discard_nodata=False),
        ],
        }

    return PARAM