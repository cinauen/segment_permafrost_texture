'''
Parameters for creating input data
- tiling (create prediction and trainng tiles for CNN)
- extracting untiled input for ML or for test patches
'''

import param_settings.param_utils as param_utils

def add_proc_param(PARAM):
    '''
    Parameters are defined per label_area
    For the training areas this can be A01 or A02 with additinal sub areas
    For the test patches these are juat the separate test-01 etc patches
    '''
    # update to incorporate cmd input
    if ('WINDOW_TYPE' not in PARAM.keys()
        or PARAM['WINDOW_TYPE'] != 'untiled'):
        param_utils.update_window_param(PARAM)

    window_type =  PARAM['WINDOW_TYPE']
    if 'labelling_area' in PARAM.keys():
        label_area = PARAM['labelling_area']
        label_version = f'_{PARAM["LABEL_VERSION"][label_area.split("-")[0]]}'
    elif 'prediction_area' in PARAM.keys():
        label_area = PARAM['prediction_area']
        label_version = ''

    scale_type = PARAM['SCALE_TYPE']
    set_version = PARAM['SPLIT_SET_VERSION']

    # SUB_AREAS are used to define separate areas though which can loop
    # during cross validation: e.g. with ['train-01', 'train-02', 'train-03']
    # could do cross validation using always one area for validation
    # and two areas for training
    if label_area not in PARAM['SUB_AREAS_dict'].keys():
        # if do not have any sub areas (e.g  for creating test patches
        # or for prediction tiles)
        PARAM['SUB_AREAS'] = ['']
    else:
        PARAM['SUB_AREAS'] = PARAM['SUB_AREAS_dict'][label_area]

    # !!! for subfolder split label_area name label_area.split("-")[0]
    # this is that the distributed test patches end up in the same folder
    PARAM['SUBFOLDER_PREFIX'] = (
        f'{PARAM["PROJ_PREFIX"]}_{label_area.split("-")[0]}_{scale_type}{label_version}_{window_type}_{set_version}')
    # e.g. BLyaE_SPOT2018_A01_perc_8bit_Lv02_w298-298_v00_raw_rot_keep_resol

    # sub areas, if any, are added to the file suffix
    PARAM['FILE_PREFIX'] = [
        f'{PARAM["PROJ_PREFIX"]}_{label_area}{x}'
        for x in PARAM['SUB_AREAS']]

    PARAM['out_type_data'] = 'uint8'  # what format output data should have
    # set here to int since greyscale imagery is uint8
    # GLCMs are not included in this processing step. However, if want
    # to include GLCMs anyway could set out_type_data to None (will
    # keep float and nan)

    # could be used to add additional bands. However this is deprecated.
    # as GLCM bands are added later in data_loader
    PARAM['ADD_CHANNEL_IMG'] = []

    # deprecated leave this empty here. Is done later in dataloader.
    PARAM['ADD_CHANNEL_BAND'] = [[]]

    # if trim missing values or fill with nan if widow size
    # doesn't fit when tiling with coarsen()
    PARAM['TRIM_SUB_IMG'] = False


    #AOI_site_prefix = PARAM['AOI_extract_area_prefix'] + '_' + label_area
    #if ('AOI_extract_area_dict' in PARAM.keys()
    #    and label_area in PARAM['AOI_extract_area_dict'].keys()):
    PARAM['AOI_extract_area'] = PARAM['AOI_area_dict'][label_area]
    #else:
    #    PARAM['AOI_extract_area'] = AOI_site_prefix + '.geojson'

    # e.g. 'BLyakhE_labelling_AOI_32654_A01.geojson'
    #if 'AOI_extract_sub_area_dict' in PARAM.keys():

    # select required subarea
    # if main area is not split into subareas then just use main area
    # this is e.g. the case for the test patches.
    PARAM['AOI_extract_sub_area'] = [
        PARAM['AOI_sub_area_dict'][f'{label_area}{x}']
        if x != '' else PARAM['AOI_area_dict'][label_area]
        for x in PARAM['SUB_AREAS']]
    #else:
    #    PARAM['AOI_extract_sub_area'] = [
    #        f'{AOI_site_prefix}_{x}.geojson'
    #        if x != '' else f'{AOI_site_prefix}.geojson'
    #        for x in PARAM['SUB_AREAS']]

    # re-labelling is done later. Thus can leave empty here
    PARAM['TRAIN_RELABEL'] = {}
    #e.g. {1: 2, 3: 2}  # from to class

    PARAM['PARALLEL'] = True  # parallel is faster


    return