"""
Parameters for site/imagery specific processing:
PROC_STEP:
  1: - IMAGE PREPROCESSING: adjust intensity range and quantization
  2: - CALCULATE TEXTURE for whole imagery and statistics for full
       area and sub areas
  3: - SPLIT IMAGERY into tiles for CNN training and prediction
     - EXTRACT input for supervised classification and for test patches
       (no tiling required, extracts label and data full-area image
       using the specified AOIs)
  4: - CALCULATE TEXTURE on augmented training and non augmented
       prediction tiles (required for offline augmentation and
       prediction when using GLCM features)

!!! Note on data usage in this example:
    This example uses a limited amount of input data in order
    to reduce computation times and data size. Only four fine-tuning and four test patches are used.
    Furthermore, due to commercial restrictions, the SPOT input data cannot provided.
    Thus, this example uses the freely downloadable Hexagon (KH-9PC) data only.

"""
import os

def get_proj_param(PROC_STEP, PATH_BASE, SCALE_TYPE):
    '''
    Parameters
    ----------
    PROC_STEP: int
        processing step number, used to assign specific parameters
    PATH_BASE: string
        path to base folder in which processing folder (i.e. default folder
        1_site_preproc or other is saved). Can e.g. be used to set e.g.
        REF_IMG_PATH below
    SCALE_TYPE: str (provided as function input, used for proc steps 2, 3, 4)
       Image input intensity scaling type for further processing (texture calc and segmentation).
       (Note: for texture calculation input image will always be converted to 4 bit
        PARAM['BIN_TO'] see parameter file PARAM02...)
    '''
    PARAM = {}
    PARAM['PROC_STEP'] = PROC_STEP
    PARAM['SCALE_TYPE'] = SCALE_TYPE

    # ===== fine-tuning and test patch limit !!!!
    # to keep the example small we used here only a limited number of
    # fine-tuning and test patches
    patch_limit = 4

    # ========== COMPUTATION SETTINGS ==========
    PARAM['DEBUG'] = True

    # ======== PROJECT SEPCIFICATIONS ========
    # main site folder in ...\1_site_preproc\FOLDER_NAME\
    PARAM['FOLDER_NAME'] = 'FadN_v1'
    PARAM['SITE_NAME'] = 'FadN'   # site name
    # for output file prefixes
    PARAM['PROJ_PREFIX'] = PARAM['SITE_NAME'] + '_HEX1980'
    PARAM['EPSG_TARGET'] = 32654  # target coord syst

    # ========= AOI areas =========
    # AOI for clipping outer processing extent
    PARAM['AOI_full_area'] = 'FadN_full_area_AOI_buffered_32654_reduced.geojson'

    AOI_prefix = 'FadN_labelling_AOI_32654'
    PARAM['AOI_area_dict'] = {}
    # -- training areas --
    fine_tune_areas = [
        f'fine_tune-{x:02d}' for x in range(1, patch_limit + 1)]
    PARAM['AOI_area_dict'].update(
        {x: f'{AOI_prefix}_{x}.geojson' for x in fine_tune_areas})
    # -- test areas --
    test_areas = [f'test-{x:02d}' for x in range(1, patch_limit + 1)]
    PARAM['AOI_area_dict'].update(
        {x: f'{AOI_prefix}_{x}.geojson' for x in test_areas})
    # -- processing and prediction area
    PARAM['AOI_area_dict'].update(
        {'all': PARAM['AOI_full_area']})

    # --------- AOI sub areas ------
    # The sub areas to further split the training areas into
    # cross-validation areas
    # They can be provided as {'A02': ['train-01', 'train-02', 'train-03']}
    # The assignement to test and validation sets is done later.
    # Note: futher training areas can be added as additional dictionary items
    PARAM['SUB_AREAS_dict'] = {}
    # assigns the geojson file to the sub areas e.h.
    # {'A02train-01': 'FadN_labelling_AOI_32654_A02_train-01.geojson'}
    # here empty since no traing subareas
    PARAM['AOI_sub_area_dict'] = {
        f'{x}{yy}': f'{AOI_prefix}_{x}_{yy}.geojson'
        for x, y in PARAM['SUB_AREAS_dict'].items()
        for yy in y
        }

    # ==== PROCESSING STEP SPECIFIC PARAMETERS ====
    # ---- IMAGE INTENSITIES ADJUSTMENT ----
    if PROC_STEP in [1]:
        # Input raw data image for pre-processing. Afterwards continue with scaled image.
        PARAM['GREYSCALE_FILE_INP'] = 'FadN_HEX1980_D3C1216-200448F004_georef_v1_reduced.tif'
        # output resolution
        PARAM['RESOLUTION_TARGET'] = 1.5

        # Path to reference image for hist matching
        # (e.g. if want to hist match intenstites from one site to another)
        # Can be single string or dict to test different referenc image or AOIs
        # (key is used for output file naming)
        # use None if no hist matching is done
        PARAM['REF_IMG_PATH'] = {'Rv1': os.path.normpath(os.path.join(
            PATH_BASE,
            r'1_site_preproc/BLyaE_v1/02_pre_proc/BLyaE_HEX1979_scale_perc0-2_g0-3_8bit.tif')
            ).replace('\\', os.path.sep)}
        # Clip area for hist match reference file (needs to be full path)
        # use None if no clipping of ref file
        PARAM['AOI_REF_FILE_CLIP'] = {'Rv1': os.path.normpath(os.path.join(
            PATH_BASE,
            r'1_site_preproc/BLyaE_v1/01_input/BLyaE_labelling_AOI_32654_A02.geojson')
            ).replace('\\', os.path.sep)}


    # ----- TEXTURE CALCULATION -----
    if PROC_STEP in [2]:
        # Specify area over which to calculate texture
        # If want use the whole area then PARAM['AOI_TEX_SUFFIX'] should
        # be named 'all'
        PARAM['AOI_TEX_SUFFIX'] = 'all'
        PARAM['AOI_TEX_clip'] = PARAM['AOI_full_area']
        PARAM['PLOT_TEX_DISTIBUTION'] = True  # if want to save plot with
        # texture and distribution (only for proc step 2)

        # Specify areas for where to calculate statistics
        # these will be used to for caclulating stats
        # (stats from training arrea can later be used to standardize data)
        PARAM['AOI_stats_calc'] = PARAM['AOI_sub_area_dict']
        PARAM['STATS_FILE_SUFFIX'] = 'P02_tex_stats_file'


    # --- INPUT CREATION FOR SEGMENTATION ---
    # ---- LABEL DEFINITIONS ----
    if PROC_STEP in [3]:
        # -- specify label files per training or test patch --
        # The dictionary specifies the labelling data (.tif file) per AOI
        PARAM['LABEL_FILE_INP'] = {
            'test-01': 'FadN_HEX1980_test_patch1_labels_Lv01_class_certainty.tif',
            'test-02': 'FadN_HEX1980_test_patch2_labels_Lv01_class_certainty.tif',
            'test-03': 'FadN_HEX1980_test_patch3_labels_Lv01_class_certainty.tif',
            'test-04': 'FadN_HEX1980_test_patch4_labels_Lv01_class_certainty.tif',
            'fine_tune-01': 'FadN_HEX1980_fine_tune_patch1_labels_Lv01_class_certainty.tif',
            'fine_tune-02': 'FadN_HEX1980_fine_tune_patch2_labels_Lv01_class_certainty.tif',
            'fine_tune-03': 'FadN_HEX1980_fine_tune_patch3_labels_Lv01_class_certainty.tif',
            'fine_tune-04': 'FadN_HEX1980_fine_tune_patch4_labels_Lv01_class_certainty.tif',
            }
        # It is also possible to use the combined label files since the AOIs
        # from above will be used to clip the labels to the required area
        #PARAM['LABEL_FILE_INP'].update(
        #    {f'test-{x:02d}': 'FadN_HEX1980_test_labels_Lv01_class_certainty.tif'
        #    for x in range(1, patch_limit + 1)})  # use one combined file which includes all test patches
        #PARAM['LABEL_FILE_INP'].update(
        #    {f'fine_tune-{x:02d}': 'FadN_HEX1980_fine_tune_labels_Lv01_class_certainty'
        #    for x in range(1, patch_limit + 1)})  # use one combined file which includes all test patches


        # --- CLASS SPECIFICATIONS ----
        # The class numbers and names here corresponds to the numbering
        # as in the label files (.tif files above).
        # Label renumbering and renaming can be done during training (if
        # required to adjust numer order or merge two classes)
        PARAM['LABEL_NAMING'] = [
            [0, 1, 2, 3, 4, 5, 6],
            ['nan', 'baydzherakhs', 'ridges_ponds', 'undisturbed',
             'gully_base', 'ponds', 'snow']]

        # --- DATA FILE SPECIFICATIONS ---
        # DATA_IMG corresponds to the image used as input data (intensity scaled image)
        # (per default, this is here the single band panchromatic image,
        #  other bands could be added by specifying PARAM['ADD_CHANNEL_IMG']
        #  in PARAM03_extract_data.py)
        PARAM['DATA_IMG'] = (
            f"{PARAM['PROJ_PREFIX']}_scale_{PARAM['SCALE_TYPE']}.tif")
        # Note: the DATA_IMG is also used as reference for image extent
        #       and resolution to match_project the label files.
        #       The label files are not used as reference as they can
        #       be smaller that the data image if have all zeros at the
        #       edges (NaN class)

        # Input data folder
        # Per default this should be 02_pre_proc as here the output from the
        # intensity adjustments e.g. 'BLyaE_SPOT2018_scale_std_8bit.tif'
        # is saved
        PARAM['DATA_FOLDER'] = '02_pre_proc'

        # Metadata, which was generated during the GLCM feature calculation
        # over the entire area (using texture_calc/MAIN_calc_texture.py).
        # The metadata is used to extract the GLCM features for the
        # test patches and the untiled ml_workflow input
        # If do not want to use any texture features, PARAM['META_FILE_TEX'] can be
        # set to None
        PARAM['META_FILE_TEX'] = (
            f"{PARAM['PROJ_PREFIX']}_all_{PARAM['SCALE_TYPE']}_P02_tex_proc_file.txt")


    # ------- TRAINING SET AND TILING SPECIFICATIONS ----
    if PROC_STEP in [3, 4]:
        # ---- NAMING of TRAINING SETS ----
        # The following is used for folder and file namings

        # Label specifying the split (which training, test and fine tuning AOIs)
        PARAM['SPLIT_SET_VERSION'] = 'v00'

        # labelling version (per patch set)
        PARAM['LABEL_VERSION'] = {
            # 'A01': 'Lv01',  #  add additional areas if required
            'A02': 'Lv01', 'test': 'Lv01', 'fine_tune': 'Lv01'
            }

        # ---- TILING PARAMETERS (for CNN training and prediction) ---
        # -- Window size for tiling --
        # The script adds additional padding to avoid edge effects for the
        # texture calculation this is done in GLCM calculation
        # (see param_utils.update_window_param(PARAM))
        PARAM['window_size'] = 256

        # How much to shift the clip window for tile overlap
        # !!! It is multiplied to the padded window size !!!
        # (Thus for very small windows one needs to make sure that there
        # is no gap.)
        PARAM['window_shift_factor'] = 0.5

        # Add degree if need to rotate labelling area (is true for BLyaE A01)
        PARAM['ROTATE_DEGREE'] = {# 'A01': -45,
                                  'A02': 0, 'all': 0}
        # after rotation pixels are interpolated to keep resolution
        # interpolation method is defined with:
        PARAM['rot_inperp'] = 'cubic'  # options here are 'linear', 'nearest', 'cubic'
        # (nearest does not provide good results for grey scale img)
        # interp is added to folder name: ...w298-298_v00_rot_cubic

    # ------- SETTINGS FOR OFFLINE AUGMENTATION ------
    if PROC_STEP in [4]:
        # ---- parameters for augmentations outside training loop ---
        # (for detailed option description see docs/PARAM_options_training_CNN.md)
        PARAM['n_augment'] = 4
        # probability to apply geometric augmentation (rotate, flip)
        PARAM['aug_geom_probab'] = [1.0, 0.5]
        # probability to apply color augmentation (BrightnessContrast, gamma)
        PARAM['aug_col_probab'] = [1.0, 0.5]
        # augmentation stengths
        PARAM['aug_col_param'] = [0.2, 0.2, 80, 120]
        # which band to use for color augmentation !!! is index not band name !!!
        PARAM['band_lst_col_aug'] = 0
        # which augmentation version to use
        PARAM['aug_vers'] = 1

    return PARAM

