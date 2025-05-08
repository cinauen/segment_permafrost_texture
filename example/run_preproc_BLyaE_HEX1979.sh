#!/bin/bash

# run script within the perma_segment folder
# nohup bash ./example/run_preproc_BLyaE_HEX1979.sh &>./example/BLyaE_HEX1979_preproc.log &

# adjust environment if required
source activate $R_ENVS/py_gpu_segment

# ===== adjust input parameters =====
GPU_NUM=7
INPUT_PARAM_FILE=PROJ_PARAM_BLyaE_HEX1979_v01
OUTPUT_FOLDER=BLyaE_v1
OUTPUT_PREFIX=BLyaE_HEX1979
SCALE_TYPE=perc0-2_g0-3_8bit
OUTPUT_SUFFIX=perc0-2_g0-3_8bit  # just used for log file suffix


# -- Assign the base path via the python script
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# set environment variables to path
eval $(python $SCRIPT_DIR/add_base_path.py)

BASE_PATH=$PROC_BASE_PATH


# ============ PROCESSING STEPS =========
# P01) --- Preproc greyscale image (scaling)
python ./src/preproc/MAIN_img_preproc.py ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/01_${OUTPUT_PREFIX}_image_preproc.log


# P02) --- Calculate texture for full area
python ./src/texture_calc/MAIN_calc_texture.py --PARAM_FILE PARAM02_calc_texture --GPU_DEVICE $GPU_NUM ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE $SCALE_TYPE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/02_${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_texture_calc_all.log
# --> extracts also statistics from whole image and sub AOIs


# P03) ---- Extract untiled training data (labels and data) for supervised classification
# here just the area A02 is processses to do several areas use e.g. A01:A02
python ./src/preproc/MAIN_extract_input_untiled.py --PARAM_FILE PARAM03_extract_data ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE A02 $SCALE_TYPE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/03_${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_extract_ML_training_input.log


# P03) ---- Extract untiled labels and data for test patches
# here the test patch areas test-01, test-02, test-03, test-04 are processed
python ./src/preproc/MAIN_extract_input_untiled.py --PARAM_FILE PARAM03_extract_data ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE test-01:test-02:test-03:test-04 $SCALE_TYPE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/03_${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_extract_test_patches_data.log


# P03) ---- Create training tiles for CNN
# here just the area A02 is processses to do several areas use e.g. A01:A02
python ./src/preproc/MAIN_create_training_tiles.py --PARAM_FILE PARAM03_extract_data ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE A02 $SCALE_TYPE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/03_${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_create_tiles.log


# P03) ---- Create prediction tiles (for full area prediction with CNN)
python ./src/preproc/MAIN_create_prediction_tiles.py --PARAM_FILE PARAM03_extract_data ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE $SCALE_TYPE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/03_${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_create_prediction_tiles.log


# P04) -- Augment and calculate texture for training tiles --
# --- OPTIONAL: for CNN if do offline augmentation when using GLCM input
# (training with offline augmentation is mainly suitable for pre-tests
# as it is much faster than on-the-fly augmentation where GLCMs are recalculated
# at every epoch after augmentation)
# here just the area A02 is processses to do several areas use e.g. A01:A02
python ./src/preproc/MAIN_augment_calc_texture.py --PARAM_FILE PARAM04_augment_data --GPU_DEVICE $GPU_NUM ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE A02 $SCALE_TYPE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/04_${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_augment_calc_texture.log



# P04) -------- Calcuate texture for prediction tiles --------
# --- OPTIONAL: for prediction with CNN including GLCM texture as input
python ./src/preproc/MAIN_calc_texture_tiles.py --PARAM_FILE PARAM04_augment_data --GPU_DEVICE $GPU_NUM ${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER} $INPUT_PARAM_FILE $SCALE_TYPE >${BASE_PATH}/1_site_preproc/${OUTPUT_FOLDER}/04_${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}_calc_texture_prediction_tiles.log







