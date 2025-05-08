#!/bin/bash

# run script e.g. with
# nohup bash ./example/run_CNN_predict.sh &>./example/CNN_predict_t16v079.log &

# adjust environment if required
source activate $R_ENVS/py_cpu_segment

# ===== adjust input parameters =====
GPU_NUM=99  # with 99 do not calc on GPU
PARAM_TRAIN_ID=t16onl
PARAM_PREP_ID=v079
INPUT_PARAM_FILE=PARAM06_predict_model_BLyaE_v1_HEX1979_A02_set01a_on_BLyaE
CV_NUM_INP=(0 1 2)
# adjust the epchs to the best or last epoch
epoch_arr_CV=(43 72 54)

# -- Assign the base path via the python script
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# set environment variables to path
eval $(python $SCRIPT_DIR/add_base_path.py)

BASE_PATH=$PROC_BASE_PATH
TEMP_PATH=$PROC_TEMP_PATH
# $PROC_TEMP_PATH


# loop through CVs
for id_f in "${!CV_NUM_INP[@]}"; do

python ./src/cnn_workflow/MAIN_predict.py --GPU_LST_STR $GPU_NUM --CV_NUM ${CV_NUM_INP[id_f]} ${BASE_PATH}/2_segment $TEMP_PATH $PARAM_PREP_ID $PARAM_TRAIN_ID ${epoch_arr_CV[id_f]} $INPUT_PARAM_FILE

done





