#!/bin/bash

# run script with
# nohup bash ./example/run_CNN_train_offline_aug.sh &>./example/CNN_train_test_off_t16v079.log &

# adjust environment if required
source activate $R_ENVS/py_gpu_segment


# ===== adjust input parameters =====
GPU_NUM=6
PARAM_TRAIN_ID=t16off
PARAM_PREP_ID="v079"  # "v079,v158" (use entries divided by comma to loop through different setups)
INPUT_PARAM_FILE_TRAIN=PARAM06_train_BLyaE_v1_HEX1979_A02_set01a
INPUT_PARAM_FILE_TEST=PARAM06_test_model_BLyaE_v1_HEX1979_A02_set01a
CV_NUM=0

# -- Assign the base path via the python script
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# set environment variables to path
eval $(python $SCRIPT_DIR/add_base_path.py)

BASE_PATH=$PROC_BASE_PATH


# separate the PARAM_PREP_ID input into an array
IFS=',' read -r -a ID_FEATURE_PREP_ARRAY <<< "$PARAM_PREP_ID"

# ----- loop thorugh different training options
for id_f in "${ID_FEATURE_PREP_ARRAY[@]}"; do

	echo ${id_f} is running

	# ---- Training by looping through all cross-validations ------
	python ./src/cnn_workflow/MAIN_train_offl_aug_CV.py --GPU_LST_STR $GPU_NUM ${BASE_PATH}/2_segment $id_f $PARAM_TRAIN_ID $INPUT_PARAM_FILE_TRAIN
	#python ./src/cnn_workflow/MAIN_train_offl_aug.py --GPU_LST_STR $GPU_NUM --CV_NUM $CV_NUM ${BASE_PATH}/2_segment $id_f $PARAM_TRAIN_ID $INPUT_PARAM_FILE_TRAIN


	# ---- Testing by looping through all cross-validations ------
	python ./src/cnn_workflow/MAIN_evaluate_model_CV.py --GPU_LST_STR $GPU_NUM ${BASE_PATH}/2_segment $id_f $PARAM_TRAIN_ID $INPUT_PARAM_FILE_TEST
	#python ./src/cnn_workflow/MAIN_evaluate_model.py --GPU_LST_STR $GPU_NUM --CV_NUM $CV_NUM ${BASE_PATH}/2_segment $id_f $PARAM_TRAIN_ID $INPUT_PARAM_FILE_TEST


done







