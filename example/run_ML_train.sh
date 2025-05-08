#!/bin/bash

# run script with
# nohup bash ./example/run_ML_train.sh &>./example/ML_train_test_vML017tML02_cv00.log &

# adjust environment if required
source activate $R_ENVS/py_gpu_segment

# ===== adjust input parameters =====
GPU_NUM=5
N_JOBS=20
PARAM_PREP_ID="vML017"  # for several items separate by comma e.g. "vML017,vML018"
PARAM_TRAIN_ID=tML02
PARAM_FILE=PARAM06_RFtrain_HEX1979_A02
CV_NUM=0

# -- Assign the base path via the python script
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# set environment variables to path
eval $(python $SCRIPT_DIR/add_base_path.py)

BASE_PATH=$PROC_BASE_PATH


# separate the PARAM_PREP_ID input into an array
IFS=',' read -r -a ID_FEATURE_PREP_ARRAY <<< "$PARAM_PREP_ID"


# ------- hyerparameter tuning
# ---- grid search ------
# for grid seach choose tML01
python ./src/ml_workflow/MAIN_RF_classify.py --GPU_LST_STR $GPU_NUM --N_JOBS $N_JOBS --CV_NUM $CV_NUM ${BASE_PATH}/2_segment 'gs' vML060 tML01 $PARAM_FILE

# ---- feature importance ------
# select best traiing setup from grid search
# is here tML02
# later use feature importance result to select most important features to inlcude in training
python ./src/ml_workflow/MAIN_RF_classify.py --GPU_LST_STR $GPU_NUM --N_JOBS $N_JOBS --CV_NUM $CV_NUM ${BASE_PATH}/2_segment 'fi' vML060 tML02 $PARAM_FILE

# ----- loop thorugh different training options
for id_f in "${ID_FEATURE_PREP_ARRAY[@]}"; do

	echo ${id_f} is running

	# ---- Training, predict validation and test ------
	python ./src/ml_workflow/MAIN_RF_classify.py --GPU_LST_STR $GPU_NUM --N_JOBS $N_JOBS --CV_NUM $CV_NUM ${BASE_PATH}/2_segment 'rt' $id_f $PARAM_TRAIN_ID $PARAM_FILE

done







