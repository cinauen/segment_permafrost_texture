# ML training, evaluation and prediction (P06)
## Summary
The ML segmentation can involve input data from several sites and from
different satellite imagery.<br>
(all outputs are saved in the 2_segment folder).

The parameters for train, grid search, feature importance analysis and testing
can be provided in one project file. It defines the following:

- Input data: Which data to include by providing the path to the specific
  folders with the data/label tiles (*example/1_site_preproc/SITE/03_train_inp/*)
- area (AOI) to use per phase and cross-validation run: the phases can:
    - train
    - validation
    - test_SITE_YEAR: e.g. BLyaE_HEX1979_test (by adding specific sites
      or imagery to the test phase name it is possible to analyse the
      performance per site/imagery separately)
- Class label options: in case classes need to be relabelled
- Other training and performance specific options

A parameter example file can be found in *example/2_segment/01_input/*:

- *PARAM06_RFtrain_HEX1979_A02.py.py*

Various training setups can be tested by selecting:

 - Different feature preparation options as pre-defined in
   *src/param_settings/PARAM_inp_ML_feature_prep_v01.txt* and selected
   with PARAM_PREP_ID
 - Training options as pre-defined in *src/param_settings/PARAM_inp_ML_train_v01.txt*
   and selected with PARAM_TRAIN_ID

The processing phase can be is defined as command line input (PROC_STEP parmeter).
It can be:

 - 'gs': hyperparameter tuning (grid search)
 - 'fi': feature importance (shap analysis)
 - 'rt': train, test, predict

All python scripts are located in *ml_workflow*


## Hyperparameter tuning

Tests different Random Forest parameter setups.

**Script** *MAIN_RF_classify.py* or *MAIN_RF_classify_CV.py*<br>
combined with PROC_STEP: 'gs'
(..CV.py loops through all cross-validation options)

**Input**

  - Untiled training data as saved in
    *example/1_site_preproc/SITE/03_train_inp/*

**Main Output**
(in 2_segment/02_train/MODEL_BASE_FOLDER_ML_gs/{PARAM_PREP_ID}_{PARAM_TRAIN_ID}_cvXX)

 - Grid search output table providing a ranking and the accompaigning metrices
    for different parameter combinations


## Feature importance analysis
Evaluate the importance of different input features.

**Script** *MAIN_RF_classify.py* or *MAIN_RF_classify_CV.py*<br>
combined with PROC_STEP: 'fi'
(..CV.py loops through all cross-validation options)

**Input**

  - Untiled training data as saved in
    *example/1_site_preproc/SITE/03_train_inp/*

**Main Output**
(in 2_segment/02_train/MODEL_BASE_FOLDER_ML_fi/{PARAM_PREP_ID}_{PARAM_TRAIN_ID}_cvXX)

 - when run on GPUs: plot of SHAP feature importances
 - when run on GPU or CPU: scikit-learn permutation test plot


## Train, test, prediction
Trains the model and runs the test on the test patches.
The Random Forest parameters (defined with PARAM_TRAIN_ID) and the feature
selection (defined with PARAM_PREP_ID) can be selected according to the
above tests.

**Script** *MAIN_RF_classify.py* or *MAIN_RF_classify_CV.py*<br>
combined with PROC_STEP: 'rt'
(..CV.py loops through all cross-validation options)

**Input**

  - Untiled training data as saved in
    *example/1_site_preproc/SITE/03_train_inp/*
  - Test tiles saved in the subfolders in
    *example/1_site_preproc/SITE/03_train_inp/*

**Output**
(in 2_segment/02_train/MODEL_BASE_FOLDER_ML_fi/{PARAM_PREP_ID}_{PARAM_TRAIN_ID}_cvXX)

 - Model
