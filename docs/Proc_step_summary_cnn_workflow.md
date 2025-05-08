# CNN training, evaluation and prediction (P06)
## Summary
The CNN segmentation can involve input data from several sites and from
different satellite imagery.<br>
(all outputs are saved in the 2_segment folder).

The input parameters must be defined in project and phase (e.g train,
test, fine-tune, predict) specific parameter files. The inputs to be
defined include the following:

- Input data: Which data to include by providing the path to the specific
  folders with the data/label tiles (*example/1_site_preproc/SITE/03_train_inp/*)
- Area (AOI) to use per phase and cross-validation run: the phases can:
    - train
    - validation
    - test_SITE_YEAR: e.g. BLyaE_HEX1979_test (by adding specific sites
      or imagery to the test phase name it is possible to analyse the
      performance per site/imagery separately)
- Class label options: in case classes need to be relabelled
- Other training and performance specific options

Examples of parameter files can be found in *example/2_segment/01_input/*:

- *PARAM06_train_BLyaE_v1_HEX1979_A01_A02_set01a.py* (train)<br>
- *PARAM06_test_model_BLyaE_v1_HEX1979_A01_A02_set01a_on_BLyaE.py* (test)<br>
- *PARAM06_train_BLyaE_v1_HEX1979_A01_A02_set01a_v079t16_fine_tune_on_FadN.py* (fine-tune)<br>
- *PARAM06_predict_model_BLyaE_v1_HEX1979_A01_A02_set01a_on_BLyaE.py* (predict)<br>

Various training setups can be tested by selecting:

 -  Different feature preparation options as pre-defined in *src/param_settings/PARAM_inp_CNN_feature_prep_v01.txt* and selected with PARAM_PREP_ID
 - Training options as pre-defined in *src/param_settings/PARAM_inp_CNN_train_v01.txt*
   and selected with PARAM_TRAIN_ID


All python scripts are located in *cnn_workflow*


## On-the-fly or offline training and fine-tuning

*On the fly augmentation*: Training and validation, while the augmentation
and the recalculation of the GLCMs for the augmented imagery is done within
the training loop at every epoch.
*Offline augmentation*: Training and validation while pre augmented data
and recalculated GLCMs are directly loaded but (i.e. no augmentation
during training)<br>
Note: Offline augmentation should mainly be considered for fast testing
as it is much faster than on-the-fly augmentation. However, it is prone
to overfitting.
*Fine-tuning*: Training and validation on a pre-trained model
including additional fine-tuning data (only implemented for on-the-fly
augmentation). This can be used to adapt a model to an additional site or
imagery

**Script** *MAIN_train_incl_aug.py* or *MAIN_train_offl_aug.py*<br>
*MAIN_train_incl_aug_CV.py* or *MAIN_train_offl_aug_CV.py*
(loops through all cross-validation options)

**Input**

  - Training tiles or or fine-tuning tiles saved in the subfolders in
    *example/1_site_preproc/SITE/03_train_inp/*
  - Pre-trained model (for fine-tuning)
  - File names and class counts from metadata

**Output**
(in 2_segment/02_train/MODEL_BASE_FOLDER/{PARAM_PREP_ID}_{PARAM_TRAIN_ID}_cvXX)

 - CNN model saved at epochs depending on performance improvement and
   specified epoch numbers (.tar files)<br>
 - Metrics summaries including class IoU and dice, confusion matrix,
   overall recall, precision, accuracy and IoU for training and validation
   (calculated with torchmetrics (tm) and ignite (ign) for comparison)<br>
   e.g. v079t16onl_cv00_ncla7_summary_trainRun_tm.txt
 - Performance plots during training:

    - Learning curves with loss or class IoU (is updated using training)<br>
      e.g. v079t16onl_cv00_ncla7_loss_acc.pdf<br>
      e.g. v079t16onl_cv00_ncla7_class_IoU.pdf<br>
    - Confusion matrix for validation<br>
      e.g. v079t16onl_cv00_ncla7_confusion_m_validate_ep99.png
    - Prediction examples for validation<br>
      e.g. v079t16onl_cv00_ncla7_validate_ep99_pred.pdf



## Model evaluation on test patches
Model testing on the distributed test patches (for RF and CNN segmentation).

**Script** *MAIN_evaluate_model.py* or
*MAIN_evaluate_model_CV.py* (loops through all cross-validation options)<br>

**Input**

  - Trained model
  - Test tiles saved in the subfolders in *example/1_site_preproc/SITE/03_train_inp/*
  - File names from metadata

**Output** (in 2_segment/02_train/ within model output folder from trainig)
The output is saved in a subfolder names after the test phase (e.g. BLyaE_HEX1979_test)
The filenames contain the segmentation framework prefix (e.g. v079t16onl_cv00_ncla7).

 - Predicted classes and probabilities as well as the true classes per
   test patch.
   Saved as raster (.tif) and shape file (.geojson)<br>
 - Test metrics including: class IoU and dice, confusion matrix,
   overall recall, precision, accuracy and IoU
   (calculated with torchmetrics (tm) and ignite (ign) for comparison)
   e.g. v079t16onl_cv00_ncla7_summary_class_dice_BLyaE_HEX1979_test_cummul_tm.txt
 - Analysis of True positives and false positives per shape certainty (weight)
   (only for the specified class)
   e.g. ..._BLyaE_HEX1979_test-01_ep10_class_TP_FN_class1.txt
   `..._{site, imagery and number of tested patch}_ep{epoch number}_class_TP_FN_class{evaluated class}.txt`
 - Geometric properties of the predicted and true shapes
   e.g. ..._BLyaE_HEX1979_test-04_ep64_class_pred.txt
   `..._{site imagery and number of tested patch}_ep{epoch number}_class_{true or predicted}.txt`
 - Metric plots: confusion matrix (.png), predicted test patches (.pdf)


## Prediction
Prediction on the full area.<br>
Per default the tiles are predicted separately, center cropped
(to avoid edge effects) and then merged by the "first"-choice rule or by
averaging the probabilities and taking softmax. Optionally it is also
possible to predict the entire prediction area at once (PARAM['prediction_type']).<br>
For all input data, the same standardisation as was applied to the training
data is also applied to the prediction data.

**Script** *MAIN_predict.py*<br>

**Input**

  - Trained model
  - Prediction tiles saved in the subfolders in *example/1_site_preproc/SITE/03_train_inp/*
    or is use the entire area prediction the intensity corrected input image
    (e.g. from *example/1_site_preproc/SITE/02_pre_proc*)<br>
  - File names from metadata (for tile type prediction)

**Output** (in 04_predict within model and framework specific sub-folders)

 - Predicted classes and probabilities saved as raster (.tif)<br>
   e.g. in folder .../BLyaE_v1_pred_HEX1979_perc0-2_g0-3_8bit/BLyaE_v1_HEX1979_A02/cv00
   `{prediction area tag}_{used model folder name}_{CV num}.txt`
   Files with tile merging are:
   - ..._tile_merged_pred_class.tif (tile merged classes with "first"-choice rule)
   - ..._tile_merged_pred_proba_raw_mean_softmax_class.tif
     (tile averaged probabilities and classes derived after softmax and with argmax)
   - ..._tile_merged_pred_proba_raw_mean_proba_softmax.tif
     (tile averaged probabilities after softmax)

