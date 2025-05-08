
## Definition of the training options for the CNN workflow

The parameters for the training options are defined in:
param_settings/PARAM_inp_CNN_train_v01.txt


### General notes for inputs:
- additional framework set parameters are defined in the
  project parameter file e.g.:<br>
  2_segment/01_input/PARAM06_train_BLyaE_v1_HEX1979_A01_A02_set01a.py<br>
  2_segment/01_input/PARAM06_test_model_BLyaE_v1_HEX1979_A01_A02_set01a_on_BLyaE.py<br>
  2_segment/01_input/PARAM06_predict_model_BLyaE_v1_HEX1979_A01_A02_set01a_on_BLyaE.py<br>


### Parameter explanation
- **PARAM_TRAIN_ID** (str)<br>
    ID used as identifier for parameter selection (e.g. "t16onl").<br>
    - "onl": is for using on-the-fly augmentation (training run with
      MAIN_train_incl_aug.py)
    - "off: is for using offline augmentation (training run with
      MAIN_train_offl_aug.py)
- **N_BATCH** (str)<br>
    batch number to be used in the training.
    For testing and prediction it is overwritten in the project parameter
    file.
- **MODEL_ARCH** (str)<br>
    CNN model architecture. Possible options are: smpUNet, smpUNetPP,
    smpMAnet, smpDeepLabV3, smpDeepLabV3plus
    (the specified models are initialized in custom_model.py and
    correspond to predefined models from segmentation_models_pytorch)
- **MODEL_ENCODER_DEPTH** (int)<br>
    Maximum depth possible is 5.
- **loss_weights** (1111, 9999 or None)
    Type of loss weighting:
    - 1111: use class weights as counted form the whole training set
    - 9999: weights are adapted for each batch according to number of
        pixels per class
    - None: no class weights are used
    Note: weights are set to 0 for background pixels
- **loss_reduction** (str: 'mean', 'sum', 'none')
    If sample weights are used and how the final loss is calculated
    - 'none': SAMPLE WEIGHTS ARE USED <br>
        Note: Without reduction (='none'), the loss is provided per pixel,
            which were optionally weighted by the class weight (option 1111
            or 9999 above). The pixel loss is then additionally weighted
            with the sample weights (pixel uncertainty), according to the
            "weight_calc_type" condition.
    - 'mean': NO USAGE OF SAMPLE WEIGHTS<br>
        Note: loss corresponds to sum(loss_px*class_weight_px)/sum(class_weights_px)
            = weighted mean <br>
    - 'sum': NO USAGE OF SAMPLE WEIGHTS<br>
        Note: loss corresponds to sum(loss_px*class_weight_px)/cts_of_non_zero_pixels
            = actual mean <br>
- **weight_calc_type** (str: 'not_used', 'norm_weight', 'average_weight') <br>
    How sample weights (pixel uncertainty) are included in the loss function
    - not_used: if loss_reduction != 'none' (thus if only class weights are used)
    - norm_weight: sum(loss_class_weighted_px/sum(class_weights_px) * sample_weight_px)
    - average_weight: = mean(loss_class_weighted_px * sample_weight_px)
        (same as in Bressan et al (2022))
- **otimizer** (str) <br>
    Pytorch optimiser type (e.g. torch.optim.Adam)
- **n_augment** (int) <br>
    Number of augmentations.
    - for offline augmentation the amount corresponds to:
        1 (original dataset) + n_augment x (separately augmented data set)
    - for augmentation corresponds to n_augment-fold augmentation
        (i.e. dataset is increased four times)
- **aug_geom_probab** (str list separated by ":" with two items, None) <br>
    At what probability geometric augmentation is applied.
    e.g. "0.5:0.5" for 0.5 probability to rotate and 0. probability to mirror
    None is for no geometric augmentation.
- **aug_col_probab** (float list separated by ":" with either two or three items, None) <br>
    At what probability color augmentation is applied. E.g.:<br>
    - "0.5:0.5" 0.5 probability RandomBrightnessContrast adjustments and
        0.5 probability RandomGamma <br>
    - "0.5:0.5:0.2"  0.5 probab RandomBrightnessContrast, 0.5 probab
        RandomGamma and ColorJitter, 0.2 probab sharpen and blur<br>
- **aug_col_param** (float/int list separated by ":" with four items, None) <br>
    Defines strength of color augmentation <br>
    e.g. "0.2:0.2:80:120" is brightness_limit=0.2, contrast_limit=0.2,
        gamma_min=80, gamma_max=120
- **aug_vers** (int) <br>
    ONLY used for OFFLINE AUGMENTATION to define which augmentation set to use.
    1 for example corresponds to augmentation set 1 and for example metadata file
    BLyakhE_HEX1979_A02_train-01_meta_data_augment_Av1.txt is used.
    Parameters for augmentation set generation for offline augmentation are
    defined in the project parameter file e.g.
    1_site_preproc/BLyaE_v1/01_input/PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01.py
- **IF_FAD_AUG** (boolean) <br>
    !!! NOT IMPLEMENTED IN THE CURRENT VERSION !!!
    If Fourier domain adaptation (FDA) augmentation should be used.
- **band_lst_col_aug** (int) <br>
    Which band to augment (corresponds to number index and not band name).
    Is 0 for default greyscale. The GLCMs in this case not augmented but
    recalculated after the grescale imagery augmentation.
- **metrics_ignore_index** (int or -100) <br>
    Defines a class index (orderd number index not class name) should be
    ignored in the metrics calculation.
    Per default this value should be kept as 0 as the nodata class should
    be ignored in the metrics calculation. Can be set to -100 if to not
    want to ignore any index.
- **learning_rate** (float) <br>
    Initial learning rate. E.g. 0.01, 0.001, 0.0001
- **lr_scheduler** (str) <br>
    - None: no learning rate scheduler, this learning rate is not changes
        and stays at the initial learning_rate
    - exp: exponential learning rate decay.
    - mulstep: multistep learning rate as define lr milestones
- **lr_gamma** (float) <br>
    Defines the decay of the learning rate of each parameter group
    (for exp at every epoch or for mulstep at the specified epochs).
- **lr_milestones** (int list separated by ":") <br>
    Epochs at which the learning rate of each parameter group is decayed
    by gamma (lr_gamma).
- **use_batchnorm** (boolean)<br>
    If batchnorm should be applied.
    Note: we kept it as True for all training frameworks as without
    batchnorm the convergence was bad
- **comment**<br>
    Comment about the specific parameter set


