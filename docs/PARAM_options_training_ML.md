## Definition of the training options for the ML workflow

The parameters for the training options are defined in:
param_settings/PARAM_inp_ML_train_v01.txt


### General notes for inputs:
- additional framework set parameters are defined in the
  project parameter file e.g.:<br>
  2_segment/01_input/PARAM06_RFtrain_HEX1979_A01_A02.py<br>


### Parameter explanation
- **PARAM_TRAIN_ID** (str)<br>
    ID used as identifier for parameter selection (e.g. "tML02").<br>
    Note: tML01 should be used for hyperparameter tuning
- **n_estimators** (int)<br>
    Total number of trees in the forest.<br>
    (cuml defualt: 100)
- **max_depth** (int)<br>
    Maximum tree depth. Must be greater than 0.<br>
    (cuml default: 16)
- **max_samples** (float)<br>
    Ratio of dataset rows used while fitting each tree.<br>
    (cuml default: 1.0)
- **max_features** (str, float, int)<br>
    Ratio of number of features (columns) to consider per node split.
    - 'sqrt': max_features=1/sqrt(n_features)
    - 'log2': max_features=log2(n_features)/n_features
    - None: max_features = 1.0
    - float: fraction of features
    - int: absolute count of features to be used
- **bootstrap** (boolean)<br>
    - True: each tree in the forest is built on a bootstrapped sample
        with replacement
    - False: the whole dataset is used to build each tree
- **comment**<br>
    Comment about the specific parameter set




