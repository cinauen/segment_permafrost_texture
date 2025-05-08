# Example

This *example*-folder contains the folder structure and bash scripts
required to run the processing.
If you want to run the example and see the outputs, please download
the full example-set from
*Zenodo: http://dx.doi.org/10.5281/zenodo.15325757 ***(NOT YET PUBLIC)***. <br>

As input imagery, the example-set contains a clipped and georeference
Hexagon (KH-9PC) image from the site BLyaE and FadN is included.
The original data was provided by the United States Geological Survey
Earth Resources Observation and Science (EROS) Center (USGS, 2018)*
and can be downloaded from the EarthExplorer (https://earthexplorer.usgs.gov/).

The manual labels were generated as part of the paper accompanying this
script. The complete set of all labels, the pre-processed data, as well as
the final output predictions can be downloaded from
*Zenodo: http://dx.doi.org/10.5281/zenodo.15325742 ***(NOT YET PUBLIC)*** <br>

Please note that the output predictions resulting from this example differ
from the results provided in the accompanying paper, which used the full
set of training data including the SPOT-7 imagery which cannot be provided
due to licencing restrictions.



## Project folder structure

```
example
├── 1_site_preproc (site specific pre-processing)
│   ├── BLyaE_v1 (site 1)
│   │   ├── 00_labelling (manual labels)
│   │   ├── 01_input (input imagery, AOIs, project parameter files)
│   │   ├── 02_pre_proc (output: intensity scaling texture calc)
│   │   └── 03_train_inp (output: training, test, prediction data/tiles)
│   ├── FadN_v1
│   │   └── ...
└── 2_segment (segmentation)
    ├── 01_input (parameter files for train, test, prediction)
    ├── 02_train (training outputs)
    ├── 03_analyse (comparison figures of different models)
    └── 04_predict (output predictions)
```

## Running the scripts
The complete workflow can be run-through with the following bash scripts:

 - Pre-processing: [example/run_preproc_BLyaE_HEX1979.sh](run_preproc_BLyaE_HEX1979.sh)
 - ML workflow: [example/run_ML_train.sh](run_ML_train.sh)
 - CNN workflow:
    - train and test with on-the-fly augmentation:
      [example/run_CNN_train_online_aug.sh](run_CNN_train_online_aug.sh)
    - train and test with offline augmentation:
      [example/run_CNN_train_offline_aug.sh](run_CNN_train_offline_aug.sh)
    - fine-tune and test with on-the-fly augmentation:
      [example/run_CNN_fine_tune.sh](run_CNN_fine_tune.sh)

Note: different segmentation frameworks can be tested by adjusting the
PARAM_PREP_ID (see [options ML](./../docs/PARAM_options_feature_preparation_ML.md)
and [options CNN](./../docs/PARAM_options_feature_preparation_CNN.md))
and PARAM_TRAIN_ID (see: [options ML](./../docs/PARAM_options_training_ML.md)
and [options CNN](./../docs/PARAM_options_training_CNN.md).


### Processing setup adjustments
The following can be adjusted:
- Project parameter:
    - site and imagery specific pre-processing parameters in:
      [1_site_preproc/BLyaE_v1/01_input](1_site_preproc/BLyaE_v1/01_input)<br>
      e.g. *PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01.py*
    - Training, testing and prediction specific parameters:
      [2_segment/01_input](2_segment/01_input)<br>

- Project path in [example/add_base_path.py](add_base_path.py)
  (if the processing directory needs to be changed)


<br>
<br>

----
*USGS. (2018). Archive - Declassified Data - Declassified Satellite Imagery - 3.
Earth Resources Observation and Science (EROS) Center. doi: 10.5066/F7WD3Z10