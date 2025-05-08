# File structure and naming

The generated GTiff file structure used as further input in the processing
consists of the following:


## Base data

This is the panchromatic single band imagery after intensity adjustment.
Different extents are created depending on the utilisation (e.g. training, testing, prediction).
  - <u>Full area files</u><br>
    BLyaE_HEX1979_scale_perc0-2_g0-3_8bit.tif <br>
    `{site}_{imagery tag}_{scaling type}_{bit depth}.tif`

  - <u>Untiled training or test areas files</u><br>
    (used as ML input or for distributed test patches): <br>
    BLyaE_HEX1979_A02train-01_data.tif <br>
    BLyaE_HEX1979_test-01_data.tif <br>
    `{site}_{imagery tag}_{area tag}_data.tif`

  - <u>Tiles (used as CNN input)</u>: <br>
    BLyaE_HEX1979_A02train-01_00_00-00_data.tif (for training)<br>
    BLyaE_HEX1979_all_00_00-01_data.tif (for prediction)
    `{site}_{imagery tag}_{area tag}_{shift number}_{tile local coord}_data.tif`


## Derived input features

This are the features derived from the base data (i.e. the GLCM textures).
The, files contain all derived texture features as separate bands
(HOMOGENEITY, CONTRAST, ASM, VAR, CORRELATION, DISSIMILARITY, MEAN).

However, separate files are created per GLCM input parameters e.g.
*BLyaE_HEX1979_A02train-01_data_a0-1-2-3_r05_norm_C01.tif* or<br>
*BLyaE_HEX1979_A02train-01_data_r05_calc_std.tif*<br>
`{site}_{imagery_tag}_{area}_{derived_feature_tag}.tif`<br>
*BLyaE_HEX1979_A02train-01_00_00-00_data_a0-1-2-3_r05_norm_C01.tif* (tiles)<br>
`{site}_{imagery tag}_{area tag}_{shift number}_{tile local coord}_data_{derived_feature_tag}.tif`

The `derived_feature_tag` refers to the GLCM properties as follows<br>
`{GLCM direction}_{GLCM window radius}_{norm}_{channel number}`<br>
  - <u>GLCM direction</u> a0-1-2-3 (0, 1, 2, 3 are all directions E, SE, S, SW)
  - <u>GLCM window radius</u> r05 window radius -> window size = 11 x 11 pixels
  - <u>norm</u> means that the GLCM features ranges were moved to the [0, 1]
      range (from the min and max possible values, not from the actual
      min max values)
  - <u>channel number</u> the amount od channels from ewhich the GLCMs were
    derived (here single channel data --> C01)<br>

or `{GLCM window radius}_{stats calculated}`<br>
  - <u>stats calculated</u> the statistics (e.g. calc_std = standardeviation
    over the GLCMs calculated for all four different directions

Depending on the area coverage (i.e. full area, tiles or untiled areas),
the `derived_feature_tag` is just added to the base data file name.

## Augmented tiles for offline augmentation
If use offline augmentation the n the augmented tiles are saved and the GLCMs are recalculates. The file names are then e.g.:

- *BLyaE_HEX1979_A02train-03_03_02-01_data_aug1-01.tif* (augmented image)
- *BLyaE_HEX1979_A02train-03_03_02-01_data_aug1-01_a0-1-2-3_r05_norm_C01.tif* (GLCMs calculated after augmentation)<br>
  `{site}_{imagery tag}_{area tag}_{shift number}_{tile local coord}_data_{augmentation label}_{derived_feature_tag}.tif`<br>
  augmentation label: augmentation type 1, first augmentation number



## Ground truth data

The ground truth data consist of the manually created class labels and
optionally the pixel uncertainty weights.

For te different area coverage they are as follows
  - <u>Untiled labelling area area tile</u><br>
    BLyaE_HEX1979_A02train-01_seg.tif<br>
    `{site}_{imagery tag}_{area tag}_seg.tif`
  - <u>Label tiles</u><br>
    BLyaE_HEX1979_A02train-01_01_00-01_seg.tif<br>
    `{site}_{imagery tag}_{area tag}_{shift number}_{tile local coord}_seg.tif`


Note: The grid of the labels and the corresponding data match exactly.