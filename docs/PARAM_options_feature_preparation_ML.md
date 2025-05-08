## Definition of the feature preparation options for the ML workflow

The parameters for the feature preparation options are defined in:
param_settings/PARAM_inp_ML_feature_prep_v01.txt

### General notes for inputs:
- Additional input features (GLCM features) are defined by the file
  suffix (file_suffix_lst) and the band names (merge_bands). The default
  data file (greyscale imagery) and its band (1) need also to be included
  or excluded in merge_bands.
- For testing, instead of using the custom_data_loader module, scikit-learn
  can be used for pre-processing. This is the case when the PARAM_PREP_ID
  includes scikit (vMLXXX_scikit). In this case the data_loader option
  are not used but instead the scikit options at the end of the following
  list are used.
- additional framework set parameters are defined in the
  project parameter file e.g.:
  2_segment/01_input/PARAM06_RFtrain_HEX1979_A01_A02.py

### Parameter explanation
- **PARAM_PREP_ID** (str)<br>
    ID used as identifier for parameter selection
    (e.g. "vML000").
- **file_suffix_lst:** (str list separated by ":")<br>
    Defines the file suffixes of additional feature inputs (here GLCMs).
    For example "a0-1-2-3_r05_norm_C01:a0-1-2-3_r10_norm_C01"
    corresponds to the GLCMs files with window size 11 pixels (radius 5px)
    and 21 pixels (radius 10px). For both files, the GLCMs were calculated
    in all directions [0, 1, 2, 3 = E, SE, S, SW] and averaged.
    The norm indicates that all GLCM textures were moved to the 0 - 1 range.
    However, the norm here is calculated by using the possible min/max
    values and NOT by using the actual min/max values (see further below).
- **merge_bands** (str list separated by ":")<br>
    Defines which bands to include in the classification<br>
    For example "1:HOMOGENEITY:CONTRAST:ASM", where 1 corresponds to the
    greyscale image (band 1) and HOMOGENEITY, CONTRAST, ASM correspond
    to the bands of the GLCM files.
- **feature_add** (identifier: "sensor" or None)<br>
    If another feature should be generated. Implemented is the option
    "sensor". This would add the sensor number as an additional feature.<br>
    Nothing is done with None.
- **standardize_add** (boolean), data_loader option<br>
    If the additional input features (GLCMs) should be standardised by
    using the mean and standard deviation (std) calculated from the training areas
    (e.g. A01, A02) and per satellite imagery (e.g. SPOT or HEX).
    The values are read from the pre-calculated stats file as defined
    with the parameter PARAM['STATS_FILE']
- **standardize_individual** (identifier: "No", "add", "all"), data_loader option<br>
    Standardisation is done based on mean and std of the individual tile.<br>
    - "add": only additional features (GLCMs, ass specified with
    file_suffix_lst and merge_bands) are standardized<br>
    - "all": all bands are standardised
- **if_norm_min_max_band1** (boolean), data_loader option<br>
    bring band 1 (greyscale band) from [1 bit_depth] to [0 1]
- **if_std_band1** (boolean), data_loader option<br>
    standardise band 1 with the mean and std of the
    training area<br>
    (!!! this is only done if if_norm_min_max_band1 is False !!!)
- **norm_on_std** (float), data_loader option<br>
    normalize bands after standardisation<br>
    With 999 no normalisation if done.
    With e.g. 0.5 the values are clipped at the 0.5 and the 99.5 percentile.
    Values outside the percentiles are set to the min/max value.
    If "standardize_add" and/or "if_std_band1" are True, then the
    percentiles are taken from the training area (precalculated stats file).
- **norm_clip** (boolean), data_loader option<br>
    If clip the input features to [0 1] after normalisation (only used if
    norm_on_std != 999). Values smaller than 0 or larger than 1 are set
    to 0 or 1 respectively.
- **calc_PCA** (identifiers: "all", "separate", False), data_loader option<br>
    If calculate the principal component analysis (PCA) from the
    additional features (GLCMs)<br>
    - "all": the PCA are calculated over all bands and additional features.
    - "separate": the PCAs are calculated per additional features file.
       Thus, e.g. per a0-1-2-3_r05_norm_C01 or per a0-1-2-3_r10_norm_C01
       separately.
- **PCA_components** (int), data_loader option<br>
    Number of principal components to be calculated.
- **take_log**, (boolean) data_loader option<br>
    If log should be taken from the additional features (GLCMs)
- **preproc_scikit** (str), scikit option for testing (ONLY for vMLXXX_scikit)<br>
    Defines how pre-processing is done.<br>
    - ":": no standardisatin is used
    - if not "No": scikit-lean module is used to pre-process the data.<br>
        Options are: , standardscaler, normalizer, minmaxscaler, pca <br>
        The options can be chained with ":" (e.g. standardscaler:pca)
- **take_log_scikit**, scikit option for testing (ONLY for vMLXXX_scikit)<br>
    - None: nothing is done
    - { 'log': ['CONTRAST', 'ASM', 'VAR', 'DISSIMILARITY'],
        'exp': ['HOMOGENEITY',   'CORRELATION', 'MEAN']} dict defines
        for which band log or exp will be taken.
- **comment**<br>
    Comment about the specific parameter set



### Note on GLCM normalisation:
Within GLCM_cupy the calculated GLCM are normalised according to the
possible min/max values (not according to the actual min/max values).:

Source glcm_cupy
(https://github.com/Eve-ning/glcm-cupy/blob/master/glcm_cupy/utils.py)

```
def normalize_features(ar: ndarray, bin_to: int) -> ndarray:
    """ This scales the glcm features to [0, 1] """
    ar[..., Features.CONTRAST] /= (bin_to - 1) ** 2
    ar[..., Features.MEAN] /= (bin_to - 1)
    ar[..., Features.VARIANCE] /= (bin_to - 1) ** 2
    ar[..., Features.CORRELATION] += 1
    ar[..., Features.CORRELATION] /= 2
    ar[..., Features.DISSIMILARITY] /= (bin_to - 1)
    return ar
```