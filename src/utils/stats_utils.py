"""
Functions to retrieve statistics of raster files as well as
functions to calculate normalise and/or standardise array.

Uses dask for parallelisation and allows usage of GPUs.

"""
import numpy as np
import pandas as pd
import dask

import utils.geo_utils as geo_utils


def get_stats_df(xarr, perc_min=0.5, perc_max=99.5, prefix=''):

    stats_lst = get_stats_per_band(
        xarr, perc_min=perc_min, perc_max=perc_max, prefix=prefix)

    df_out_lst = []
    for i_name, i_val in stats_lst:
        df_out_lst.append(i_val.to_dataframe(name=i_name)[[i_name]])
    return pd.concat(df_out_lst, axis=1)


def get_stats_per_band(xarr, perc_min=0.5, perc_max=99.5, prefix=''):
    """
    dask.compute(...) is faster than separate compute e.g.
    tt_std = tt.std(dim=('x', 'y'), skipna=True).compute()
    """
    if xarr.dtype.name.find('int') > -1:
        # convert to float due to merge later
        xarr = geo_utils.convert_img_to_dtype(
            xarr, dtype_out='float64', nodata_out=np.nan,
            replace_zero_with_nan=True)

    band_min, band_max, band_mean, band_std, band_qmin, band_qmax, band_qmin1, band_qmax1 = dask.compute(
        xarr.min(dim=('x', 'y'), skipna=True),
        xarr.max(dim=('x', 'y'), skipna=True),
        xarr.mean(dim=('x', 'y'), skipna=True),
        xarr.std(dim=('x', 'y'), skipna=True),
        xarr.quantile(perc_min/100, dim=('x', 'y'), skipna=True),
        xarr.quantile(perc_max/100, dim=('x', 'y'), skipna=True),
        xarr.quantile(0.1/100, dim=('x', 'y'), skipna=True),
        xarr.quantile(99.9/100, dim=('x', 'y'), skipna=True))

    dump, band_std_qmin, band_std_qmax = standardize_xarray(
        xarr,  mean_inp=band_mean, std_inp=band_std,
        get_perc_qmin_qmax=[perc_min, perc_max])

    dump, band_std_qmin1, band_std_qmax1 = standardize_xarray(
        xarr,  mean_inp=band_mean, std_inp=band_std,
        get_perc_qmin_qmax=[0.1, 99.9])

    out = [band_min, band_max, band_mean, band_std,
           band_qmin, band_qmax, band_qmin1, band_qmax1,
           band_std_qmin, band_std_qmax, band_std_qmin1, band_std_qmax1]
    col_title = ['min', 'max', 'mean', 'std',
                 'perc_' + str(perc_min), 'perc_' + str(perc_max),
                 'perc_0.1', 'perc_99.9',
                'std_perc_' + str(perc_min), 'std_perc_' + str(perc_max),
                'std_perc_0.1', 'std_perc_99.9']
    out_name = [prefix + x for x in col_title]
    return list(zip(out_name, out))


def standardize_xarray(
        xarr, mean_inp=None, std_inp=None, get_perc_qmin_qmax=None):
    """
    Standardizes an xarray Dataset by subtracting the mean and dividing
    by the standard deviation, optionally scaling to specified percentiles.

    !!! INPUT DATASETS NEED TO BE FLOATS FOR INT ZEROS WOULDN'T BE
    SKIPPED !!!

    Parameters
    ----------
    xarr : xarray.DataArray
        The input DataArray to be standardized.
    mean_inp : float, optional
        mean to be used for standardisation.
        If not provided, it will be computed.
    std_inp : float, optional
        standard deviation used for standardisation.
        If not provided, it will be computed.
    get_perc_qmin_qmax : list of float, optional
        Min max percentile bounds, which should be extracted on the
        standardised array e.g. [0.5, 99.5] to calculate the 0.5th and
        99.5th percentile of the standardised array. The values can be
        used e.g. for later normalisation

    Returns
    -------
    xarray.DataArray
        The standardized DataArray.
    float
        The low qauntile value after scaling, if `get_perc_qmin_qmax`
        is provided.
    float
        The high quantile value after scaling, if `get_perc_qmin_qmax`
        is provided.

    Notes
    -----
    - The input dataset must be of float type to avoid issues with integer
       zeros being skipped.
    - If `get_perc_qmin_qmax` is provided, the function will calculate
      the specified percentiles after standardisation.
    - If the DataArray is chunked, the function will attempt to rechunk
       it to for the prcentile calculation.
    """

    if mean_inp is None or std_inp is None:
        std_inp, mean_inp = dask.compute(
            xarr.std(dim=('x', 'y'), skipna=True),
            xarr.mean(dim=('x', 'y'), skipna=True))

    xarr_std = ((xarr - mean_inp)/std_inp) #   .compute()

    std_qmin, std_qmax = None, None
    if get_perc_qmin_qmax is not None:
        try:
            std_qmin, std_qmax = dask.compute(
                xarr_std.quantile(
                    get_perc_qmin_qmax[0]/100, dim=('x', 'y'),
                    skipna=True),
                xarr_std.quantile(
                    get_perc_qmin_qmax[1]/100, dim=('x', 'y'),
                    skipna=True))
        except:
            # if thee are chunk
            xarr_std_rech = xarr_std.chunk(dict(x=-1, y=-1))
            std_qmin, std_qmax = dask.compute(
                xarr_std_rech.quantile(
                    get_perc_qmin_qmax[0]/100, dim=('x', 'y'),
                    skipna=True),
                xarr_std_rech.quantile(
                    get_perc_qmin_qmax[1]/100, dim=('x', 'y'),
                    skipna=True))

    return xarr_std, std_qmin, std_qmax


def normalize_xarray_perc_clip(
    xarr, perc_min=0.5, perc_max=99.5, perc_min_inp=None,
    perc_max_inp=None, norm_clip=True):
    """
    Normalize an xarray dataset by clipping values based on percentiles.

    !!! INPUT DATASETS NEED TO BE FLOATS FOR INT ZEROS WOULDN'T BE SKIPPED !!!

    Parameters
    ----------
    xarr : xarray.DataArray
        The input DataArray to normalize.
    perc_min : float, optional
        The lower percentile to use for normalization. Default is 0.5.
        (only used if perc_min_inp and perc_max_inp are None)
    perc_max : float, optional
        The upper percentile to use for normalization. Default is 99.5.
        (only used if perc_min_inp and perc_max_inp are None)
    perc_min_inp : float, optional
        The precomputed lower percentile value. It is calculated if None.
    perc_max_inp : float, optional
        The precomputed upper percentile value. It is calculated if None.
    norm_clip : bool, optional
        Whether to clip the normalized values between 0 and 1. Default is True.

    Returns
    -------
    xarray.DataArray
        The normalized dataset optionally with the values clipped based
        on the specified percentiles.

    Notes
    -----
    - The input dataset needs to be in floating-point format to correctly handle integer zeros.
    - If `perc_min_inp` or `perc_max_inp` is provided, `xarr`
        must be an xarray with band as a coordinate and shape [n_band].
        If the band names of perc_min_inp are correct then the order of
        bands is not important (values will be assigned correctly)
        (this can be iportant since if xarray was created from a pndas
        dataframe then the band order might have been ordered in a different
        way (alphabetically))
    - create xarray e.g. with
        xarr_stats = read_stats_to_xarray(path, file_prefix, bands_ordered_lst=None)
        xarr_stats['perc_99-5'].sel(name=img_key, drop=True)

        """

    if perc_min_inp is None or perc_max_inp is None:
        perc_min_inp, perc_max_inp = dask.compute(
            xarr.quantile(perc_min/100, dim=('x', 'y'), skipna=True),
            xarr.quantile(perc_max/100, dim=('x', 'y'), skipna=True))

    xarr_norm = ((xarr - perc_min_inp)/(perc_max_inp-perc_min_inp))  # .compute()
    if norm_clip:
        xarr_norm = xarr_norm.clip(min=0, max=1)

    return xarr_norm


def normalize_xarray(array_inp, min_norm=None, max_norm=None):
    '''
    Normalize to 0 - 1 range

    Parameters:
        array_inp: numpy array or rio data array
        (e.g. single band with rio_array.sel(band='band_name').values)
        but can also use full array rio_array.data

    Returns:
        numpy array
    '''
    if min_norm is None:
        min_norm = np.nanmin(array_inp)
    if max_norm is None:
        max_norm = np.nanmax(array_inp)

    array_range = max_norm - min_norm
    out = (array_inp - min_norm) / array_range
    geo_utils.get_add_img_attrs(array_inp, out)
    return out


def standardize_norm(img, mean_inp=None, std_inp=None, sym_min_max=None,
                     if_norm=True, get_perc_qmin_qmax=None, norm_clip=True):
    '''
    !!! INPUT DATASETS NEED TO BE FLOATS.
    FOR INT ZEROS WOULDN'T BE SKIPPED !!!
    '''

    if sym_min_max is not None or not if_norm:
        get_perc_qmin_qmax = None
    elif get_perc_qmin_qmax is None:
        get_perc_qmin_qmax = [0.5, 99.5]

    stats_test_std, std_min, std_max = standardize_xarray(
        img,  mean_inp=mean_inp, std_inp=std_inp,
        get_perc_qmin_qmax=get_perc_qmin_qmax)

    if if_norm:
        if sym_min_max is None:
            sym_min_max = np.max(np.abs([std_min, std_max]))

        stats_test_std = normalize_xarray_perc_clip(
            stats_test_std, perc_min_inp=-sym_min_max,
            perc_max_inp=sym_min_max, norm_clip=norm_clip)

    # add image attributes at
    geo_utils.get_add_img_attrs(img, stats_test_std)

    return stats_test_std
