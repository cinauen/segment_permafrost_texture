"""
Functions specifically for xarray DataArrays or Datasets

"""

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr


def merge_xarray_dict(da_dict, name_new_dim="regio", test_coords=True,
                      name_out='merged', concat_coords_inp='minimal'):
    '''
    merges dict to new dimension in xarray
    !!! But needs to have same coords. Otherwise this will give wrong coords !!!

    concat_coords_inp default is "different"

    '''
    if test_coords:
        check_if_coords_equal(da_dict)
    out = xr.concat(da_dict.values(),
                    dim=pd.Index(da_dict.keys(), name=name_new_dim),
                    coords=concat_coords_inp)
    if isinstance(out, xr.DataArray):
        out.name = name_out
    return out


def check_if_coords_equal(da_dict, coords_check=None):
    '''
    coord_lst is list of coords to compare
    e.g. ['month'] or ['latitude', 'longitude']
    '''
    keys_lst = list(da_dict.keys())
    for i_key, ii_key in zip(keys_lst[:-1], keys_lst[1:]):
        if coords_check is None:
            coords_check = da_dict[i_key].coords.dims
        for i in coords_check:
            if np.all(da_dict[i_key][i] == da_dict[ii_key][i]):
                print(i + ': ' + i_key + ' & ' + ii_key + ': are equal')
            else:
                print(i + ': ' + i_key + ' & ' + ii_key + ': are NOT the same !!!!!!')
    return


def relabel_xarray(arr_inp, relabel_dict):
    '''
    relabel dict is
    {old_val1: new_val1, old_val2: new_val2}
    # !!! seems always to be a copy!
    '''
    arr = arr_inp.copy()

    # create masks
    mask = {}
    # !!! mask is False where old value occurs
    # Thus values to keep are true
    for i_old, i_new in relabel_dict.items():
        mask[i_old] = np.where(arr==i_old, False, True)

    # exchange value
    for i_old, i_new in relabel_dict.items():
        arr = arr.where(mask[i_old], i_new)

    return arr
