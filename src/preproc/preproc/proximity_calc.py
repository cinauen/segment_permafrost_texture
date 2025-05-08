"""
Functions for proximity calculation and boundary weight calculation
using xarray-spatial

"""

import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
from xrspatial import proximity
from joblib import Parallel, delayed, cpu_count



def get_proximity(poly_inp, cvs):
    '''
    output line_proximity is xarray with dimension of cvs but no
    crs or other rio infos attached
    '''
    x, y = poly_inp.exterior.coords.xy

    df_shape = pd.DataFrame(
        {"x": x.tolist(),
         "y": y.tolist(),
        "id": list(range(len(x)))})
    line_agg = cvs.line(df_shape, x="x", y="y")
    line_proximity = proximity(line_agg)

    return line_proximity


def create_cvs(ref_img, px_size=1.5):
    '''
    create datashader canvas with shape corresponding to input base
    '''
    arr_shape = ref_img.shape
    arr_x_coord = ref_img.coords['x'].values
    arr_y_coord = ref_img.coords['y'].values

    cvs = ds.Canvas(
        plot_width=arr_shape[1], plot_height=arr_shape[1],
        x_range=(arr_x_coord.min() - px_size/2,
                 arr_x_coord.max() + px_size/2),
        y_range=(arr_y_coord.min() - px_size/2,
                 arr_y_coord.max() + px_size/2))

    return cvs


def get_proximity_from_gdf(poly_lst, ref_img, n_jobs=6, px_size=1.5):
    '''
    poly_lst is list with polygons from thich want to calculate proximity

    poly_lst = gdf.query('area > (1.5*1.5/1000000)*3 and raster_val == 1').geometry.tolist()
    '''
    n_jobs = min(int(cpu_count()/10), n_jobs)

    cvs = create_cvs(ref_img, px_size=px_size)

    w = Parallel(n_jobs=n_jobs, verbose=0)(delayed(
            get_proximity)(k, cvs) for k in poly_lst)

    return w


def merge_proximity_to_weight(dist_lst, ref_img, shift=0.5, half_rad=2.5):
    '''
    dist_lst is output from
    out = get_proximity_from_gdf(
        seg_arr, gdf.query('area > (1.5*1.5/1000000)*3 and raster_val == 1').geometry.tolist(),
        n_jobs=6, px_size=1.5)

    '''
    # keep min value
    xr_dist = xr.concat(dist_lst, dim='shape').min(dim='shape')

    # apply same function to all patches
    xr_dist_weight = 1 - np.exp(-(shift + xr_dist**2/half_rad**2))
    xr_dist_weight.rio.write_crs(ref_img.rio.crs, inplace=True)

    return xr_dist_weight

