"""
Functions to convert in between xarray/rioxarray images DataFrames on
CPU (pandas) or GPU (cudf)
"""

import numpy as np
import xarray



def df_to_img(df, EPSG_target, attrs=None,
              nodata=np.nan):
    ''''''
    img = xarray.Dataset.from_dataframe(df).to_array(
        dim='band').sortby('y', ascending=False)

    epsg = 'epsg:' + str(int(EPSG_target))
    img.rio.write_crs(epsg, inplace=True)

    if attrs is not None:
        img.attrs.update(attrs)
        img.attrs.update(
            {'orig_type': str(img.data.dtype)})
    img.rio.write_nodata(nodata, inplace=True)

    return img


def img_to_df_proc(inp_img, rename_dict=None, use_GPU=False):
    '''
    create cudf DataFrame from xarray
    e.g. as input for cluster calculation

    previousely called: xarray_to_cudf

    !!! use_GPU is here set to False per defauly this is because
        for ml_workflow False is here required for preprocessing
    '''
    df = xarray_to_df(inp_img)
    if rename_dict is not None:
        df = df.rename(columns=rename_dict)
    if use_GPU:
        df = df_to_cudf(df)
    gdf_out = df.reset_index()
    gdf_out.index.name = 'id'
    gdf_out.columns.name = None
    crs_img = inp_img.rio.crs
    feature_cols = df.columns.tolist()
    del df

    return gdf_out, crs_img, feature_cols


def df_to_cudf(df):
    import cudf
    gdf = cudf.DataFrame.from_pandas(df)
    return gdf


def xarray_to_df(img, drop_col='spatial_ref',
                 bands=None, stack_dim='band'):
    ''' create cudf dataframe from xarray'''

    if bands is not None:
        img = img.sel(band=bands)

    df = img.to_dataset(dim=stack_dim).to_dataframe()
    if drop_col is not None:
        df.drop(drop_col, axis=1, inplace=True)

    # unstack is very slow. with to_dataframe can avoid it
    #df = df.unstack(level=0)
    #if drop_main_level:
    #    df = df.droplevel(0, axis=1)
    return df