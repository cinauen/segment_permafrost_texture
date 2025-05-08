"""
Functions for principal component analysis (PCA)
Allows usage of GPUs
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import xarray
import pandas as pd


# ---- set use_GPU according to if GPU modules are available
# could also check with torch if GPU ids available (if torch installed):
# torch.cuda.is_available()
import importlib.util
module_search = importlib.util.find_spec("cudf")
if module_search is not None:
    use_GPU = True
else:
    use_GPU = False

import utils.conversion_utils as conversion_utils


def get_PCA(img_std, file_path=None, file_prefix='', col_use=None,
            n_components=None, whiten=False, bands=None):
    '''
    img_std is standardized image

    whiten: If True, de-correlates the components. This is done by
    dividing them by the corresponding singular values then multiplying
    by sqrt(n_samples). Whitening allows each component to have unit
    variance and removes multi-collinearity. It might be beneficial for
    downstream tasks like LinearRegression where correlated features
    cause problems.

    !!! NO standardization is done here !!!
    '''

    df = conversion_utils.xarray_to_df(img_std, bands=bands)
    if use_GPU:
        gdf = conversion_utils.df_to_cudf(df)
    else:
        gdf = df

    out, pca_out = PCA_decomposition(
        gdf, file_path=file_path, file_prefix=file_prefix, col_use=col_use,
        n_components=n_components, whiten=whiten)

    if use_GPU:
        df_out = out.to_pandas().set_index(['y', 'x'])
    else:
        df_out = out.set_index(['y', 'x'])
    df_out.rename(
        columns={x: 'P' + '{0:02d}'.format(x) for x in df_out.columns},
        inplace=True)

    # sort in order to keep x in descending order. This is important if
    # create numpy arrays for plotting with e.g. imshow (from_dataframe sorts
    # coordinates in ascending order)
    out_img = xarray.Dataset.from_dataframe(df_out).to_array(
        dim='band').sortby('y', ascending=False)

    attrs = img_std.attrs.copy()
    attrs['long_name'] = tuple(out_img.band.values)
    out_img.attrs.update(attrs)
    out_img = out_img.rio.write_crs(img_std.rio.crs)

    return out_img


def PCA_decomposition(
        gdf, file_path=None, file_prefix='', col_use=None, n_components=None,
        whiten=False):
    if use_GPU:
        import cudf
        from cuml import PCA
    else:
        from sklearn.decomposition import PCA
    '''
    Input is cudf DataFrame

    whiten:
    If True, de-correlates the components. This is done by dividing them
    by the corresponding singular values then multiplying by
    sqrt(n_samples). Whitening allows each component to have unit
    variance and removes multi-collinearity. It might be beneficial for
    downstream tasks like LinearRegression where correlated features
    cause problems.
    '''

    # need to drop inidces
    if col_use is not None:
        dgdf = gdf.loc[:, col_use].reset_index(drop=True).dropna(axis=0, how='any')
    else:
        dgdf = gdf.reset_index(drop=True).dropna(axis=0, how='any')
    pca_out = PCA(n_components=n_components, whiten=whiten)
    pca_out.fit(dgdf)

    trans_ggdf = pca_out.transform(dgdf)

    # pca_float.components_
    # pca_float.explained_variance_
    # pca_float.explained_variance_ratio_

    # check importance of each component
    if file_path is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(pca_out.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        file_name = file_prefix + '_PCA_explained_variance.pdf'
        plt.savefig(os.path.join(file_path, file_name), format='pdf')

    out = gdf.reset_index().loc[:, ['y', 'x']]
    if use_GPU:
        out = cudf.concat([out, trans_ggdf], axis=1)
    else:
        out = pd.concat(
            [out, pd.DataFrame(trans_ggdf,
                               index=dgdf.index.tolist())], axis=1)

    return out, pca_out

