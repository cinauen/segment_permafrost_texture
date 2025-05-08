'''
========== Plot image properties =========

Script to compare properties of the test patches per site.
It plots the following:
- intensity distribution: histogram, median, mean, standard deviation
- image properties: local standard deviation,
    shannon entropy (for uncertainty/variability)

!!! The INPUT has been adjusted to match the data available in the
!!! repository (SPOT is not available due to data sharing restrictions)

'''
import os
import sys
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.ndimage import generic_filter
from skimage.metrics import structural_similarity as ssim

import numpy as np
from skimage.measure import shannon_entropy
from scipy.ndimage import uniform_filter

plt.rcdefaults() # restore [global] defaults
matplotlib.rcParams['pdf.fonttype'] = 42

# ----------------- PATHS & GLOBAL PARAM -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, MODULE_PATH)
sys.path.insert(0, BASE_PATH)

# -------------- import specific utils --------------
import utils.geo_utils as geo_utils
import utils.conversion_utils as conversion_utils
import utils.plotting_utils as plotting_utils

# ===================== INPUT =====
# -------- read test patches from all sites: --------
site_order = [
    'BLyaE',
    #'BLyaS',
    'FadN'
    ]
sites_folders = {
    'BLyaE': 'BLyaE_v1',
    #'BLyaS': 'BLyaS_v1',
    'FadN': 'FadN_v1',
    }

test_folders = {
    'BLyaE': {
        'HEX1979': 'BLyaE_HEX1979_test_perc0-2_g0-3_8bit_Lv01_untiled_v00',
        #'SPOT2018': 'BLyaE_SPOT2018_test_std4_8bit_Lv01_untiled_v00',
        },
    #'BLyaS': {
    #    'HEX1979': 'BLyaS_HEX1979_test_perc0-2_g0-3_8bit_Lv01_untiled_v00',
    #    'SPOT2019': 'BLyaS_SPOT2019_test_std4_8bit_Lv01_untiled_v00',
    #    },
    'FadN': {
        'HEX1980': 'FadN_HEX1980_test_histm_b0-5aRv1_8bit_Lv01_untiled_v00',
        #'SPOT2018': 'FadN_SPOT2018_test_std3_8bit_Lv01_untiled_v00',
        }
    }

# --------- min expected imag size of test patches ---------
x_exp = 333
y_exp = 333
font_size = 8

test_patches = {
    'BLyaE': [f'test-{x:02d}' for x in range(1, 5)],
    #'BLyaE': [f'test-{x:02d}' for x in range(1, 7)],
    #'BLyaS': [f'test-{x:02d}' for x in range(1, 8)],
    'FadN': [f'test-{x:02d}' for x in range(1, 5)],
    #'FadN': [f'test-{x:02d}' for x in range(1, 9)],
    }

sensor_lst = ['HEX']  # !!!! SPOT is not available in exmaple repo ['HEX', 'SPOT']

# ----------------------- define paths -----------------------
# set project paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH_INP = os.path.join(path_proc_inp, '1_site_preproc')
PATH_EXPORT = PATH_INP

# -------- statistics to be plotted ---------------
plot_groups = [
    ['mean', 'median'], ['std'], ['local_std'], ['entrophy'],
    #  ['mean_similarity'],
    ]
n_stats = len(plot_groups)


# =========== read data, extract statistic and create plot =========
# ---- import all images
df_dict_site = {}
img_col_lst = []
img_arr_lst = []
for i_site, i_folder_dict in test_folders.items():
    df_dict_sensor = {}
    for i_sensor, i_folder in i_folder_dict.items():
        path_inp = os.path.join(
            PATH_INP, sites_folders[i_site], '03_train_inp', i_folder)
        df_dict_test = {}
        for i_test in test_patches[i_site]:
            f_test = f"{i_site}_{i_sensor}_{i_test}_data.tif"
            test_img = geo_utils.read_to_xarray(
                os.path.join(path_inp, f_test),
                mask_nan=True, chunk=None)
            img_arr_lst.append(test_img.sel(band=1).values.squeeze())
            img_col_lst.append(
                tuple([i_site, i_sensor[:-4], i_sensor, i_test]))

            df_dict_test[i_test], df_crs, df_cols = conversion_utils.img_to_df_proc(
                test_img)
        df_dict_sensor[i_sensor] = pd.concat(
            df_dict_test, axis=0, names=['patch'])
    df_dict_site[i_site] = pd.concat(
        df_dict_sensor, names=['sensor'])
df = pd.concat(
    df_dict_site, axis=0, names=['site']).dropna(axis=0, how='any')
df.drop(['x', 'y'], axis=1, inplace=True)
df.rename({1: 'val'}, axis=1, inplace=True)
df.reset_index(inplace=True)
df['sensor_type'] = df['sensor'].apply(lambda x: x[:-4])
df.set_index(['site', 'sensor_type', 'sensor', 'patch', 'id'], inplace=True)

BLyaE_HEX_ind = [
    e for e, x in enumerate(img_col_lst) if 'BLyaE' in x and 'HEX' in x]
BLyaE_SPOT_ind = [
    e for e, x in enumerate(img_col_lst) if 'BLyaE' in x and 'SPOT' in x]

# ----------- initialize plot --------------
fig = plt.figure(figsize=(11.7, 5))
gs0 = gridspec.GridSpec(
    2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1],
    wspace=0.2, hspace=0.5)

# subplot 0 to 2
ax = []
for i in range(2):
    gs00 = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=gs0[0, i], wspace=0.2)
    ax.append(fig.add_subplot(gs00[:]))

for i in range(2):
    gs01 = gridspec.GridSpecFromSubplotSpec(
        1, n_stats, subplot_spec=gs0[1, i], wspace=0.5)
    ax.extend([fig.add_subplot(gs01[0, x]) for x in range(n_stats)])


stats_index = pd.MultiIndex.from_tuples(
    img_col_lst, names=['site', 'sensor_type', 'sensor', 'patch'])

# format axes
plotting_utils.format_axes_general(
    ax, list(range(len(ax))),
    dict(labeltop=False, labelbottom=True,
        labelleft=True, labelright=False, labelsize=font_size,
        left=True, bottom=True, right=False, top=False),
        ax_excl_lst=['top', 'right'])

fig_settings = dict(
    alpha=0.5,
    linewidth=0.0,
    fill=True,
    stat='count',  # show the number of observations in each bin
    multiple="dodge",
    kde=True,
    bins='auto',
    element='step'
    )

# ------------ plot histogram (per site) -------------
sns.histplot(
    data=df.query('sensor_type == "HEX"'), x='val', hue="site",
    hue_order=site_order,
    palette='viridis',
    ax=ax[0], legend=True, zorder=1,
    **fig_settings)
ax[0].grid(True, axis='both', lw=0.25, zorder=1.0)

if "SPOT" in sensor_lst:
    sns.histplot(
        data=df.query('sensor_type == "SPOT"'), x='val', hue="site",
        hue_order=site_order,
        palette='viridis',
        ax=ax[1], legend=True, zorder=1,
        **fig_settings)
    print('t')
    ax[1].grid(True, axis='both', lw=0.25, zorder=1.0)


groupby_param = ['site', 'sensor_type', 'sensor', 'patch']

BLyaE_ind = {}
BLyaE_ind['HEX'] = [
    e for e, x in enumerate(img_col_lst)
    if 'BLyaE' in x and 'HEX' in x]
if "SPOT" in sensor_lst:
    BLyaE_ind['SPOT'] = [
        e for e, x in enumerate(img_col_lst)
        if 'BLyaE' in x and 'SPOT' in x]


def describe_image(df_inp, groupby_param, img_arr_lst, img_col_lst,
                   stats_index):
    ''' Function to extract image stats '''
    stats_lst = []

    df_grouped = df_inp.groupby(groupby_param)
    # -- Brightness
    # mean
    mean_val = df_grouped.mean()
    stats_lst.append(mean_val.rename({'val': 'mean'}, axis=1))
    # median
    stats_lst.append(
        df_grouped.median().rename({'val': 'median'}, axis=1))
    # Contrast
    stats_lst.append(
        df_grouped.std().rename({'val': 'std'}, axis=1))

    df_stats = pd.DataFrame(
        np.ones([len(img_col_lst), 2])*np.nan,
        columns=['entrophy', 'speckle_proxy'], index=stats_index)

    for i_col, i_img in zip(img_col_lst, img_arr_lst):
        df_stats.loc[i_col, 'entrophy'] = shannon_entropy(i_img)

        # Speckle proxy (local std dev)
        local_std = uniform_filter(i_img**2, size=5) - uniform_filter(i_img, size=5)**2
        df_stats.loc[i_col, 'speckle_proxy'] = np.mean(local_std)

        df_stats.loc[i_col, 'local_std'] = generic_filter(i_img, np.std, size=3).mean()

        sim_lst = []
        for i_t in BLyaE_ind[i_col[1]]:
            sim_ind = ssim(img_arr_lst[i_t][:y_exp, :x_exp],
                                i_img[:y_exp, :x_exp], multichannel=False,
                                data_range=255, win_size=51)
            if i_col != img_col_lst[i_t]:
                sim_lst.append(sim_ind)
        df_stats.loc[i_col, 'mean_similarity'] = np.mean(sim_lst)

    df_stats = pd.concat([df_stats] + stats_lst, axis=1)

    return df_stats

# ------------------- get image stats ------------------
df_stats = describe_image(df, groupby_param, img_arr_lst, img_col_lst,
                          stats_index)

df_stats.columns.name = 'stats'

stats_min = df_stats.min(axis=0)
stats_max = df_stats.max(axis=0)


# ------------------- plot stats (per site) ------------------
for e, i in enumerate(sensor_lst):
    df_stats_plt = df_stats.query('sensor_type == @i').stack('stats').to_frame(name='val').reset_index()
    df_stats_plt['group_index'] = df_stats_plt.loc[:, groupby_param].apply(lambda x: ':'.join(x), axis=1)
    for e_stats, i_stats in enumerate(plot_groups):
        legend_plt = True if e_stats == (n_stats - 1) else False
        df_stats_plt_val = df_stats_plt.query('stats in @i_stats')
        ax_num = 2 + e_stats + n_stats*e
        sns.stripplot(
            data=df_stats_plt_val,
            x="stats", y="val", hue="site", ax=ax[ax_num],
            palette='viridis',
            dodge=True, alpha=.4, legend=False,
        )
        sns.pointplot(
            data=df_stats_plt_val,
            x="stats", y="val", hue="site", ax=ax[ax_num],
            dodge=.4, linestyle="none", errorbar=None,
            palette='viridis',
            marker="_", markersize=20, markeredgewidth=1,
            legend=legend_plt,
        )

        ax[ax_num].set_ylim(
            stats_min.loc[i_stats].min()*0.95,
            stats_max.loc[i_stats].max()*1.05)

        ax[ax_num].grid(True, axis='both', lw=0.25, zorder=1.0)

# --------------- save plot ------------
f_path = os.path.join(PATH_EXPORT,
                      'image_similarity_test_patches_v1.pdf')
fig.savefig(f_path, format='pdf')

print('END')