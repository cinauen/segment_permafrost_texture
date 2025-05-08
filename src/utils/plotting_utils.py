"""
Functions and utilities for plotting: histograms, raster and
vector shape plots

"""

import os
import numpy as np
import holoviews as hv
from bokeh.io import export_svg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors
import matplotlib.ticker as ticker


def get_font_dict(SMALL_SIZE=10, MEDIUM_SIZE=12, BIGGER_SIZE=14):

    out = {'font.size': SMALL_SIZE,
           'axes.titlesize':SMALL_SIZE,
           'axes.labelsize':MEDIUM_SIZE,
           'xtick.labelsize':SMALL_SIZE,
           'ytick.labelsize':SMALL_SIZE,
           'legend.fontsize':SMALL_SIZE,
           'figure.titlesize':BIGGER_SIZE
           }

    return out


def plot_hist_from_img(img, path_out, file_name_out):
    '''img is xarray'''

    # extract data to be plotted from image
    df = img.to_dataframe(name='test')
    df = df.drop('spatial_ref', axis=1).dropna(
        how='all', axis=0).unstack('band')
    df = df.droplevel(0, axis=1)
    df_col = df.columns
    bands_plot = img.band.values.tolist()
    n_bands_plot = len(bands_plot)

    # plot
    with matplotlib.rc_context(
            {'axes.prop_cycle': plt.cycler(
             "color", plt.cm.gnuplot(np.linspace(0, 1, n_bands_plot)))}):
        fig, axs = plt.subplots(2, figsize=(12, 8))
        df.loc[:, df_col[:1]].plot.hist(bins=50, alpha=0.5, ax=axs[0],
                                        grid=True)
        if len(df_col) > 1:
            df.loc[:, df_col[1:]].plot.hist(
                bins=50, alpha=0.5, ax=axs[1], grid=True)
            axs[1].legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))
        fig.tight_layout()
        fig.savefig(os.path.join(path_out, file_name_out),
                    format=file_name_out.split('.')[-1])
        plt.close(fig)

    return


def plot_save_hist_np(arr, path_exp, name, fig_format='png'):
    import pandas as pd
    fig, axs = plt.subplots(2, figsize=(12, 8))
    pd.DataFrame(arr.reshape(-1, 1)).plot.hist(bins=50, ax=axs[0])
    fig.savefig(os.path.join(path_exp, name), format=fig_format)
    return


def plot_hist_match_curves(img, ref, matched, path_exp, fig_suffix,
                           fig_type='pdf', axes=None, file_prefix=''):
    '''
    img: is input image
    ref: is reference image to which input image should be matched
    matches: is hist matches image

    all inputs ar enumy arrays of format: [y, x, n_bands] or [x, n_bands]

    Source: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html
    '''
    from skimage import exposure
    n_bands = img.shape[-1]
    if axes is None:
        fig, axes = plt.subplots(nrows=n_bands, ncols=3, figsize=(9, n_bands*3))
        axes = np.atleast_2d(axes)
        fig_save = True
    else:
        fig_save = False

    x_min = np.floor(np.nanmin([np.nanmin(x) for x in [img, ref, matched]]))
    x_max = np.ceil(np.nanmax([np.nanmax(x) for x in [img, ref, matched]]))

    for i, img in enumerate((img, ref, matched)):
        for e_band in range(n_bands):
            img_no_nan = img[..., e_band][~np.isnan(img[..., e_band])]
            img_hist, bins = exposure.histogram(img_no_nan, source_range='dtype')
            axes[e_band, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img_no_nan)
            axes[e_band, i].plot(bins, img_cdf)
            axes[e_band, i].grid(True)
            axes[e_band, i].set_xlim([x_min, x_max])
            axes[e_band, 0].set_ylabel(f"{fig_suffix}\nband: {e_band}")

    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')

    if fig_save:
        file_name = f"{file_prefix}_{fig_suffix}_hist_plt.png"
        fig.tight_layout()
        fig.savefig(os.path.join(path_exp,
                                 f"{file_prefix}_{fig_suffix.split('.')[0]}.{fig_type}"),
                                 format=fig_type)
    return


def plot_hist(arr_inp, name, alpha=0.3, nodata=np.nan,
              fontsize=None, range_tuple=None, bin_num=50,
              width=None, height=None, show_grid=True):
    '''
    arr_inp is numpy array
    e.g. after ds.utils.orient_array(img.sel(band=bands[0]))
    or simply
    img.sel(band=bands[0]).values
    or
    normalize_intensity_corr(r, percent_min_max=(2, 98))

    overlay several histrograms with
    display((plot_hist(arr_inp_r, 'r') * plot_hist(arr_inp_g, 'g') * plot_hist(arr_inp_b, 'b')).opts(width=900, height=400))

    e.g. width=1200, height=500
    '''

    if fontsize is None:
        fontsize = {'title': 14, 'labels': 12,
                    'ticks': 12, 'legend': 12}

    vect = arr_inp.reshape(-1, 1)
    if np.isnan(nodata):
        vect = vect[~np.isnan(vect)]
    else:
        vect = vect[vect != nodata]

    # adjust plotting options
    ropts = dict(label=name)
    ropts1 = dict(alpha=alpha, fontsize=fontsize, show_grid=show_grid)
    if hv.Store.current_backend == 'matplotlib':
        ropts1.update(dict(linewidth=0, color='blue'))
        if width is not None and height is not None:
            ropts1.update(
                {'fig_size': 400, 'aspect':np.round(width/height, 0)})
    else:
        ropts1.update(dict(line_width=0))
        if width is not None and height is not None:
            ropts1.update({'width':width, 'height':height})

    if range_tuple is not None:
        ropts1.update(dict(xlim=range_tuple))

    return hv.Histogram((np.histogram(vect, bin_num)), **ropts).opts(
        **ropts1), ropts1


def save_img_hv(img_layout, PATH_EXPORT, file_name, file_type='png',
                dpi=300):
    '''

    pdf is only possible with matplotlib backend.....
    '''


    if file_type == 'png':
        hv.save(img_layout, os.path.join(PATH_EXPORT, file_name + '.png'), fmt='png', dpi=dpi)
    elif file_type == 'svg':
        plot_out = hv.render(img_layout, dpi=dpi)
        export_svg(plot_out, filename=os.path.join(PATH_EXPORT, file_name + '.svg'))
    elif file_type == 'pdf':
        # !!! only possible with matplotlib backend.
        try:
            hv.save(img_layout, os.path.join(PATH_EXPORT, file_name + '.pdf'), fmt='pdf', dpi=dpi)
        except:
            plot_out = hv.render(img_layout, dpi=dpi)
            export_svg(plot_out, filename=os.path.join(PATH_EXPORT, file_name + '.svg'))
    else:
        hv.save(img_layout, os.path.join(PATH_EXPORT, file_name + '.' + file_type),
                fmt=file_type, dpi=dpi)

    return


def overview_plot_texture(img_np, img_texture_np, title, file_name, path,
                  figsize=None, fontsize=10, min_perc=2, max_perc=98):
    if figsize is None:
        figsize = [11.7, 8.3]  # [wsize, hsize]

    fig, ax = plt.subplots(3, 3, figsize=figsize)

    count = 0
    for j_ax in  range(ax.shape[0]):
        for i_ax in range(ax.shape[1]):
            if count >= len(title):
                make_patch_spines_invisible(ax[j_ax, i_ax], 1)
                continue
            if i_ax == 0 and j_ax == 0:
                img_inp = img_np
                label = 'img'
                color_inp = 'gray'
            else:
                img_inp = img_texture_np[count, :, :]
                label = title[count]
                color_inp = 'viridis'
                count += 1
            lim_min, lim_max = find_limits(
                img_inp, low_perc=min_perc, high_perc=max_perc)

            fig_plot = ax[j_ax, i_ax].imshow(
                img_inp, vmin=lim_min, vmax=lim_max, cmap=color_inp)
            #ax[j_ax, i_ax].set_yticklabels(
            #    ax[j_ax, i_ax].get_yticklabels(), fontsize=fontsize)

            #set colorbar
            create_colorbar(
                fig, ax[j_ax, i_ax], fig_plot, label, fontsize=fontsize,
                size='2.0%', pad=0.4)

    # plt.tight_layout()

    out_path = os.path.join(path, file_name + '.pdf')
    fig.savefig(out_path, format='pdf')

    return


def plot_all_bands_mask(fig, ax, image, mask, bands, title_prefix,
                        cmap_bands=None, weights=None):
    # plot image
    n_bands = len(bands)
    if cmap_bands is None:
        cmap_bands = ['Greys_r'] + ['viridis']*(n_bands-1)
    for i_x in range(n_bands):
        img =  ax[i_x].imshow(image[i_x, :, :], cmap=cmap_bands[i_x])
        arr_min = np.nanmin(image[i_x, :, :].values)
        arr_max = np.nanmax(image[i_x, :, :].values)
        ax[i_x].set_title(
            f"{title_prefix} img, {str(bands[i_x])}\n min: {arr_min:.4f} | max: {arr_max:.4f}")
        fig.colorbar(img, ax=ax[i_x], spacing='proportional',
                     shrink=0.7)

    # plot segmentation mask
    img = ax[-1].imshow(mask)  #  interpolation="nearest"
    ax[-1].set_title(title_prefix + ' mask')
    fig.colorbar(img, ax=ax[-1], spacing='proportional',
                     shrink=0.7)
    if weights is not None:
        img = ax[-2].imshow(weights)  #  interpolation="nearest"
        ax[-2].set_title(title_prefix + ' weights')
        fig.colorbar(img, ax=ax[-2], spacing='proportional',
                     shrink=0.7)
    return


def make_patch_spines_invisible(ax, del_labels=0):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    if del_labels == 1:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)


def create_colorbar(fig, ax, p_ax, axis_label, fontsize=12, size='1.0%',
                    pad=0.2):
    '''
    ax: input axes to plot colorbar on
    (colorbar does not need separate subplot)

    '''

    divider = make_axes_locatable(ax)

    cax = divider.append_axes('right', size=size, pad=pad)
    make_patch_spines_invisible(cax, del_labels=1)
    #cax.set_yticklabels(cax.get_yticklabels(), fontsize=fontsize)
    cbar = fig.colorbar(p_ax, cax=cax, orientation='vertical', extend='both')
    #cax.set_yticklabels(cax.get_yticklabels(), fontsize=fontsize)
    cbar.set_label(axis_label, rotation=270, fontsize=fontsize,
                   labelpad=fontsize)
    return


def find_limits(np_array, low_perc=10, high_perc=90, scale=0.05,
                perc=True, equal=False, mid_val=0):
    lim_min = 0
    lim_max = 100

    if perc:
        lim_min = np.nanpercentile(np_array, low_perc)
        lim_max = np.nanpercentile(np_array, high_perc)

    if not perc or lim_min == lim_max:
        v_min = np.nanmin(np_array)
        v_max = np.nanmax(np_array)
        diff = abs(v_max - v_min)

        add_scale = diff*0.05
        lim_min = v_min - add_scale
        lim_max = v_max + add_scale

    if equal:
        ampl = np.nanmax([abs(lim_max - mid_val), abs(lim_min - mid_val)])
        lim_min = mid_val - ampl
        lim_max = mid_val + ampl

    return lim_min, lim_max


def get_quantile_range(img, perc_min=2, perc_max=98):
    clim = (float(img.quantile(perc_min/100)),
            float(img.quantile(perc_max/100)))
    return clim


def plot_xarray_imshow(img, path_export, file_n,
                       grid=None, fig_size=None, c_bar_fontsize=8,
                       file_type='pdf'):
    """
    Plots an xarray DataArray as images in a grid layout.

    Parameters
    ----------
    img : xarray.DataArray
        The input DataArray to plot.
    path_export : str
        The directory where the exported figure will be saved.
    file_n : str
        The file name for the exported figure.
    grid : list, optional
        The grid layout for the subplots [rows, columns],
        by default [1, img.shape[0]].
    fig_size : tuple, optional
        The size of the figure in inches [width, height],
        by default [11.7, 5.].
    c_bar_fontsize : int, optional
        The font size for the colorbar labels, by default 8.
    file_type : str, optional
        The file format for the exported figure, by default 'pdf'.

    Returns
    -------
    None
    """
    if grid is None:
        grid = [1, img.shape[0]]
    if fig_size is None:
        fig_size = (11.7, 5.)

    fig, ax = plt.subplots(grid[0], grid[1], sharex=True, sharey=True,
                           figsize=fig_size)
    for e, i_band in enumerate(img.band.values.tolist()):
        im_plot = ax[e].imshow(img.sel(band=i_band), aspect='equal')
        create_colorbar(
            fig, ax[e], im_plot, i_band, fontsize=c_bar_fontsize,
            size='2.0%', pad=0.4)

    fig.tight_layout()
    fig_path = os.path.join(
        path_export, f'{file_n.split(".")[0]}.{file_type}')
    fig.savefig(fig_path, format=file_type)
    # plt.close(fig)  # commented as gives tkinter error
    return


def initialize_plot_custom(grid, h_ratios, w_ratios, count=None, figsize=None,
                           l_top=False, l_bot=True, l_left=True, l_right=False,
                           e_left=0.1, e_right=0.95, e_bottom=0.15, e_top=0.95,
                           e_wspace=0.15, e_hspace=0.25, share_axis=None,
                           aspect_ratio='auto', fontsize=10):
    '''

    grid, h_ratios, w_ratios, count=None, margins=None,
                           spaces=None, figsize=None, l_top=False, l_bot=True,
                           l_left=True, l_right=False, share_axis=None)


    initialize histogram plots
        grid = [n, m]
        h_ratios = [n1, n2, n3, n4]
        w_ratios = [m1, m2, m3, m4]

        initialize_plot_custom('plot1', [3, 2], [0.5, 0.5, 0.5], [1, 0.2], [0, 1, 2])

        -- plot direction:
        normal plot direction with count is down and then to the side
            array([[0, 3],
                   [1, 4],
                   [2, 5]])
            thus default count with  array([0, 1, 2, 3, 4, 5]) can be used

        if want to have plotting in direction right and then down need to define count as:
            count = np.ravel(np.arange(grid[0]*grid[1]).reshape(grid[1], grid[0]).T)
            count = array([0, 3, 1, 4, 2, 5])

    '''
    if figsize is None:
        figsize = [11.7, 8.3]  # [wsize, hsize]
    # in inch Breite, Höhe 11.7 x 16.5 A4 8.3 x 11.7
    n_subp = grid[0]*grid[1]
    if count is None:
        count = list(range(n_subp))

    # convert l_top etc to list in case want different labeling for each subplot
    if  not isinstance(l_top, list):
        l_top = [l_top]*n_subp
    l_top_all = np.any(l_top)
    if not isinstance(l_bot, list):
        l_bot = [l_bot]*n_subp
    l_bot_all = np.any(l_bot)
    if not isinstance(l_left, list):
        l_left = [l_left]*n_subp
    l_left_all = np.any(l_left)
    if not isinstance(l_right, list):
        l_right = [l_right]*n_subp
    l_right_all = np.any(l_right)

    gs = []
    ax = []
    fig = []

    plot_numbering = np.arange(grid[0]*grid[1]).reshape(grid[0], grid[1])

    #define plot area
    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    gs = gridspec.GridSpec(grid[0], grid[1], left=e_left, right=e_right,
                           bottom=e_bottom, top=e_top, wspace=e_wspace,
                           hspace=e_hspace, height_ratios=h_ratios,
                           width_ratios=w_ratios)

    if not isinstance(share_axis, (list, np.ndarray)):
        share_axis = [share_axis]*len(count)

    for i in count:
        x = i%grid[0]
        y = int(np.floor(i/grid[0]))

        #testing
        if share_axis[i] == 2 and len(ax) > 0:
            ax.append(plt.subplot(gs[x, y], sharex=ax[-1], sharey=ax[-1]))
        elif share_axis[i] == 1 and len(ax)> 0:
            ax.append(plt.subplot(gs[x, y], sharex=ax[-1]))
        else:
            ax.append(plt.subplot(gs[x, y]))

        fig.add_subplot(ax[-1])

        ax[-1].set_aspect(aspect_ratio)
        ax[-1].tick_params(which='major', direction='out',
                           labelsize=str(fontsize),
                           labeltop=l_top[i], labelbottom=l_bot[i],
                           labelleft=l_left[i], labelright=l_right[i],
                           top=l_top_all, bottom=l_bot_all, left=l_left_all,
                           right=l_right_all)

    return ax, fig, plot_numbering


def cmap_to_hex(cmap, num_bin=None):
    '''
    num bin can be None if use discrete colorscale (e.g. tab10) which would give out fixed colors (for tab10 ten colors)
    For continuouse xcolorscales e.g. viridis, gan use num bin to define how many output colors want
    (by default varidis is forned by 256 colors ) can use cmap_plt.N to check

    to plot cororbar to check can use:
    tt = LinearSegmentedColormap.from_list('mycmap', colors=colors, N=num_bin)
    tt
    '''
    cmap_plt = plt.get_cmap(cmap)
    if num_bin is None:
        colors = cmap_plt.colors
    else:
        colors = cmap_plt(np.linspace(0, 1, num_bin))

    return [matplotlib.colors.to_hex(x) for x in colors]


def format_axes_general(axg, ax_lst, param_dict, ax_excl_lst=None):
    '''
    param_dict = dict(labeltop=False, labelbottom=True,
                      labelleft=True, labelright=False,
                      labelsize=font_size)
    '''
    param_inp = {'width': 0.4, 'length': 3, 'left': True, 'bottom': True}
    param_inp.update(param_dict)
    # for first plot in row add left labels
    if ax_excl_lst is None:
        ax_excl_lst = ['top']
    for i_ax in ax_lst:
        for axis in ['top','bottom','left','right']:
            if axis not in ax_excl_lst and axg[i_ax]._axes.name != 'polar':
                axg[i_ax].spines[axis].set_linewidth(0.4)
                axg[i_ax].spines[axis].set_color('k')
            elif axg[i_ax]._axes.name != 'polar':
                axg[i_ax].spines[axis].set_linewidth(0.0)
        axg[i_ax].tick_params(**param_inp, direction='out', pad=2.0)
    return


def adjust_tick_params(
        fig, ax_counts=None, ax_lw=0.5, pad=3.0, font_size=8, axis_col='k',
        tick_spacing_maj=None, tick_spacing_min=None,
        tick_spacing_majy=None, tick_spacing_miny=None,
        tick_length=3.0, label_kws=None):
    '''
    e.g. label_kws:
    dict(labelbottom=False, labeltop=False, labelleft=False, labelright=False,
         bottom=True, top=True, left=True, right=True)
    '''
    ax_counts = list(range(len(fig.axes))) if ax_counts is None else ax_counts
    for i_ax in ax_counts:
        # for first plot in row add left labels
        for axis in ['top','bottom','left','right']:
            fig.axes[i_ax].spines[axis].set_linewidth(ax_lw)
            if axis_col is not None:
                fig.axes[i_ax].spines[axis].set_color(axis_col)

        fig.axes[i_ax].tick_params(
            length=tick_length, direction='out', width=ax_lw, labelsize=font_size, pad=2.0,
            **label_kws)

        if tick_spacing_maj is not None:
            fig.axes[i_ax].xaxis.set_major_locator(
                ticker.MultipleLocator(tick_spacing_maj))
        if tick_spacing_min is not None:
            fig.axes[i_ax].xaxis.set_minor_locator(
                ticker.MultipleLocator(tick_spacing_min))
        if tick_spacing_majy is not None:
            fig.axes[i_ax].yaxis.set_major_locator(
                ticker.MultipleLocator(tick_spacing_majy))
        if tick_spacing_miny is not None:
            fig.axes[i_ax].yaxis.set_minor_locator(
                ticker.MultipleLocator(tick_spacing_miny))
    return


def plot_class_boundaries(gdf_inp, param_col_dict, ax_inp, col=None, linewidth=0.75):
    for i_class, i_lst in param_col_dict.items():
        if i_class not in gdf_inp.index:
            continue
        if col is None:
            line_col = i_lst[0]
        else:
            line_col = col
        gdf_inp.loc[[i_class], :].boundary.plot(
            ax=ax_inp, linewidth=linewidth, color=line_col, alpha=1.0)

