"""
Functions to create segmentation metrics plots

"""

import os
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import utils.plotting_utils as plotting_utils


def plot_cm_matrix(df_plot, df_plot_n_true, df_plot_n_pred,
                   path, file_name, title1, title2, title3):
    '''
    plot confusion matrix
    '''
    with plt.style.context(['fast', plotting_utils.get_font_dict()]):
        #plotting_utils.set_global_plot_fontsize(
        #    SMALL_SIZE=12, MEDIUM_SIZE=14, BIGGER_SIZE=16)
        fig, ax = plt.subplots(1, 3, figsize=(25, 8))
        sn.heatmap(
            df_plot,
            annot=True, ax=ax[0], annot_kws={"size":14}, fmt=".0f",
            cmap="Blues", norm=colors.LogNorm(), cbar_kws={"shrink": 0.6})
        sn.heatmap(
            df_plot_n_true,
            annot=True, ax=ax[1], annot_kws={"size":14}, fmt=".2f",
            cmap="Blues", cbar_kws={"shrink": 0.6})
        sn.heatmap(
            df_plot_n_pred,
            annot=True, ax=ax[2], annot_kws={"size":14}, fmt=".2f",
            cmap="Blues", cbar_kws={"shrink": 0.6})
        for e, i in enumerate([title1, title2, title3]):
            ax[e].set_title(i, fontsize=12)
            ax[e].tick_params(axis='y', labelsize=str(12), rotation=0)
            ax[e].tick_params(axis='x', labelsize=str(12), rotation=90)
            ax[e].set_aspect('equal')
            ax[e].set_ylabel('true')
            ax[e].set_xlabel('predicted')

        fig.tight_layout()
        fig.savefig(os.path.join(path, file_name),
                    format=file_name.split('.')[-1])
        plt.close()


def get_cm_ratio(df_plot, axis=1):
    '''
    input must be pandas df

    axis 1 is normalized over the true conditions
    axis 0 is normalized over the predicted conditions
    '''
    return (df_plot/np.expand_dims(np.sum(df_plot, axis=axis), axis=axis)).fillna(0)


def plot_loss_matrix(
        seg_inp, weight_inp, loss_inp, file_path, file_prefix,
        n_classes, max_subp=5):
    n_batch_curr = min(loss_inp.shape[0], max_subp)

    plot_seg = seg_inp.cpu().detach().numpy()
    plot_weight = weight_inp.cpu().detach().numpy()
    plot_loss = loss_inp.cpu().detach().numpy()

    fig, ax = plt.subplots(
        nrows=4, ncols=n_batch_curr,
        figsize=(11.69, 11.69))
    if len(ax.shape) < 2:
        ax = ax[:, np.newaxis]

    for i in range(n_batch_curr):
        plot = ax[0, i].imshow(plot_seg[i, :, :], cmap='viridis',
                               clim=(0, n_classes))
        fig.colorbar(plot, ax=ax[0, i])

        plot = ax[1, i].imshow(plot_weight[i, :, :], cmap='plasma',
                               clim=(0, 1))
        fig.colorbar(plot, ax=ax[1, i])

        plot = ax[2, i].imshow(plot_loss[i, :, :], cmap='viridis')
        fig.colorbar(plot, ax=ax[2, i])

        plot = ax[3, i].imshow(
            plot_loss[i, :, :], cmap='viridis',
            norm=colors.LogNorm())
        fig.colorbar(plot, ax=ax[3, i])
    ax[0, 0].set_title('classes')
    ax[1, 0].set_title('weights')
    ax[2, 0].set_title('loss')
    ax[3, 0].set_title('log loss')

    fig.tight_layout()
    fig.savefig(
        os.path.join(file_path, file_prefix + '_loss_matrix.pdf'),
        format='pdf')
    plt.close()

    return
