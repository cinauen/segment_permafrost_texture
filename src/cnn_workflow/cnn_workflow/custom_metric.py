

"""
Functions for metric logging

"""
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

import ignite
import torchmetrics

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import utils.plotting_utils as plotting_utils
import utils.metrics_plotting_utils as metrics_plotting_utils


class MetricMonitor:
    '''
    Source https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/

    added n_batch
    '''
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(
            lambda: {"val": 0, "sum": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n_batch=1):
        metric = self.metrics[metric_name]

        metric["val"] = val
        metric["sum"] += val * n_batch
        metric["count"] += n_batch
        metric["avg"] = metric["sum"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


class ignMetric:
    '''
    metrics are setup in train or test loop
    metrics updated and reset within and afer each epoch
    (separately for train and validate)
    '''
    def __init__(self, n_classes, class_labels, device, ignore_index):
        self.n_classes = n_classes
        self.device = device
        self.class_labels = class_labels
        self.ignore_index = ignore_index

    def setup_ign_metrics(self):
        self.ign_acc_all = ignite.metrics.Accuracy()
        self.ign_cm = ignite.metrics.ConfusionMatrix(
            num_classes=self.n_classes, device=self.device)

        # for following confusion matrix needs to be initialized independently!!!
        # otherwise sampes are accumulated too many times....
        self.ign_mIoU = ignite.metrics.mIoU(
            ignite.metrics.ConfusionMatrix(num_classes=self.n_classes),
            ignore_index=self.ignore_index)
        self.ign_mIoU_all = ignite.metrics.mIoU(
            ignite.metrics.ConfusionMatrix(num_classes=self.n_classes))

        # !!!! class IoU includes also nan class, thus there might be
        # issues if NaN area was wrongly classified (is the case if
        # weight down nan class)
        self.ign_IoU_all = ignite.metrics.IoU(
            ignite.metrics.ConfusionMatrix(num_classes=self.n_classes))
        self.ign_dice_all = ignite.metrics.DiceCoefficient(
            ignite.metrics.ConfusionMatrix(num_classes=self.n_classes))

        self.ign_metrics_title = (
            ['acc_all', 'mIoU', 'mIoU_all']
            + ['IoU_class' + '{0:02d}'.format(x) + '_' + self.class_labels[1][x]
               for x in range(self.n_classes)]
            + ['dice_class' + '{0:02d}'.format(x) + '_' + self.class_labels[1][x]
               for x in range(self.n_classes)])

        return

    def update_ign_metrics(self, outputs, y):
        # done in set
        self.ign_acc_all.update((outputs, y))
        self.ign_cm.update((outputs, y))
        self.ign_mIoU.update((outputs, y))
        self.ign_mIoU_all.update((outputs, y))
        self.ign_IoU_all.update((outputs, y))
        self.ign_dice_all.update((outputs, y))

        return

    def compute_ign_metrics(self, count, epoch):
        '''
        count can be epoch (training pahse) or dataset number (testing phase)
        '''
        # update all metrics (epoch average per dataset)
        acc_ign_all = self.ign_acc_all.compute()
        cm_ign = self.ign_cm.compute()

        # !!! with ignite mIoU is still influenced by NaN class if
        # have nan area has false label predition !!!
        ign_mIoU = self.ign_mIoU.compute()
        ign_mIoU_all = self.ign_mIoU_all.compute()
        IoU_ign_all = self.ign_IoU_all.compute()
        dice_ign_all = self.ign_dice_all.compute()

        # merges all metrics into one row dataframe (is merged later outside)
        df_metric = pd.DataFrame(
            [[acc_ign_all, ign_mIoU.item(), ign_mIoU_all.item()]],
             columns=self.ign_metrics_title[:-self.n_classes*2],
             index=[[epoch], [count]])
        df_metric.index.names = ['epoch', 'count']

        df_class_metric = pd.DataFrame(
            [IoU_ign_all.numpy().tolist()],
             columns=self.ign_metrics_title[-self.n_classes*2:-self.n_classes],
             index=[[epoch], [count]])
        df_class_metric.index.names = ['epoch', 'count']

        df_class_dice_metric = pd.DataFrame(
            [dice_ign_all.numpy().tolist()],
             columns=self.ign_metrics_title[-self.n_classes:],
             index=[[epoch], [count]])
        df_class_dice_metric.index.names = ['epoch', 'count']

        cm_index = pd.MultiIndex.from_product(
            [[epoch], [count], self.class_labels[1]],
             names=['epoch', 'count', 'class_name'])
        df_cm = pd.DataFrame(
            cm_ign.cpu().numpy(), index=cm_index,
            columns=self.class_labels[1])

        return df_metric, df_class_metric, df_class_dice_metric, df_cm

    def reset_ign_metrics(self):
        self.ign_acc_all.reset()
        self.ign_cm.reset()
        self.ign_mIoU.reset()
        self.ign_mIoU_all.reset()
        self.ign_IoU_all.reset()
        self.ign_dice_all.reset()
        return


class tmMetric:
    '''
    metrics are setup in trai nor test loop
    metrics updated and reset within and afer each epoch
    (separately for train and validate)
    '''
    def __init__(self, n_classes, class_labels, device, ignore_index):
        self.n_classes = n_classes
        self.device = device
        self.class_labels = class_labels
        self.ignore_index = ignore_index

    def setup_tm_metrics(self):
        ''''''
        self.tm_metrics_title = (
            ['acc_all', 'acc', 'acc_macro', 'acc_w',
             'dice', 'dice_macro', 'f1_macro',
             'precision_macro', 'recall_macro',
             'jacc', 'jacc_macro']
            + ['jacc_class' + '{0:02d}'.format(x) + '_' + self.class_labels[1][x]
               for x in range(self.n_classes)]
            + ['dice_class' + '{0:02d}'.format(x) + '_' + self.class_labels[1][x]
               for x in range(self.n_classes)])

        # accuracies
        self.tm_acc_all = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes).to(self.device)
        self.tm_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='micro').to(self.device)
        self.tm_acc_macro = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='macro').to(self.device)
        self.tm_acc_w = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='weighted').to(self.device)

        # over all classes
        self.tm_dice = torchmetrics.Dice(
            num_classes=self.n_classes, ignore_index=self.ignore_index,
            average='micro').to(self.device)

        # Calculate the metric for each class separately, and average
        # the metrics across classes equal weights
        self.tm_dice_macro = torchmetrics.Dice(
            num_classes=self.n_classes, ignore_index=self.ignore_index,
            average='macro').to(self.device)
        self.tm_dice_class = torchmetrics.F1Score(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average=None).to(self.device)

        self.tm_f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='macro').to(self.device)

        self.tm_precision_macro = torchmetrics.Precision(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='macro').to(self.device)
        self.tm_recall_macro = torchmetrics.Recall(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='macro').to(self.device)

        # same as IoU
        self.tm_jacc = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='micro').to(self.device)
        self.tm_jacc_macro = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average='macro').to(self.device)
        self.tm_jacc_class = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.n_classes,
            ignore_index=self.ignore_index, average=None).to(self.device)

        # as comparison use torcmetrics
        self.tm_cm = torchmetrics.ConfusionMatrix(
            num_classes=self.n_classes, task='multiclass',
            ignore_index=self.ignore_index).to(self.device)

        return

    def update_tm_metrics(self, outputs, y):
        # done in set
        self.tm_acc_all(outputs, y)
        self.tm_acc(outputs, y)
        self.tm_acc_macro(outputs, y)
        self.tm_acc_w(outputs, y)

        self.tm_dice(outputs, y)
        self.tm_dice_macro(outputs, y)
        self.tm_f1_macro(outputs, y)
        self.tm_precision_macro(outputs, y)
        self.tm_recall_macro(outputs, y)
        self.tm_jacc(outputs, y)
        self.tm_jacc_macro(outputs, y)

        self.tm_jacc_class(outputs, y)
        self.tm_dice_class(outputs, y)
        self.tm_cm(outputs, y)

        return

    def compute_tm_metrics(self, count, epoch):
        '''
        count can be epoch (training phase) or dataset number (testing phase)
        during training phase count and epoch are the same
        however during trainig phase can use count for batch set counds
        (will be cummulative) and epoch for the epoch of the choosen
        model
        '''
        tm_acc_all = self.tm_acc_all.compute()
        tm_acc = self.tm_acc.compute()
        tm_acc_macro = self.tm_acc_macro.compute()
        tm_acc_w = self.tm_acc_w.compute()

        tm_dice = self.tm_dice.compute()
        tm_dice_macro = self.tm_dice_macro.compute()
        tm_f1_macro = self.tm_f1_macro.compute()
        tm_precision_macro = self.tm_precision_macro.compute()
        tm_recall_macro = self.tm_recall_macro.compute()
        tm_jacc = self.tm_jacc.compute()
        tm_jacc_macro = self.tm_jacc_macro.compute()

        tm_jacc_class = self.tm_jacc_class.compute()
        tm_dice_class = self.tm_dice_class.compute()
        tm_cm = self.tm_cm.compute()

        # maybe test adding .item()
        df_tm_metric = pd.DataFrame(
            [torch.stack([tm_acc_all, tm_acc, tm_acc_macro, tm_acc_w,
                          tm_dice, tm_dice_macro,
                          tm_f1_macro, tm_precision_macro, tm_recall_macro,
                          tm_jacc, tm_jacc_macro]).tolist()],
              columns=self.tm_metrics_title[:-self.n_classes*2],
              index=[[epoch], [count]])
        df_tm_metric.index.names = ['epoch', 'count']

        df_class_tm_metric = pd.DataFrame(
            [tm_jacc_class.tolist()],
             columns=self.tm_metrics_title[-self.n_classes*2:-self.n_classes],
             index=[[epoch], [count]])
        df_class_tm_metric.index.names = ['epoch', 'count']

        df_class_dice_tm_metric = pd.DataFrame(
            [tm_dice_class.tolist()],
             columns=self.tm_metrics_title[-self.n_classes:],
             index=[[epoch], [count]])
        df_class_dice_tm_metric.index.names = ['epoch', 'count']

        cm_index = pd.MultiIndex.from_product(
            [[epoch], [count], self.class_labels[1]],
             names=['epoch', 'count', 'class_name'])
        df_tm_cm = pd.DataFrame(
            tm_cm.tolist(), index=cm_index,
            columns=self.class_labels[1])

        return df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm

    def reset_tm_metrics(self):
        self.tm_acc_all.reset()
        self.tm_acc.reset()
        self.tm_acc_macro.reset()
        self.tm_acc_w.reset()

        self.tm_dice.reset()
        self.tm_dice_macro.reset()
        self.tm_f1_macro.reset()
        self.tm_precision_macro.reset()
        self.tm_recall_macro.reset()
        self.tm_jacc.reset()
        self.tm_jacc_macro.reset()

        self.tm_jacc_class.reset()
        self.tm_dice_class.reset()
        self.tm_cm.reset()
        return


class MetricSummary:
    '''
    This class manages summaries and plotting
    It has to be initialized within train or test and separately for
    ignite (ign) or torchmetrics (tm) metrics
    '''
    def __init__(self, phase_lst, out_path, prefix):
        # for train_run, phase list is
        # phase_lst = ['train', 'validate']
        self.phase_lst = phase_lst
        self.out_path = out_path
        self.prefix = prefix
        self.initialize_metric_summaries()


    def initialize_metric_summaries(self):
        '''
        initializes dicts for metrics
        '''
        if '_'.join(self.phase_lst).find('test') == -1:
            self.summary_loss = {x: [] for x in self.phase_lst}
        self.summary = {x: [] for x in self.phase_lst}
        self.summary_class = {x: [] for x in self.phase_lst}
        self.summary_class_dice = {x: [] for x in self.phase_lst}
        self.summary_cm = {x: [] for x in self.phase_lst}
        return

    def merge_metric_summaries(self, dict_inp=None):
        #df_inp must be dict with Dataframe
        # merge metrics
        summary_df = {}
        summary_class_df = {}
        summary_class_dice_df = {}
        for i in self.phase_lst:
            summary_df[i] = pd.concat(self.summary[i], axis=0)
            # add loss
            if i.find('test') == -1:
                summary_df[i].loc[:, 'loss'] = self.summary_loss[i]

            summary_class_df[i] = pd.concat(self.summary_class[i], axis=0)
            summary_class_dice_df[i] = pd.concat(self.summary_class_dice[i], axis=0)

        self.summary_df = pd.concat(summary_df, axis=1)
        # merge other metrics e.g. lr (learning rate)
        if dict_inp is not None:
            for i_key, i_val in dict_inp.items():
                self.summary_df['add', i_key] = i_val

        self.summary_class_df = pd.concat(summary_class_df, axis=1)
        self.summary_class_dice_df = pd.concat(summary_class_dice_df, axis=1)

        # merge confusion matrix
        summary_cm_df = {}

        for i in self.phase_lst:
            summary_cm_df[i] = pd.concat(
                self.summary_cm[i], axis=0)
        self.summary_cm_df = pd.concat(summary_cm_df, axis=1)
        return

    def update_metric_summaries(
            self, df_metric, df_class_metric, df_class_dice_metric,
            df_cm, phase, epoch_loss=None):
        self.summary[phase].append(df_metric)
        self.summary_class[phase].append(df_class_metric)
        self.summary_class_dice[phase].append(df_class_dice_metric)
        self.summary_cm[phase].append(df_cm)

        if '_'.join(self.phase_lst).find('test') == -1:
            self.summary_loss[phase].append(epoch_loss)
        return

    def save_metric_summaries(self, suffix):
        ''''''
        path_file = os.path.join(
            self.out_path,
            self.prefix + '_summary_' + suffix + '.txt')
        self.summary_df.to_csv(
            path_file, sep='\t', lineterminator='\n', header=True)

        path_file = os.path.join(
            self.out_path,
            self.prefix + '_summary_cm_' + suffix + '.txt')
        self.summary_cm_df.to_csv(
            path_file, sep='\t', lineterminator='\n', header=True)

        path_file = os.path.join(
            self.out_path,
            self.prefix + '_summary_class_IoU_' + suffix + '.txt')
        self.summary_class_df.to_csv(
            path_file, sep='\t', lineterminator='\n', header=True)

        path_file = os.path.join(
            self.out_path,
            self.prefix + '_summary_class_dice_' + suffix + '.txt')
        self.summary_class_dice_df.to_csv(
            path_file, sep='\t', lineterminator='\n', header=True)

        return

    def plot_cm(self, phase=None, epoch=False):
        '''
        Plot confusion matrices at specified epoch.
        The matrix is plotted at absolute values and as normalized over
        true and normalized over predicted
        '''
        # plot confusion matrix
        df_plot = self.summary_cm_df.loc[epoch, phase].reset_index('count', drop=True).astype(float)
        df_plot_n_true = metrics_plotting_utils.get_cm_ratio(df_plot, axis=1)
        df_plot_n_pred = metrics_plotting_utils.get_cm_ratio(df_plot, axis=0)

        out_name = (
            f"{self.prefix}_confusion_m_{phase}_ep{epoch:02d}.png")

        title1 = 'counts, epoch: ' + str(epoch)
        title2 = 'normalized over true, epoch: ' + str(epoch)
        title3 = 'normalized over predicted, epoch: ' + str(epoch)
        metrics_plotting_utils.plot_cm_matrix(
            df_plot, df_plot_n_true, df_plot_n_pred, self.out_path,
            out_name, title1, title2, title3)
        return

    def plot_learning_curve(self, checkp):
        ''''''
        metr_plot = ['loss', 'acc', 'jacc', 'jacc_macro']
        color = plotting_utils.cmap_to_hex(
            'plasma', num_bin=len(metr_plot)*2)
        with plt.style.context(['fast', plotting_utils.get_font_dict()]):
            fig, ax = plt.subplots(1, 1)

            self.summary_df.loc(axis=1)[['train', 'validate'], metr_plot].swaplevel(axis=1).sort_index(axis=1).plot(
                ax=ax, color=color)

            ax_r = ax.twinx()
            self.summary_df.loc[:, ('add','lr')].plot(
                ax=ax_r, color='k', linestyle='dashed')
            ax.scatter(
                checkp, self.summary_df.loc[checkp, ('validate', 'loss')],
                marker='x', c='red', label='ckeckp')
            ax.grid(True)
            ax.set_ylim([0, 1.5])
            ax_r.set_ylim([0, 0.01])
            ax_r.set_ylabel('learning rate')
            ax.set_ylabel('Loss/Metric')
            ax.set_xlabel('Epoch')
            out_name = (self.prefix + '_loss_acc.pdf')
            fig.savefig(os.path.join(self.out_path, out_name),
                        format='pdf')
            plt.close()
        return

    def plot_class_IoU(self, checkp):

        color = plotting_utils.cmap_to_hex(
            'viridis', num_bin=self.summary_class_df.loc[:, ('train', slice(None))].shape[1])
        with plt.style.context(['fast', plotting_utils.get_font_dict()]):
            fig, ax = plt.subplots(2, 1, figsize=(11.7, 8.3))
            self.summary_class_df.loc[:, ('train', slice(None))].plot(
                ax=ax[0], color=color, legend=False)
            ax[0].vlines(checkp, 0, 1, colors='red',
                         linestyles='dashed')
            ax[0].set_title('train')
            ax[0].grid(True)
            ax[0].set_ylabel('Jacc')
            ax[0].set_xlabel('Epoch')
            self.summary_class_df.loc[:, ('validate', slice(None))].plot(
                ax=ax[1], color=color)
            ax[1].vlines(checkp, 0, 1, colors='red',
                         linestyles='dashed')
            ax[1].set_title('validate')
            ax[1].legend(bbox_to_anchor=(1.0, 1.0))
            ax[1].grid(True)
            ax[1].set_ylabel('Jacc')
            ax[1].set_xlabel('Epoch')
            fig.tight_layout()
            out_name = (self.prefix + '_class_IoU.pdf')
            fig.savefig(os.path.join(self.out_path, out_name),
                        format='pdf')
            plt.close()
        return


class runningScore(object):
    '''
    Source https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/metrics.py
    '''
    def __init__(self, n_classes, label_min=0):
        self.n_classes = n_classes
        self.label_min = label_min
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= self.label_min) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        # sum up all labels (pred or gt)
        # flatten: flannten matrix to vector

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum() # overall accuracy
        # this correcsponds to (predb.argmax(dim=1) == yb.to(device)).float().mean()

        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        # accuracy per class: (correct per class / sum per class)
        # then take mean (classes not present in image are not taken into account)

        # -- mean intersection over Union --
        # Mean Intersection over Union (IoU) is the area of overlap
        # between the predicted segmentation and the ground truth divided
        # by the area of union between the predicted segmentation and the ground truth.
        # The mean IoU of the image is calculated by taking the IoU of
        # each class and averaging them.
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)

        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value
    Source https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/metrics.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PlotPred:
    def __init__(self, n_batch, n_classes, class_labels,
                 phase,
                 figsize, fontsize, max_subp=5):
        self.n_batch = n_batch
        self.n_batch_plot = min(max_subp, n_batch)
        self.n_classes = n_classes
        self.class_labels = class_labels
        self.phase = phase

        self.figsize = figsize
        self.fontsize = fontsize

    def initialize_plot_pred(self):

        grid = [self.n_batch_plot, 4]
        count_p = np.ravel(np.arange(grid[0]*grid[1]).reshape(
            grid[1], grid[0]).T)
        share_axis = None
        self.ax_pred, self.fig_pred, self.plot_nr_pred = plotting_utils.initialize_plot_custom(
            grid, [1]*self.n_batch_plot, [1]*(grid[1] - 1) + [0.05], count_p,
            figsize=(8.27, 11.69), e_left=0.05, e_right=0.85,
            fontsize=self.fontsize-2, share_axis=share_axis,
            e_wspace=0.01)
        return

    def get_title_pred(self, metric_show, epoch):
        '''
        get metric_show with
        df_metric = self.monitor_tm.summary[self.phase][self.epoch]
        df_metric.loc[self.epoch, 'acc']
        '''
        text_title = '{} epoch {}, Acc: {:.4f} | Jacc_macro: {:.4f}'.format(
            self.phase, epoch, metric_show['acc'], metric_show['jacc_macro'])
        return text_title

    def plot_pred_per_batch(self, x, y, pred_seg,
                            inp_img_title='grey scale img'):
        '''
        get pred_seg e.g.with
        pred_seg = self.outputs.argmax(dim=1).cpu()
        '''
        # loop thorugh batches for plotting
        n_batch_curr = min(x.shape[0], self.n_batch_plot)

        for i in range(n_batch_curr):
            # input grey image
            plot0 = self.ax_pred[self.plot_nr_pred[i,0]].imshow(
                x[i, 0, :, :].cpu(), cmap='Greys_r')
            # ground truth
            plot1 = self.ax_pred[self.plot_nr_pred[i,1]].imshow(
                y[i].cpu(), clim=(0, self.n_classes), cmap='viridis')
            # predicted
            plot2 = self.ax_pred[self.plot_nr_pred[i,2]].imshow(
                pred_seg[i, :, :], clim=(0, self.n_classes),
                cmap='viridis')

            cbar = self.fig_pred.colorbar(
                plot2, cax=self.ax_pred[self.plot_nr_pred[i, 3]])
            cbar.set_ticks(
                self.class_labels[0], labels=self.class_labels[1],
                fontsize=self.fontsize-2)

        # make axis etc invisible for empty plots (in last batch
        # is smaller that n_batches)
        self.ax_pred[self.plot_nr_pred[0,0]].set_title(inp_img_title)
        self.ax_pred[self.plot_nr_pred[0,1]].set_title('ground truth')
        self.ax_pred[self.plot_nr_pred[0,2]].set_title('predicted')

        diff_b = self.n_batch_plot - n_batch_curr
        for i_del in range(diff_b):
            for i_subp in range(4):
                plotting_utils.make_patch_spines_invisible(
                    self.ax_pred[self.plot_nr_pred[-i_del, i_subp]], 1)
        return

    def save_plot_pred(
            self, file_prefix, file_path, text_title):

        self.fig_pred.suptitle(text_title, fontsize=self.fontsize)
        self.fig_pred.savefig(
            os.path.join(file_path, file_prefix + '_pred.pdf'),
            format='pdf')
        plt.close()
        return
