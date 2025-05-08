"""
Customised train, test and prediction classes

"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import time
import rasterio
from rasterio.merge import merge as rasterio_merge
import matplotlib
matplotlib.use('Agg')
import torch.optim.lr_scheduler as lr_scheduler
from joblib import Parallel, delayed, cpu_count

import gc

from tqdm import tqdm

# ----------------- import custom utils -----------------
import cnn_workflow.custom_metric as custom_metric
import cnn_workflow.custom_data_loader as custom_data_loader
import utils.geo_utils as geo_utils
import utils.metrics_plotting_utils as metrics_plotting_utils
import postproc.MAIN_extract_segmentation_properties as MAIN_extract_segmentation_properties
import postproc.MAIN_extract_TP_TN_per_class_uncert as MAIN_extract_TP_TN_per_class_uncert


class TrainRun(custom_metric.ignMetric, custom_metric.tmMetric,
               custom_metric.PlotPred):
    def __init__(self, model, train_loader, valid_loader,
                 criterion, optimizer, acc_fn,
                 n_batch, n_classes, n_epochs, device,
                 class_labels, out_path, prefix,
                 early_stopping=True,
                 update_best_model=True,
                 epoch_save_min=10,
                 patience=10, extended_output=False,
                 loss_weights=None, scheduler_lr=None,
                 ignore_index=-100,
                 loss_reduction='mean',
                 weight_calc_type='norm_weight', max_subp=5):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.criterion = criterion  # loss function
        self.optimizer = optimizer
        self.acc_fn = acc_fn

        self.loss_weights = loss_weights  # if this is 9999 then
        # weights are adapted for each batch loop
        self.loss_reduction = loss_reduction

        self.weight_calc_type = weight_calc_type  # how sample weights are
        # included in loss calculation

        self.n_batch = n_batch
        self.n_batch_plot = min(max_subp, n_batch)
        self.n_classes = n_classes
        self.n_epochs = n_epochs

        self.device = device

        self.class_labels = class_labels
        self.prefix = prefix
        self.out_path = out_path

        self.extended_output = extended_output
        # extended output allows outputs more files fo debugging
        # saves metrics summary in each epoch
        # creates validation and confusion matrix plots in each poch

        self.save_metric = 'jacc_macro'  # which metric to use to
        # check save and early stopping criterion
        self.early_stopping = early_stopping
        self.patience = patience  # for early stopping
        # how many bad iterations to wait until stop training
        self.update_best_model = update_best_model  # if update self.best_model to
        # then later use it for prediction

        # only start saving epochs after this value
        self.epoch_save_min = min(epoch_save_min, self.n_epochs)

        self.ignore_index = ignore_index

        #self.initialize_summaries()
        self.monitor_ign = custom_metric.MetricSummary(
            ['train', 'validate'], out_path, prefix)
        self.monitor_tm = custom_metric.MetricSummary(
            ['train', 'validate'], out_path, prefix)

        self.fontsize = 10
        self.figsize = [9, 13]  # [8.3, 11.7]

        # initialize list with saved models sublist [epoch, save_path, best_acc]
        self.model_save_lst = []
        if scheduler_lr is None:
            self.scheduler_lr = []
        else:
            self.scheduler_lr = scheduler_lr

        self.epoch_init = 0  # normally start wirh zero
        # if load pretrained checkpoint then this could be adjusted

        return

    def run_train(self):
        ''' run full training run'''

        start = time.time()
        self.train_loss, self.valid_loss = [], []
        self.lr = []

        self.criterion_save_stop = EarlyStopping_acc_save(
            epoch_min=self.epoch_save_min, patience=self.patience,
            last_epoch=self.n_epochs - 1)  # minus one since start counting at 0

        self.setup_epoch_metrics()

        # loop over epochs
        for self.epoch in range(self.epoch_init, self.n_epochs):
            # first train
            self.train()
            self.validate()

            # update scheudler
            # get current lr
            self.lr.append(self.optimizer.param_groups[-1]['lr'])
            for i_s in self.scheduler_lr:
                # adjust lr for next step
                # or get learning rate with i_s.get_last_lr()
                i_s.step()

            self.best_acc = self.criterion_save_stop.check(
                self.monitor_tm.summary[self.phase][self.epoch].loc[(self.epoch, self.epoch), self.save_metric],
                self.epoch)
            if self.extended_output:
                self.monitor_ign.merge_metric_summaries()
                self.monitor_tm.merge_metric_summaries({'lr': self.lr})
                self.monitor_ign.save_metric_summaries(
                    'trainRun_ign')
                self.monitor_tm.save_metric_summaries(
                    'trainRun_tm')
                self.monitor_tm.plot_learning_curve(
                    self.criterion_save_stop.checkpoints)
                self.monitor_tm.plot_class_IoU(
                    self.criterion_save_stop.checkpoints)

            if self.criterion_save_stop.save:
                self.monitor_tm.plot_cm(
                    phase='validate', epoch=self.epoch)
                self.save_model_state_dict()
            if self.criterion_save_stop.early_stop and self.early_stopping:
                break


        self.save_model_summary()
        self.monitor_ign.merge_metric_summaries()
        #lr_index = pd.MultiIndex.from_product(
        #    [[self.epoch], [self.epoch]], names=['epoch', 'count'])
        self.monitor_tm.merge_metric_summaries({'lr': self.lr})
        self.monitor_ign.save_metric_summaries('trainRun_ign')
        self.monitor_tm.save_metric_summaries('trainRun_tm')
        self.monitor_tm.plot_learning_curve(
            self.criterion_save_stop.checkpoints)
        self.monitor_tm.plot_class_IoU(
            self.criterion_save_stop.checkpoints)
        for i_epoch in self.criterion_save_stop.checkpoints[-5:]:
            # plot confution matrix
            self.monitor_tm.plot_cm(
                phase='validate', epoch=i_epoch)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        return

    def train(self):
        '''
        train_loader: train_dl
        criterion: loss_fn (e.g. torch.nn.CrossEntropyLoss())

        1) loop through all image batches from data loader
        2) predict model output
        3) calculate error (loss)
        4) calculate gradients (backpropagation)
        5) update model params
        '''
        self.model.train()
        self.phase = 'train'
        self.dataloader = self.train_loader
        self.stream = tqdm(self.dataloader)

        # -- iterate over data batches
        for i, (x, y, sample_w) in enumerate(self.stream, start=1):
            # x is data input [n_batch, n_channel, height, width]
            x = x.to(self.device, non_blocking=True)
            self.n_batch_curr = x.shape[0]

            # y is segmentation mask [n_batch, height, width]
            y = y.to(self.device, non_blocking=True)
            # calculate model output
            self.outputs = self.model(x)

            # -- optionally adjust class weights according to class occurence in batch
            if self.loss_weights == [9999]:
                # get 1/counts per class for weighting
                ww = y.unique(sorted=True, return_counts=True)
                # the following is added to accommodate if one class is
                # missing in y
                ww_0 = ww[0].tolist()  # unique class number
                ww_1 = ww[1].tolist()  # amount occurences per class
                # List of weights. Missing classes have weight 0
                weight_lst = [
                    1/ww_1[ww_0.index(x)] if x in ww_0 else 0
                    for x in self.class_labels[0]]

                # set weights of indices to be ignored to 0
                weight_lst[self.criterion.ignore_index] = 0

                # assign batch specific class weights to criterion
                self.criterion.weight = torch.FloatTensor(
                    weight_lst).to(self.device)

            # ---- calculate loss
            # Loss will be multiplied by number of batches and divided
            # by amount of images. Thus, get average metric per image
            # per epoch.
            self.loss = self.criterion(self.outputs, y)

            # ----- compare different loss options:
            # - with weights
            # l_weighted = torch.nn.CrossEntropyLoss(
            #    weight=self.criterion.weight, ignore_index=0,
            #    reduction='none')
            # - without weights
            # l_unweighted = torch.nn.CrossEntropyLoss(
            #    ignore_index=0, reduction='none')
            # loss1 = l_weighted(self.outputs, y)
            # loss2 = l_unweighted(self.outputs, y)
            # loss3 = loss2*self.criterion.weight[y]
            # torch.all(loss1 == loss3) is True
            # --> THUS for the case without reduction (='none') class
            #     weights are just multiplied to loss

            # - test option with reduction mean (default)
            # l_weighted_red_mean = torch.nn.CrossEntropyLoss(
            #   weight=self.criterion.weight, ignore_index=0)
            # loss1w = l_weighted_red_mean(self.outputs, y)

            # - loss1w with reduction mean is same as reduction none summed and
            # divided by weight sum
            # l_weighted_red_none = torch.nn.CrossEntropyLoss(
            #   weight=self.criterion.weight,
            #   ignore_index=0, reduction='none')
            # loss2w = l_weighted_red_none(self.outputs, y)

            # l_unweighted_red_none = torch.nn.CrossEntropyLoss(
            #   ignore_index=0, reduction='none')
            # loss_un = l_unweighted_red_none(self.outputs, y)
            # loss2uw = loss_un * self.criterion.weight[y]
            # --> torch.all(loss2w == loss2uw) is True

            # loss3w = loss2uw.sum()/self.criterion.weight[y].sum()
            # loss3w = (loss2uw/self.criterion.weight[y].sum()).sum()
            # --> loss3w == loss1w
            # Thus reduction mean calcuates weighted (weighted with weights)
            # mean and not full mean as loss2w.mean()

            # ---- adapt loss in for loss reduction = 'sum' or in case
            # if want to include sample (uncertainty) weights
            if self.loss_reduction == 'none':
                # ----- option to add sample (uncertainty) weights to loss
                if sample_w.shape[-1] > 0:
                    sample_w = sample_w.to(self.device, non_blocking=True)
                    if self.weight_calc_type == 'norm_weight':
                        # -- option pytorch discussion
                        loss_w = (self.loss/self.criterion.weight[y].sum() * sample_w)
                        self.loss = loss_w.sum()
                        # check
                        # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
                        # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/16
                    else:
                        # -- option as in paper with sample weights (Bressan et al 2022)
                        # "average_weight"
                        loss_w = (self.loss * sample_w)
                        self.loss = loss_w.mean()

                else:
                    # normaly this should not be called if run from
                    # MAIN file since there, the loss_reduction is
                    # set to mean if sample weight are not availale
                    # (input labels y_img having only 1 band)

                    # this is same as if would have used loss_reduction = mean
                    loss_w = self.loss/self.criterion.weight[y].sum()
                    self.loss = loss_w.sum()
            elif self.loss_reduction == 'sum':
                # --- adapt class weights to be true mean not weighted mean
                # but true mean is calculated by counting only non zero weights
                self.loss = self.loss/(torch.count_nonzero(self.criterion.weight[y]))

            # --- zero the gradients
            self.optimizer.zero_grad()
            # to save memory could use
            # optimizer.zero_grad(set_to_none=True) and move this to beginning of loop
            # https://discuss.pytorch.org/t/the-location-of-zero-grad-at-the-training-loop/160206

            # --- perform backpropagation (compute gradients)
            # the backward pass frees the graph memory, so there is no
            # need for torch.no_grad in this training pass
            self.loss.backward()

            # ---- updated model parameters
            self.optimizer.step()

            # ---- update metric monitor
            self.update_metrics_on_set(y)

        # --- update epoch metrics
        self.update_reset_metrics_on_epoch()

        if self.loss_reduction == 'none' and self.extended_output:
            # -- plot loss in case of sample weighting
            file_prefix = (self.prefix + '_' + self.phase
                     + '_ep' + '{0:02d}'.format(self.epoch))
            metrics_plotting_utils.plot_loss_matrix(
                y, sample_w, loss_w, self.out_path, file_prefix,
                self.n_classes)
        return

    def validate(self):

        self.model.eval()
        self.phase = 'validate'
        self.dataloader = self.valid_loader
        self.stream = tqdm(self.dataloader)

        # .no_grad is to disable gradient calculation (to save memory)
        with torch.no_grad():
            # -- iterate over data batches
            for i, (x, y, sample_w) in enumerate(self.stream, start=1):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                self.n_batch_curr = x.shape[0]

                self.outputs = self.model(x)

                # -- optionally adjust class weights according to class occurence in batch
                if self.loss_weights == [9999]:
                    # get 1/counts per class for weighting
                    ww = y.unique(sorted=True, return_counts=True)
                    # following is added to accommodate if one class is
                    # missing in y
                    ww_0 = ww[0].tolist()  # unique class number
                    ww_1 = ww[1].tolist()  # amount occurences per class
                    # List of weights. Missing classes have weight 0
                    weight_lst = [
                        1/ww_1[ww_0.index(x)] if x in ww_0 else 0
                        for x in self.class_labels[0]]

                    # set weights of indices to be ignored to 0
                    weight_lst[self.criterion.ignore_index] = 0

                    # assign batch specific class weights to criterion
                    self.criterion.weight = torch.FloatTensor(
                        weight_lst).to(self.device)

                # ---- calculate loss
                self.loss = self.criterion(self.outputs, y)

                # ---- adapt loss in for loss reduction = 'sum' or in case
                # if want to include sample (uncertainty) weights
                if self.loss_reduction == 'none':
                    if sample_w.shape[-1] > 0:
                        sample_w = sample_w.to(self.device, non_blocking=True)
                        if self.weight_calc_type == 'norm_weight':
                            # -- option pytorch discussion
                            loss_w = (self.loss/self.criterion.weight[y].sum() * sample_w)
                            self.loss = loss_w.sum()
                            # check
                            # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
                            # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/16
                        else:
                            # -- option as in paper with sample weights
                            # (Bressan et al 2022), "average_weight"
                            loss_w = (self.loss * sample_w)
                            self.loss = loss_w.mean()
                    else:
                        # normaly this should not be called if run from
                        # MAIN file since there, the loss_reduction is
                        # set to mean if sample weight are not availale
                        # (input labels y_img having only 1 band)

                        # this is same as if would have used loss_reduction = mean
                        loss_w = self.loss/self.criterion.weight[y].sum()
                        self.loss = loss_w.sum()
                elif self.loss_reduction == 'sum':
                    # --- adapt class weights to be true mean not weighted mean
                    # but true mean is calculated by counting only non zero weights
                    self.loss = self.loss/(torch.count_nonzero(self.criterion.weight[y]))

                # ---- update metric monitor
                self.update_metrics_on_set(y)

            # --- update epoch metrics
            self.update_reset_metrics_on_epoch()

            # here use batch set
            if self.extended_output:
                # plot predictions
                self.initialize_plot_pred()
                pred_seg = self.outputs.argmax(dim=1).cpu()
                self.plot_pred_per_batch(x, y, pred_seg)
                metric_show = self.monitor_tm.summary[self.phase][-1].loc[(self.epoch, self.epoch), :]
                text_title = self.get_title_pred(metric_show, self.epoch)
                file_prefix = (self.prefix + '_' + self.phase
                     + '_ep' + '{0:02d}'.format(self.epoch))
                self.save_plot_pred(file_prefix, self.out_path,
                                    text_title)
        return

    def setup_epoch_metrics(self):
        '''
        epoch metrics are reset after each epoch and phase
        '''
        # metric monitor used for running_loss
        self.metric_monitor = custom_metric.MetricMonitor()

        self.setup_ign_metrics()
        self.setup_tm_metrics()

        return

    def update_metrics_on_set(self, y):
        '''
        up data metrics in dataset loop (sums up and divides per count)
        https://pytorch.org/ignite/metrics.html
        '''

        # get predicted classes
        # model predb = model(x) has dimension:
        # pred are probabilities of shape
        # [n_batch, n_channel(=n_classes), height, width]
        # argmax Returns the indices of the maximum value of all elements
        # in the input tensor.
        pred = self.outputs.argmax(dim=1)
        acc_all, acc = acc_metric(pred, y)

        self.update_ign_metrics(self.outputs, y)
        self.update_tm_metrics(self.outputs, y)
        # update metrics per dataset (sums up)

        # here sum up and multyply by batch size to be able to
        # provide image average per epoch
        # is done by metric monitor
        self.metric_monitor.update(
            "Loss", self.loss.item(), self.n_batch_curr)
        self.metric_monitor.update('acc_all', acc_all, self.n_batch_curr)
        self.metric_monitor.update('acc', acc, self.n_batch_curr)

        self.stream.set_description(
            "Epoch: {epoch} / {n_epochs}. {phase}. Metrics: {metric_monitor}. Tm Jacc: {jacc_macro}. AllocMem (Mb): {memory_allocated}".format(
            epoch=self.epoch, n_epochs=self.n_epochs - 1, phase=self.phase,
            metric_monitor=self.metric_monitor,
            jacc_macro='{0:.4f}'.format(self.tm_jacc_macro.compute()),
            memory_allocated=round(torch.cuda.memory_allocated(self.device)/1024/1024, 3))
            )#self.save_metric

    def update_reset_metrics_on_epoch(self):

        epoch_loss = self.metric_monitor.metrics['Loss']['avg']

        # update all metrics (epoch average per dataset)
        df_metric, df_class_metric, df_class_dice_metric, df_cm = self.compute_ign_metrics(
            self.epoch, self.epoch)
        # !!! for tm mean IoU does not take nan class into account since used
        # ignore_index=0 (for ignite still is influenced even if use
        # ignore_index)!!!
        df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm = self.compute_tm_metrics(
            self.epoch, self.epoch)

        self.monitor_ign.update_metric_summaries(
            df_metric, df_class_metric, df_class_dice_metric, df_cm, self.phase, epoch_loss)
        self.monitor_tm.update_metric_summaries(
            df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm, self.phase,
            epoch_loss)

        # Reseting metrics such that ready for new epoch
        self.metric_monitor.reset()

        # reset all metrics
        self.reset_ign_metrics()
        self.reset_tm_metrics()

        return

    def save_model_state_dict(
            self, model_only=False):
        save_path = os.path.join(
            self.out_path,
            self.prefix + '_model_ep' + '{0:02d}'.format(self.epoch))  # .pkl

        if not model_only:
            model_state = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
                'best_iou': str(self.best_acc),
                'curr_iou': str(self.criterion_save_stop.acc_curr)}
            save_path += '.tar'

            torch.save(model_state, save_path)
        else:
            save_path += '.pth'
            torch.save(self.model.state_dict(), save_path)

        if self.update_best_model:
            # need to use deepcopy since is only view/reference
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.epoch

        self.model_save_lst.append(
            [self.epoch, save_path, self.best_acc,
             self.criterion_save_stop.acc_curr,
             self.criterion_save_stop.if_checkp])
        return

    def save_model_summary(self):
        self.model_save_df = pd.DataFrame(
            self.model_save_lst,
            columns=['epoch', 'path', 'best_acc', 'curr_add', 'if_checkp'])
        self.model_save_df.set_index('epoch', inplace=True)

        path_file = os.path.join(
            self.out_path, self.prefix + '_model_save_df.txt')
        self.model_save_df.to_csv(path_file, '\t', lineterminator='\n',
                                  header=True)
        return


class EarlyStopping_acc_save():
    '''
    Adjusted according to:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    '''
    def __init__(self, patience=10, min_delta=0, epoch_min=10,
                 last_epoch=999):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.best_acc = 0
        self.acc_curr = 0
        self.if_checkp = False


        self.epoch_min = epoch_min
        self.save = False
        self.early_stop = False
        self.last_epoch = last_epoch

        self.checkpoints = []

    def check(self, acc, epoch):

        if acc > self.best_acc:
            self.best_acc = acc
            self.acc_curr = acc  # this is saved with model
            self.counter = 0
            if epoch >= self.epoch_min:
                self.save = True
                self.checkpoints.append(epoch)
                self.if_checkp = True
        elif acc < (self.best_acc - self.min_delta):
            self.save = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        if (epoch + 1)%50 == 0:
            self.save = True  # save each 50th epoch anyway
            self.acc_curr = acc  # this is saved with model
            self.if_checkp = False


        return self.best_acc


class EarlyStopping_loss():
    '''
    Adjusted according to:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    '''
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def check(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EarlyStopping_deviation():
    '''
    Adjusted according to:
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    '''
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def check(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        return self.early_stop


class LoadModel():
    def __init__(self, model_arch, model_path, prefix,  device,
                 trained_model=None, trained_epoch=None):

        self.device = device

        # model_arch is the initialized model for training e.g.
        self.model = model_arch

        self.prefix_model = prefix
        self.model_path = model_path

        if trained_model is not None:
            self.update_model(trained_model)
            self.epoch_init = trained_epoch

    def read_model_metadata(self):
        file_path = os.path.join(
            self.model_path, self.prefix_model + '_model_save_df.txt')
        self.model_meta = pd.read_csv(
            file_path, index_col=0, header=0, delimiter='\t').sort_index()
        return

    def load_model(self, epoch=None):
        ''' load trained model from file'''
        if epoch is None:
            self.epoch_init = self.model_meta.index.tolist()[-1]
        else:
            self.epoch_init = epoch

        path_load0 = self.model_meta.loc[self.epoch_init, 'path']
        path_load = os.path.join(self.model_path, os.path.basename(path_load0))
        self.update_model(
            torch.load(path_load, map_location=self.device)['model_state_dict'])
        self.print_model_info()
        #self.prefix += '_ep' + str(self.epoch_init)


    def update_model(self, trained_model):
        '''
        e.g. use self.best_model from TrainigRun if used
        update_best_model=True in save_model_state_dict)
        '''
        remove_prefix = 'module.'
        if self.device.type == 'cpu':
            # this is a workaround if read in on CPU
            # since the model initially created with DataParallel on the GPU
            # (torch.nn.DataParallel(model, device_ids=GPU_no)
            # there is a missmatch with the dict keys on open on
            # CPU there is an additional .module which needs to be removed
            trained_model_updated = {
                k[len(remove_prefix):] if k.startswith(remove_prefix)
                else k: v for k, v in trained_model.items()}
            try:
                self.model.load_state_dict(trained_model_updated)
            except:
                self.model.load_state_dict(trained_model)
        else:
            self.model.load_state_dict(trained_model)


        self.model.to(self.device)
        self.model.eval()  # make sure that model is in evaluation stage
        self.print_model_info()
        return

    def print_model_info(self):
        '''Print model's state_dict'''
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())


class TestRun(
        LoadModel, custom_metric.ignMetric, custom_metric.tmMetric,
        custom_metric.PlotPred):
    def __init__(self, model, data_loader,
                 n_batch, n_classes, class_labels,
                 model_path, out_path, prefix, device, prefix_model,
                 trained_model=None, trained_epoch=None,
                 ignore_index=-100, prefix_out_add='', max_subp=5,
                 save_files=False, EPSG_OUT=32654, DICT_RELABEL=None,
                 MASK_TO_NAN_LST=None,
                 probab_out_band_name_lst=None,
                 phase='test',
                 class_eval_weighted_TP_TN_lst=None):
        '''
        '''
        super().__init__(model, model_path, prefix_model, device, trained_model,
                         trained_epoch)  # include __init__ from inherited class

        # get self.epoch_init from model load
        self.data_loader = data_loader
        self.phase = phase
        self.n_batch = n_batch
        self.n_batch_plot = min(max_subp, n_batch)
        self.n_classes = n_classes
        self.class_labels = class_labels

        self.out_path = out_path

        self.device = device

        self.ignore_index = ignore_index

        # nitialize metric classes
        self.monitor_ign = custom_metric.MetricSummary(
            [phase], out_path, prefix)
        self.monitor_tm = custom_metric.MetricSummary(
            [phase], out_path, prefix)

        self.fontsize = 12
        self.figsize = [8.3, 11.7]

        self.n_data_batch = len(self.data_loader)
        self.prefix = prefix
        self.prefix_out_add = prefix_out_add

        self.probab_out_band_name_lst = probab_out_band_name_lst
        if probab_out_band_name_lst is None:
            self.probab_out_band_name_lst = [
                str(x) + '_probab' for x in class_labels[0]]

        self.save_files = save_files
        # the following are only used if save files
        self.EPSG_OUT = EPSG_OUT
        self.DICT_RELABEL = DICT_RELABEL
        self.MASK_TO_NAN_LST = MASK_TO_NAN_LST

        if class_eval_weighted_TP_TN_lst is not None:
            self.class_eval_weighted_TP_TN_lst = class_eval_weighted_TP_TN_lst
        else:
            self.class_eval_weighted_TP_TN_lst = []
        return

    def run_test(self):
        count = 0
        self.model.eval()  # make sure that model is in evaluation stage
        # but has already ben done in update_model and validation
        self.setup_ign_metrics()
        self.setup_tm_metrics()
        self.reset_ign_metrics()
        self.reset_tm_metrics()
        for x, y, file_n_seg, x_coords, y_coords in self.data_loader:
            with torch.no_grad():
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                predb = self.model(x)

                # update on batch set
                self.update_ign_metrics(predb, y)
                self.update_tm_metrics(predb, y)

            # !! compute metrics per batch set
            df_metric, df_class_metric, df_class_dice_metric, df_cm = self.compute_ign_metrics(
                count, self.epoch_init)
            df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm = self.compute_tm_metrics(
                count, self.epoch_init)

            self.monitor_ign.update_metric_summaries(
                df_metric, df_class_metric, df_class_dice_metric, df_cm, self.phase)
            self.monitor_tm.update_metric_summaries(
                df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm, self.phase)

            # get predictions
            pred_seg = predb.argmax(dim=1).cpu()

            if self.save_files:
                # get probabilities
                probabilities = torch.functional.F.softmax(
                    predb, dim=1).cpu()
                n_batch = x.shape[0]
                for i in range(n_batch):
                    # get class labels
                    class_pred = pred_seg[i, :][np.newaxis, :, :]
                    class_proba = probabilities[i, :]

                    file_suffix = os.path.basename(
                        file_n_seg[i]).split('_seg')[0]
                    files_names_saved = batch_save(
                        [class_pred, class_proba],
                        ['class_pred', 'proba_pred'],
                        x_coords[i][~x_coords[i].isnan()],  # remove nans that have been added such that all have same size
                        y_coords[i][~y_coords[i].isnan()],
                        f"{self.prefix}_{file_suffix}_ep{self.epoch_init:02d}",
                        self.out_path,
                        self.class_labels, self.probab_out_band_name_lst,
                        self.EPSG_OUT)


                    # --- create geojson and make stats comparison --
                    # possible TODO: use as input directly xarray such
                    # that do not have to reopen file again
                    # Note: prefix_out_test removes pred at end (pred and true
                    # will be added in the function)
                    prefix_out_test = '_'.join(
                        files_names_saved['class_pred'].split('.')[0].split('_')[:-1])
                    gdf_pred, gdf_true = MAIN_extract_segmentation_properties.main(
                        PRED_IMG=files_names_saved['class_pred'],
                        TRUE_IMG=file_n_seg[i],
                        AOI_PATH=None, EPSG=self.EPSG_OUT,
                        DICT_RELABEL=self.DICT_RELABEL,
                        MASK_TO_NAN_LST=self.MASK_TO_NAN_LST,
                        PREFIX_OUT=prefix_out_test,
                        additionally_save_with_min_px_size=None)
                    for i_class_eval in self.class_eval_weighted_TP_TN_lst:
                        df_cts = MAIN_extract_TP_TN_per_class_uncert.main(
                            PRED_IMG=files_names_saved['class_pred'],
                            TRUE_IMG=file_n_seg[i],
                            CLASS_TO_EVAL=i_class_eval,
                            AOI_PATH=None, EPSG=self.EPSG_OUT,
                            DICT_RELABEL=self.DICT_RELABEL,
                            MASK_TO_NAN_LST=self.MASK_TO_NAN_LST,
                            PREFIX_OUT=prefix_out_test,
                            min_px_size=None)

            # print prediction results
            metric_show = self.monitor_tm.summary[self.phase][-1].loc[(self.epoch_init, count), :]
            text_title = self.get_title_pred(metric_show, self.epoch_init)
            print(text_title)

            # initialize predition plot
            self.initialize_plot_pred()
            self.plot_pred_per_batch(x, y, pred_seg)

            file_prefix = (
                self.prefix + '_' + self.prefix_out_add + self.phase
                + '_ep' + '{0:02d}'.format(self.epoch_init)
                + '_test_batch' + '{0:02d}'.format(count))
            self.save_plot_pred(
                file_prefix, self.out_path, text_title)

            # plot confusion matrix
            file_name = file_prefix + '_confusion_m.png'
            cm_true = metrics_plotting_utils.get_cm_ratio(
                self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=1)
            cm_pred = metrics_plotting_utils.get_cm_ratio(
                self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=0)
            metrics_plotting_utils.plot_cm_matrix(
                self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True),
                cm_true, cm_pred, self.out_path, file_name,
                'counts \n' + text_title, 'normalized_true',
                'normalized_pred')
            count += 1

        # !! recalculate to get average for full dataset
        df_metric, df_class_metric, df_class_dice_metric, df_cm = self.compute_ign_metrics(
            9999, self.epoch_init)
        df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm = self.compute_tm_metrics(
            9999, self.epoch_init)

        self.monitor_ign.update_metric_summaries(
            df_metric, df_class_metric, df_class_dice_metric, df_cm, self.phase)
        self.monitor_tm.update_metric_summaries(
            df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm, self.phase)

        # plot confusion matrix (overall counts)
        file_name = (self.prefix + '_' + self.prefix_out_add + self.phase
                     + '_ep' + '{0:02d}'.format(self.epoch_init)
                     + '_confusion_m.png')
        cm_true = metrics_plotting_utils.get_cm_ratio(
            self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=1)
        cm_pred = metrics_plotting_utils.get_cm_ratio(
            self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=0)
        metrics_plotting_utils.plot_cm_matrix(
            self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True),
            cm_true, cm_pred,
            self.out_path, file_name, 'counts \n' + text_title,
            'normalized_true', 'normalized_pred')

        # reset all metrics
        self.reset_ign_metrics()
        self.reset_tm_metrics()

        self.monitor_ign.merge_metric_summaries()
        self.monitor_tm.merge_metric_summaries()

        # save metrics summary
        self.monitor_ign.save_metric_summaries(
            self.prefix_out_add + self.phase + '_cummul_ign')
        self.monitor_tm.save_metric_summaries(
            self.prefix_out_add + self.phase + '_cummul_tm')
        return


class TestOnPredicted(
        custom_metric.ignMetric, custom_metric.tmMetric,
        custom_metric.PlotPred):
    def __init__(self, data_loader,
                 n_classes, class_labels,
                 out_path, prefix, device,
                 ignore_index=-100, prefix_out_add='', max_subp=5,
                 save_files=False, EPSG_OUT=32654, DICT_RELABEL=None,
                 MASK_TO_NAN_LST=None,
                 probab_out_band_name_lst=None,
                 phase='test',
                 class_eval_weighted_TP_TN_lst=None):
        '''
        '''

        self.data_loader = data_loader
        self.n_batch_plot = max_subp
        self.n_classes = n_classes
        self.class_labels = class_labels
        self.phase = phase

        self.out_path = out_path

        self.device = device

        self.ignore_index = ignore_index

        # epoch is just set to 0 since the probabilities come form a mixed
        # enxemble
        self.epoch_init = 0

        # nitialize metric classes
        self.monitor_ign = custom_metric.MetricSummary(
            [phase], out_path, prefix)
        self.monitor_tm = custom_metric.MetricSummary(
            [phase], out_path, prefix)

        self.fontsize = 12
        self.figsize = [8.3, 11.7]


        self.prefix = prefix
        self.prefix_out_add = prefix_out_add

        self.probab_out_band_name_lst = probab_out_band_name_lst
        if probab_out_band_name_lst is None:
            self.probab_out_band_name_lst = [
                str(x) + '_probab' for x in class_labels[0]]

        self.save_files = save_files
        # the following are only used if save files
        self.EPSG_OUT = EPSG_OUT
        self.DICT_RELABEL = DICT_RELABEL
        self.MASK_TO_NAN_LST = MASK_TO_NAN_LST

        if class_eval_weighted_TP_TN_lst is not None:
            self.class_eval_weighted_TP_TN_lst = class_eval_weighted_TP_TN_lst
        else:
            self.class_eval_weighted_TP_TN_lst = []
        return

    def run_test_on_predict(self):
        count = 0
        self.setup_ign_metrics()
        self.setup_tm_metrics()
        self.reset_ign_metrics()
        self.reset_tm_metrics()
        for predb, y, file_n_seg, x_coords, y_coords in self.data_loader:
            #self.n_data_batch = predb.shape[0]
            # update on set
            self.update_ign_metrics(predb, y)
            self.update_tm_metrics(predb, y)

            # !! compute metrics per batch set
            df_metric, df_class_metric, df_class_dice_metric, df_cm = self.compute_ign_metrics(
                count, self.epoch_init)
            df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm = self.compute_tm_metrics(
                count, self.epoch_init)

            self.monitor_ign.update_metric_summaries(
                df_metric, df_class_metric, df_class_dice_metric, df_cm, self.phase)
            self.monitor_tm.update_metric_summaries(
                df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm, self.phase)

            # get predictions
            pred_seg = predb.argmax(dim=1).cpu()

            if self.save_files:
                # get probabilities
                probabilities = torch.functional.F.softmax(
                    predb, dim=1).cpu()
                n_batch = predb.shape[0]
                for i in range(n_batch):
                    # get class labels
                    class_pred = pred_seg[i, :][np.newaxis, :, :]
                    class_proba = probabilities[i, :]

                    file_suffix = os.path.basename(
                        file_n_seg[i]).split('_seg')[0]
                    files_names_saved = batch_save(
                        [class_pred, class_proba],
                        ['class_pred', 'proba_pred'],
                        x_coords[i][~np.isnan(x_coords[i])],  # remove nans that have been added such that all have same size
                        y_coords[i][~np.isnan(y_coords[i])],
                        f"{self.prefix}_{file_suffix}_ep{self.epoch_init:02d}",
                        self.out_path,
                        self.class_labels, self.probab_out_band_name_lst,
                        self.EPSG_OUT)


                    # --- create geojson and make stats comparison --
                    # possible TODO: use as input directly xarray such
                    # that do not have to reopen file again
                    # Note: prefix_out_test removes pred at end (pred and true
                    # will be added in the function)
                    prefix_out_test = '_'.join(
                        files_names_saved['class_pred'].split('.')[0].split('_')[:-1])
                    gdf_pred, gdf_true = MAIN_extract_segmentation_properties.main(
                        PRED_IMG=files_names_saved['class_pred'],
                        TRUE_IMG=file_n_seg[i],
                        AOI_PATH=None, EPSG=self.EPSG_OUT,
                        DICT_RELABEL=self.DICT_RELABEL,
                        MASK_TO_NAN_LST=self.MASK_TO_NAN_LST,
                        PREFIX_OUT=prefix_out_test,
                        additionally_save_with_min_px_size=None)
                    for i_class_eval in self.class_eval_weighted_TP_TN_lst:
                        df_cts = MAIN_extract_TP_TN_per_class_uncert.main(
                            PRED_IMG=files_names_saved['class_pred'],
                            TRUE_IMG=file_n_seg[i],
                            CLASS_TO_EVAL=i_class_eval,
                            AOI_PATH=None, EPSG=self.EPSG_OUT,
                            DICT_RELABEL=self.DICT_RELABEL,
                            MASK_TO_NAN_LST=self.MASK_TO_NAN_LST,
                            PREFIX_OUT=prefix_out_test,
                            min_px_size=None)

            # print prediction results
            metric_show = self.monitor_tm.summary[self.phase][-1].loc[(self.epoch_init, count), :]
            text_title = self.get_title_pred(metric_show, self.epoch_init)
            print(text_title)

            # initialize predition plot
            self.initialize_plot_pred()
            self.plot_pred_per_batch(
                predb[:, [1], :, :], y, pred_seg, inp_img_title='proba 1')

            file_prefix = (
                self.prefix + '_' + self.prefix_out_add + self.phase
                + '_ep' + '{0:02d}'.format(self.epoch_init)
                + '_test_batch' + '{0:02d}'.format(count))
            self.save_plot_pred(
                file_prefix, self.out_path, text_title)

            # plot confusion matrix
            file_name = file_prefix + '_confusion_m.png'
            cm_true = metrics_plotting_utils.get_cm_ratio(
                self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=1)
            cm_pred = metrics_plotting_utils.get_cm_ratio(
                self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=0)
            metrics_plotting_utils.plot_cm_matrix(
                self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True),
                cm_true, cm_pred, self.out_path, file_name,
                'counts \n' + text_title, 'normalized_true',
                'normalized_pred')
            count += 1

        # !! recalculate to get average for full dataset
        df_metric, df_class_metric, df_class_dice_metric, df_cm = self.compute_ign_metrics(
            9999, self.epoch_init)
        df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm = self.compute_tm_metrics(
            9999, self.epoch_init)

        self.monitor_ign.update_metric_summaries(
            df_metric, df_class_metric, df_class_dice_metric, df_cm, self.phase)
        self.monitor_tm.update_metric_summaries(
            df_tm_metric, df_class_tm_metric, df_class_dice_tm_metric, df_tm_cm, self.phase)

        # plot confusion matrix (overall counts)
        file_name = (self.prefix + '_' + self.prefix_out_add + self.phase
                     + '_ep' + '{0:02d}'.format(self.epoch_init)
                     + '_confusion_m.png')
        cm_true = metrics_plotting_utils.get_cm_ratio(
            self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=1)
        cm_pred = metrics_plotting_utils.get_cm_ratio(
            self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True), axis=0)
        metrics_plotting_utils.plot_cm_matrix(
            self.monitor_tm.summary_cm[self.phase][-1].reset_index(['epoch', 'count'], drop=True),
            cm_true, cm_pred,
            self.out_path, file_name, 'counts \n' + text_title,
            'normalized_true', 'normalized_pred')

        # reset all metrics
        self.reset_ign_metrics()
        self.reset_tm_metrics()

        self.monitor_ign.merge_metric_summaries()
        self.monitor_tm.merge_metric_summaries()

        # save metrics summary
        self.monitor_ign.save_metric_summaries(
            self.prefix_out_add + self.phase + '_cummul_ign')
        self.monitor_tm.save_metric_summaries(
            self.prefix_out_add + self.phase + '_cummul_tm')
        return


class PredictRun(LoadModel):
    def __init__(self, model, data_loader,
                 n_batch, n_classes, class_labels,
                 out_path, prefix, device, prefix_inp,
                 EPSG_OUT,
                 probab_out_band_name_lst=None,
                 trained_model=None, model_path=None,
                 width_out=None, height_out=None,
                 save_proba=False):

        # include __init__ from inherited class
        if model_path is None:
            model_path = out_path
        super().__init__(model, model_path, prefix_inp, device, trained_model)
        self.data_loader = data_loader
        self.out_path = out_path

        self.prefix = prefix

        self.n_batch = n_batch
        self.n_classes = n_classes

        self.class_labels = class_labels

        self.probab_out_band_name_lst = probab_out_band_name_lst
        if probab_out_band_name_lst is None:
            self.probab_out_band_name_lst = [
                str(x) + '_probab' for x in class_labels[0]]

        self.width_out = width_out
        self.height_out = height_out

        self.save_proba = save_proba

        self.EPSG_OUT = EPSG_OUT

        return

    def run_predict_proba(self, parallel_run=True,
                          keep_class=True, keep_proba_softm=True,
                          keep_proba_raw=True, file_suffix_out=''):
        '''
        The exported predictions include:
         - the classes
         - the probabilities after using softmax or None if not keep_proba_mean
         - the raw probablilities (no softmax), proba_raw or None if not keep_raw

         shape of x is [n_batchs, channels, y, x]
        '''
        files_saved_lst = []
        for x, file_n, x_coords, y_coords in iter(self.data_loader):
            with torch.no_grad():
                x = x.to(self.device, non_blocking=True)
                predb = self.model(x)

            n_batch = x.shape[0]
            # derive outputs
            ind_dict = {}
            if keep_class:
                # get predicted class and adjust dimensions to match the
                # probabilities with [n_batch, band(here class), y, x]
                ind_dict['class'] = predb.argmax(dim=1)[:, np.newaxis, :, :].cpu().numpy()
            if keep_proba_raw:
                # predicted raw pseudo-probabilities
                ind_dict['proba_raw'] = predb.cpu().numpy()
            if keep_proba_softm:
                # predicted probabilities with softmax shifted to 0 - 1
                ind_dict['proba_softmax'] = torch.functional.F.softmax(
                    predb, dim=1).cpu().numpy()

            # -- create inputs for batch save
            def inp_loop():
                # create input for saving the calculated output (as tif)
                # such that can save batches in parallel
                for i in range(n_batch):
                    arr_lst = []
                    suffix_lst = []
                    for i_key, i_arr in ind_dict.items():
                        arr_lst.append(i_arr[i, :, :, :])
                        suffix_lst.append(i_key)

                    file_prefix_inp = '{}{}_ep{:02d}'.format(
                        os.path.basename(file_n[i]).split('.')[0],
                        file_suffix_out, self.epoch_init)

                    yield(arr_lst, suffix_lst, x_coords[i], y_coords[i],
                          file_prefix_inp)

            # additional inputs for batch save
            inp_param_save = [
                self.out_path,
                self.class_labels, self.probab_out_band_name_lst,
                self.EPSG_OUT,
                self.width_out, self.height_out]

            if parallel_run:
                # predict batch in parallel
                n_jobs = min(int(cpu_count()-10), n_batch)
                w = Parallel(n_jobs=n_jobs, verbose=0)(delayed(
                    batch_save)(*i, *inp_param_save) for i in inp_loop())
                files_saved_lst.append(pd.DataFrame(w))
            else:
                w = []
                for i in inp_loop():
                    w.append(batch_save(*i, *inp_param_save))
                files_saved_lst.append(pd.DataFrame(w))
            del w
            gc.collect()
        self.files_saved = pd.concat(files_saved_lst, axis=0)
        return

    def img_clip_cog(self, AOI_path, AOI_file):

        # optionally clip to AOI and save to cog
        for i_key, i_file_lst in self.files_saved.items():
            for i_file in i_file_lst:
                geo_utils.read_clip_to_aoi(
                    os.path.join(AOI_path, AOI_file),
                    i_file, to_cog=True)
        return

    def merge_predicted_img(self, AOI_path=None, AOI_file=None):
        '''
        merge tiffs either with "first" and optional with mean if
        use mean_merge=True
        '''
        # prefix for merged output files
        file_prefix = f'{self.prefix}_ep{self.epoch_init:02d}_tile_merged'

        # extract class and probability .tifs as they are merged differently
        saved_types =  self.files_saved.keys()
        saved_types_c = [x for x in saved_types if x.find('class') > -1]
        saved_types_p = [x for x in saved_types if x.find('class') == -1]

        # metadata to be added to the merged raster file
        meta_tags_dict_class = {
            "class_num": tuple(self.class_labels[0]),
            "class_label": tuple(self.class_labels[1]),
            "long_name": 'class'}
        meta_tags_dict_proba = {
            "class_num": tuple(self.class_labels[0]),
            "class_label": tuple(self.class_labels[1]),
            "long_name": self.probab_out_band_name_lst}

        # create output path if non existent:
        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)

        # ---- get "class" by merging tiles with taking "first" ----
        # (it does not make sense to average classes)
        for i_key in saved_types_c:
            file_name_out = f'{file_prefix}_pred_{i_key}.tif'
            # merge by taking first pixel value
            out_path_class = merge_rasterio_taking_first(
                self.files_saved[i_key].tolist(), self.out_path,
                file_name_out, meta_tags_dict_class)

            # optionally save to cog
            geo_utils.tif_to_cog_rasterio(out_path_class)
            # other option for cog creation but this gives a warning
            # geo_utils.tif_to_cog_rio(out_path_class)

            # optionally clip to AOI and save to cog
            if AOI_file is not None:
                geo_utils.read_clip_to_aoi(
                    os.path.join(AOI_path, AOI_file),
                    out_path_class, to_cog=True)

        # -- get probabilities by averaging the overlapping probabilities
        # and derive class with argmax
        for i_key in saved_types_p:
            file_name_prefix_out = f'{file_prefix}_pred_{i_key}'
            # merge by taking mean
            out_path_proba, out_path_class = merge_array_rasterio(
                self.files_saved[i_key].tolist(),
                self.out_path, file_name_prefix_out,
                meta_tags_dict_class, meta_tags_dict_proba)

            # optionally save to cog
            geo_utils.tif_to_cog_rasterio(out_path_proba)
            geo_utils.tif_to_cog_rasterio(out_path_class)

            # optionally clip to AOI and save to cog
            if AOI_file is not None:
                # clip merged image if required
                for i_path in [out_path_proba, out_path_class]:
                    geo_utils.read_clip_to_aoi(
                        os.path.join(AOI_path, AOI_file),
                        i_path, to_cog=True)
        return


# --------------- SUBFUNCTIONS -------
def acc_metric(pred, y):
    '''
    pred = predb.argmax(dim=1) has shape [n_batches, height, width]
        (same shape as yb)

    model predb = model(x) has dimension:
    pred are probabilities of shape
    [n_batch, n_channel(=n_classes), height, width]

    argmax Returns the indices of the maximum value of all elements
    in the input tensor.

    float changes matrix to 0 and 1 and taking mean gives accuracy
    (same as count_True/count_data)
    '''
    out_all = (pred == y).float().mean()
    out = (pred == y).float()
    out[y == 0] = np.nan
    return out_all, out.nanmean()


def merge_rasterio_taking_first(file_list, path_export, file_name_out,
                                add_meta_tags_dict):
    """
    merges tifs by using the method "first"
    """

    # -- merge the file using "first"
    mosaic_merged, out_trans_merged = rasterio_merge(
        file_list)

    # -- read first file to get metadata to be added to the merged file
    file_first = rasterio.open(file_list[0], mode='r')
    out_meta = file_first.meta.copy()

    # -- Update the metadata with the new shape and transform
    # (transform comes from the merged file)
    out_meta.update({
        "height": mosaic_merged.shape[1],
        "width": mosaic_merged.shape[2],
        "dtype": mosaic_merged.dtype,
        "transform": out_trans_merged,
        })

    # -- Write the merged result to a new file

    out_path = os.path.join(path_export, file_name_out)
    with rasterio.open(out_path, 'w', **out_meta) as dest:
        dest.write(mosaic_merged)
        dest.descriptions = get_band_tuple(add_meta_tags_dict['long_name'])
        dest.update_tags(**add_meta_tags_dict.copy())

    # Close the raster files
    file_first.close()

    return out_path


def get_band_tuple(long_name):
    if isinstance(long_name, str):
        band_inp = tuple([long_name])
    else:
        band_inp = tuple(long_name)
    return band_inp


def merge_array_rasterio(
        file_merge_list, path_export, file_name_prefix_out,
        meta_tags_dict_class, meta_tags_dict_proba, use_softmax=True,
        merge_type='mean'):
    """
    merges arrays as follows:
      - takes the mean merge or max of the probabilities
      - optionally takes softmax
      - uses argmax for the classes
      - saves both the merged probabilities and the new classes

    Parameters
    ----------
    file_merge_list : list
        List of file paths or of images (opened rasterio.open(x))
        to be merged.
    path_export : str
        Directory path where the merged files will be saved.
    file_name_prefix_out : str
        Prefix for the output file names.
    meta_tags_dict_class : dict
        Dictionary containing metadata for the class array.
    meta_tags_dict_proba : dict
        Dictionary containing metadata for the probability array.
    use_softmax : bool, optional
        Whether to apply softmax to the probabilities.
    merge_type : str, optional
        Type of merge ('mean' or 'max').

    Returns
    -------
    out_path_proba : str
        Path to the merged probability array file.
    out_path_class : str
        Path to the merged class array file.
    (created merged images with probabilities and classes are saved to
     file)

    Notes
    -----
    - `meta_tags_dict_proba` and `meta_tags_dict_class` must contain
       band names to assign band names to output.
    - If `use_softmax` is True, the probabilities are transformed using
       softmax.
    - If `merge_type` is 'mean', the rasters are merged using
      copy_sum/copy_count. If 'max', the rasters are merged using
      copy_max.
    - the metadata for the output is taken for the first file in the list

    """
    if merge_type == 'mean':
        # -- calculate mean
        # Merge the rasters using the copy_sum/copy_count functions
        mosaic_sum, out_transf = rasterio_merge(
            file_merge_list, method=rasterio.merge.copy_sum)
        mosaic_count, out_transf_count = rasterio_merge(
            file_merge_list, method=rasterio.merge.copy_count)
        mosaic_out = mosaic_sum/mosaic_count
    elif merge_type == 'max':
        # -- calculate max
        mosaic_out, out_transf = rasterio_merge(
            file_merge_list, method=rasterio.merge.copy_max)
    # mosaic out is numpy array
    # out_transf contains transformation information to be added to
    #   metadata further below

    # -- get mean probabilities in range 0 - 1 with softmax
    if use_softmax:
        arr_torch = torch.tensor(
            mosaic_out, dtype=torch.float32)
        arr_proba = torch.functional.F.softmax(
            arr_torch, dim=0).cpu().numpy()
        suffix_out = 'softmax'
    else:
        arr_proba = mosaic_out
        suffix_out = 'raw'

    # -- read first file to get metadata to be added to the merged file
    file_first = rasterio.open(file_merge_list[0], mode='r')
    out_meta = file_first.meta.copy()

    # -- Update the metadata with the new shape and transform
    # (transform comes from the merged file)
    out_meta.update({
        "height": arr_proba.shape[1],
        "width": arr_proba.shape[2],
        "dtype": arr_proba.dtype,
        "transform": out_transf,
        })

    # -- Write the merged probabilities to file
    out_path_proba = os.path.join(
        path_export,
        f"{file_name_prefix_out}_{merge_type}_proba_{suffix_out}.tif")
    with rasterio.open(out_path_proba, 'w', **out_meta) as dest:
        dest.write(arr_proba)
        dest.descriptions = get_band_tuple(meta_tags_dict_proba['long_name'])
        dest.update_tags(**meta_tags_dict_proba.copy())

    # --- get CLASS values form probabilitis
    class_array = np.argmax(arr_proba, axis=0)
    # as here use metadata from above just need to adjust it
    # for classes
    out_meta.update({
        "height": class_array.shape[0],
        "width": class_array.shape[1],
        "dtype": class_array.dtype,
        "count": 1,
        "nodata": 0
        })

    # -- write classes to file
    out_path_class = os.path.join(
        path_export,
        f"{file_name_prefix_out}_{merge_type}_{suffix_out}_class.tif")
    with rasterio.open(out_path_class, 'w', **out_meta) as dest:
        dest.write(class_array[np.newaxis, :, :])
        dest.descriptions = get_band_tuple(
            meta_tags_dict_class['long_name'])
        dest.update_tags(**meta_tags_dict_class.copy())

    # Close the raster files
    file_first.close()

    return out_path_proba, out_path_class


def initialize_lr(optimizer_inp, param_lr_scheduler, param_gamma,
                  param_milestones):
    ''' initialiye learning rate'''

    if param_lr_scheduler is None:
        return []

    scheduler = []
    for i_g, i_s in zip(param_gamma, param_lr_scheduler):
        if i_s == 'exp':
            scheduler.append(
                lr_scheduler.ExponentialLR(optimizer_inp, gamma=i_g))
        elif i_s == 'mulstep':
            scheduler.append(
                lr_scheduler.MultiStepLR(
                    optimizer_inp, milestones=param_milestones,
                    gamma=i_g))
    return scheduler


def batch_save(arr_lst, suffix_lst, x_coords, y_coords,
               file_prefix, out_path,
               class_labels, probab_out_band_name_lst, EPSG_OUT,
               width_out=None, height_out=None):
    '''
    class_pred_b = pred_seg[i, :]
    class_proba_b = probabilities[i, :]
    raw_proba = pred_seg_raw[i, :] or None
    file_b = file_n[i]

    width_out and heigth_out are only used in prediction
    '''
    file_name_out = {}
    for i_suff, i_arr in zip(suffix_lst, arr_lst):
        if i_suff.find('class') > -1:
            band_lst = ['class']
            nodata_val = 0
        else:
            band_lst = probab_out_band_name_lst
            nodata_val = np.nan

        file_name_out[i_suff] = save_prediction_img_single(
                i_arr, x_coords, y_coords, band_lst,
                file_prefix, i_suff, out_path,
                EPSG_OUT, nodata_val, class_labels,
                width_out=width_out, height_out=height_out)

    return file_name_out


def save_prediction_img_single(
        pred_arr, x_coords, y_coords, band_name_lst,
        file_prefix_out, file_suffix_out, path_out,
        EPSG_INT,
        nodata_val, class_labels,
        width_out=None, height_out=None, window_scale=0.7):
    """
    Function to save predicted image. Withut cropping edges

    pred_arr could be:
        predicted classes: pred_seg[i, :][np.newaxis, :, :]
                            (with i being batch index)
        predicted probabilities: probabilities[i, :]
                            (with i being batch index)

    file_prefix_out: could be file_n[i].split('.')[0]
        is file name of xarray img. naming is used for output

    class_labels: is list with two sublists with class numbers and
        respective names

    epoch: which epoch number was used to create prediction

    probab_out_band_name_lst: list with naming for probabilities

    EPSG_INT: out epsg as int


    width_out and heigth_out are only used in prediction
    """
    # get amount of pixels

    n_x = x_coords.shape[0]
    n_y = y_coords.shape[0]

    if width_out is not None and height_out is not None:
        # -- if prediction was done on tiles (i.e. with specified window size)
        # scale output window size
        # this is required for prediction tiles to get rid of edge effects
        width_out_scaled = int(np.ceil(width_out*window_scale))
        height_out_scaled = int(np.ceil(height_out*window_scale))

        # center crop image
        pred_arr, x_crop_id, y_crop_id = geo_utils.center_crop_numpy(
            pred_arr, n_x, n_y, width_out_scaled, height_out_scaled)

        # save new coords
        x_coords_c = x_coords[x_crop_id[0]: x_crop_id[1]]
        y_coords_c = y_coords[y_crop_id[0]: y_crop_id[1]]
    else:
        # ----- if prediction was done on whole image at once ----
        # !!! here edges are not cropped !!!
        # BUT crop image back to original size (without padding to get
        # %32 size, see custom_data_loader)
        pred_arr = pred_arr[:, :n_y, :n_x]

        # here coordinates do not need to be cropped since they have
        # already the correct size
        x_coords_c = x_coords
        y_coords_c = y_coords

    # --- save image
    # Note: tiles are merged later with merge_predicted_img
    file_n_prefix_out = (
        f"{file_prefix_out}_{file_suffix_out}.tif")

    attrs = {'class_num': class_labels[0],
             'class_label': class_labels[1]}

    # save class
    file_name_exported_img = geo_utils.create_save_xarray_from_numpy(
            pred_arr, x_coords_c, y_coords_c, band_name_lst,
            path_out, file_n_prefix_out,
            EPSG_INT, attrs=attrs, nodata=nodata_val)

    # return file name excluding '_pred' suffix
    return file_name_exported_img






