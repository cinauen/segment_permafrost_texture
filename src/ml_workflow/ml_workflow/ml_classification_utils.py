'''
Class for supervised machine learning classification

Processes contained:
 - training, testing, prediction
 - feature importance analysis
 - hyperparameter tuning

Inherits ml_preproc_scikit.PreprocScikitML which can be used for
image pre-processing using scikit-learn



TODO:
- replace self.text_add with a proper logging option
- check for better options on how to bets deal with negative values
  when taking log or exp. Currently here just shift data acoording
  to min value si self.X
- adapt max_features for PCA usage (use n_components instead of
  n_features)

'''
import os
import numpy as np
import pandas as pd
import shap
import pickle
import time
import datetime as dt
import treelite

import torch
import torchmetrics

import dask_ml.model_selection as dcv

if torch.cuda.is_available():
    # if GPU is available
    import cupy
    import cudf
    # modules usd to distribute data
    from cuml.dask.common import utils as dask_utils
    import dask_cudf
    # modules for classification
    from cuml.preprocessing import StandardScaler
    from cuml.ensemble import RandomForestClassifier
    from cuml.model_selection import train_test_split
    import cuml.explainer as cu_expl
else:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import dask.dataframe as dask_df
    import dask.array as dask_array


from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ----------------- import custom utils -----------------
import ml_workflow.ml_preproc_scikit as ml_preproc_scikit
import utils.image_preproc as image_preproc
import utils.metrics_plotting_utils as metrics_plotting_utils
import utils.conversion_utils as conversion_utils


class ClassifierML(image_preproc.ImagePreproc,
                   ml_preproc_scikit.PreprocScikitML):
    '''
    class for machine learning clustering
    modified from
    https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
    '''
    def __init__(self, X_inp, Y_inp, X_inp_test, Y_inp_test,
                 features_col, class_col,
                 path_export, file_prefix, classifier_name,
                 sorted_class, sorted_class_names, dict_assign,
                 phase_lst, EPSG_target,
                 preproc_type=None, algo_param=None,
                 n_components=5, n_jobs=None,
                 train_count=1, algo_param_gs=None, classifier_optimi=None,
                 GPU_proc=True):
        '''
        '''
        self.clf_name = classifier_name
        self.file_prefix = file_prefix  # to save hdf file

        self.X = X_inp
        self.Y = Y_inp
        self.X_test = X_inp_test
        self.Y_test = Y_inp_test

        self.n_jobs = n_jobs  # how many jobs in parallel for permutation
        # tests

        self.GPU_proc = GPU_proc  # if processing is done on GPU

        # Optimisation options to set up classifier. For scikit learn
        # use e.g. {'n_jobs':20} for cuml could also specify workers or
        # n_streams.
        if classifier_optimi is None:
            self.classifier_optimi = {}
        else:
            self.classifier_optimi = classifier_optimi

        # list of column names used for training
        self.features_col = features_col  # list can bet adjusted e.g. if
        # calcuate PCA with PreprocScikitML
        self.features_col_orig = features_col.copy()
        self.col_prefix = ''  # prefix used for scikit preprocessing
        # labels in PreprocScikitML

        self.class_col = class_col  # string of column name which
        # contains category/class

        self.path_export = path_export
        self.EPSG_target = EPSG_target

        self.dict_assign = dict_assign  # to assign plotting
        # colors per class name

        self.sorted_class = sorted_class
        self.sorted_class_names = sorted_class_names
        self.n_classes = len(self.sorted_class)

        self.set_classification_param_gs(algo_param_gs) # for grid search
        self.initialize_setup(algo_param)
        self.initialize_evaluation_dicts(phase_lst)
        #if len(preproc_type) > 0:
        self.preproc_type = preproc_type
        self.initialize_preproc(preproc_type, n_components)

        out_path_text = os.path.join(
            path_export, f'{file_prefix}_eval_text.txt')
        self.text_add = open(out_path_text, 'w')  # 'a+'

        self.train_count = train_count - 1
        self.train_count_str = '{0:02d}'.format(self.train_count)

        self.pred_file_df = {}

    def initialize_setup(
            self, algo_param):
        '''
        This needs to be done before setup cluster
        '''
        self.set_classification_param(algo_param)
        self.set_up_classification_algorithms()  # this uses self.a_param

        return

    def set_classification_param_gs(self, algo_param=None):
        '''
        sets clustering parameters need to be run before run
        self.set_up_classification_algorithm()
        '''
        n_features = len(self.features_col)

        # if float the max features is used as fraction!
        max_features_inp = list(set([
            'sqrt', #sqrt = 1.0/np.sqrt(n_features)
            1.0/2.0,
            None]))  # is 1.0 = n_features
        default_base = {
            'RandomForest': {
                'n_estimators': [10, 100, 500, 1000],  # cum 100 is default
                'max_depth': [5, 10, 16],  # cuml: 16 is default
                'max_samples': [0.25, 0.5, 1],  # 1 is default
                'max_features': max_features_inp,
                #'split_criterion': [0, 1],  # cuml: gini is default
                #'min_samples_split': [2, 4],
                #'min_samples_leaf': [1, 2],  # cuml 1 is default
                #'bootstrap': [True, False],
                #'class_weight': (None, 'balanced', 'balanced_subsample'),
                #'ccp_alpha': [0, 1]
                }
            }
        if algo_param is None:
            algo_param = {}

        self.a_param_gs = default_base.copy()
        self.a_param_gs.update(algo_param)
        return

    def set_classification_param(self, algo_param=None,
                                 classifier_lst=None):
        """
        set_classification_param needs to be run before run
        self.set_up_classification_algorithm()

        Note it would be possible to implement other algorith e.g. with
            classifier_lst = ['RandomForest',
                              'NearestNeighbors',
                              'LinearSVM', 'RBF_SVM',
                              ...
                              ]
            algo_param = {
                'RandomForest': dict(
                    max_depth=5, n_estimators=10, max_features=1),
                NearestNeighbors': dict(KNeighborsMixin=3),
                'LinearSVM': dict(kernel="linear", C=0.025, probability=True),
                'RBF_SVM': dict(gamma=2, C=1, probability=True),
                ...
                }
        """
        if classifier_lst is None:
            classifier_lst = ['RandomForest']

        self.a_param = {x: {} for x in classifier_lst}
        self.a_param['RandomForest'].update(
            dict(bootstrap=True))
        if algo_param is None:
            algo_param = {}

        self.a_param.update(algo_param)
        return

    def set_up_classification_algorithms(self):
        """
        initializes all possible classification algos
        n_components is only important if have pca in preprocessing step
            (only with vMLxxx_scikit)

        Note:
            here could also add other implement other classifiers with
            classifier = {
                'RandomForest': RandomForestClassifier(
                    **self.a_param['RandomForest'], **self.classifier_optimi),  # predict_log_proba
                'NearestNeighbors': KNeighborsClassifier(
                     **self.a_param['NearestNeighbors']),
                'LinearSVM': SVC(
                     **self.a_param['LinearSVM']),
                'RBF_SVM': SVC(
                     **self.a_param['RBF_SVM']),
                ...
                }
        """
        classifier = {
            'RandomForest': RandomForestClassifier(
                **self.a_param['RandomForest'],
                **self.classifier_optimi  # optimisation e.g. if calculate on CPU use {n_jobs=20}
                    ),
                }
        self.classifier = classifier[self.clf_name]
        return

    def convert_type(self):
        self.X = self.X.astype('float32')
        self.X_test = self.X_test.astype('float32')

        self.Y = self.Y.astype('int32')
        self.Y_test = self.Y_test.astype('int32')
        return

    def run_grid_search(
            self, count_search_str,
            create_pipeline=False, scoring=None,
            refit_inp=True, gs_split='shuffle', cv_num=5, **kwargs):
        '''
        for dcv implementation see
        https://docs.rapids.ai/deployment/nightly/examples/xgboost-randomforest-gpu-hpo-dask/notebook/
        '''

        if gs_split == 'shuffle':
            cv = StratifiedKFold(n_splits=cv_num, shuffle=True,
                                 random_state=0)
        elif gs_split == 'as_train':
            # merge X and X_test for grid search to have similar dataset
            # split as for training. Normalization was done based on stats from
            # self.X only. Thus when merging can choose cv_num as
            # used for X and X_test split.
            if self.GPU_proc:
                X_inp = cudf.concat(
                    [self.X_test.compute(), self.X.compute()], axis=0)
                # merge labels
                Y_inp = np.hstack([self.Y_test.compute().values.get(),
                                   self.Y.compute().values.get()])
                grid_search_param = {}
            else:
                X_inp = pd.concat([self.X_test, self.X], axis=0)
                # merge labels
                Y_inp = np.hstack([self.Y_test.values, self.Y.values])
                grid_search_param = {'n_jobs': self.n_jobs}

            X_inp_n = X_inp.shape[0]
            # renumber indices
            X_inp.index = range(X_inp_n)

            # provide indices for exact split:
            # With this the first split corresponds to cv00
            # amount test samples per cv (train is n_num -1):
            n_num = int(X_inp_n/cv_num)
            # get indices for test sets
            cv_ind_test = [
                list(range(x*n_num, (x+1)*n_num)) for x in range(cv_num)]
            # indices for train sets
            cv_ind_train = [
                np.setdiff1d(list(range(X_inp_n)), cv_ind_test[x]).tolist()
                for x in range(cv_num)]
            # get list with indices [(train1, test1), (train2, test2), ...]
            # here need to use numpy array instead of list. Otheriwse get error
            cv_ind = [
                (np.array(x), np.array(y))
                for x, y in zip(cv_ind_train, cv_ind_test)]
            cv = cv_ind
        else:
            # this is default input for grid search
            cv = cv_num

        if create_pipeline:
            pipe = make_pipeline(StandardScaler(), self.classifier)
        else:
            pipe = self.classifier

        gclf = dcv.GridSearchCV(
                pipe, self.a_param_gs[self.clf_name],
                refit=refit_inp, cv=cv, scoring=scoring,
                return_train_score=True, **grid_search_param)

        self.text_add.write('\n start grid search fit')
        self.text_add.flush()

        if gs_split == 'as_train':
            gclf.fit(X_inp, Y_inp)
        else:
            gclf.fit(self.X, self.Y.compute().values.get())
            #except:
            #    gclf.fit(self.X, self.Y.values.get())

        self.text_add.write('\n end grid search fit')
        self.text_add.flush()

        path_name = os.path.join(
            self.path_export,
            self.file_prefix + '_grid_search_count' + count_search_str)
        pd.DataFrame(gclf.cv_results_).to_csv(
            path_or_buf=path_name + '.txt', sep='\t',
            lineterminator='\n', header=True)
        return gclf

    def assign_best_estimator(self, test_clf, estimator_type):
        '''
        test_clf can be output from run_grid_search()
        or from run_random_search()

        estimator_type: e.g. either grid_search or randomized search

        '''
        self.classifier = test_clf.best_estimator_

        self.text_add.write(
            f'\n======== search result used as classifier: {estimator_type}\n')
        self.text_add.write(str(test_clf.cv_results_))
        self.text_add.write('\n -- best parameters:\n')
        self.text_add.write(str(test_clf.best_params_))
        self.text_add.write('\n-- best estimator used as classifier \n')
        self.text_add.write(str(test_clf.best_estimator_))
        self.text_add.write('\n-- best score \n')
        self.text_add.write(str(test_clf.best_score_))
        self.text_add.flush()
        return

    def train_all(self):
        self.classifier.fit(self.X, self.Y)
        self.train_count += 1
        self.train_count_str = '{0:02d}'.format(self.train_count)
        return

    def shap_feature_importance_TreeExpl(
            self, X_train_inp):
        '''
        must be run after model.fit() thus self.train_all()
        '''

        n_features = len(self.features_col)

        '''
        # !!! other option but this is very slow
        explainer = shap.KernelExplainer(
            self.classifier.predict_proba, X_train_inp)
        # Calculates the SHAP values - It takes some time
        shap_values = explainer.shap_values(X_test_inp)
        df_shap= pd.DataFrame(
            [np.mean(np.abs(x), axis=0) for x in shap_values],
            columns=self.features_col, index=self.sorted_class_names)
        file_name = os.path.join(
            self.path_export,
            f'{self.file_prefix}_feature_importance_shap_count{self.train_count_str}.txt')
        df_shap.to_csv(
            path_or_buf=file_name, sep='\t', lineterminator='\n',
            header=True)

        shap.summary_plot(
            shap_values, X_test_inp,
            feature_names=self.features_col,
            class_names=self.sorted_class_names, show=False,
            color=self.cmap, max_display=n_features)
        file_name = os.path.join(
            self.path_export,
            f'{self.file_prefix}_feature_importance_shap_count{self.train_count_str}.pdf')
        plt.savefig(file_name, format='pdf')
        plt.close('all')
        '''
        # ------- test tree explainer ---------
        # https://docs.rapids.ai/api/cuml/stable/api/#cuml.explainer.TreeExplainer

        stream_orig = self.classifier.n_streams
        self.classifier.n_streams = 1

        explainer = cu_expl.TreeExplainer(
            model=self.classifier, data=X_train_inp)
        # Calculates the SHAP values - It takes some time
        shap_values = explainer.shap_values(X_train_inp)

        # get true classes (since e.g. for SPOT ther is no snow)
        classes_avail = self.classifier.classes_
        class_name_avail = [
            self.sorted_class_names[self.sorted_class.index(x)]
            for x in classes_avail]

        df_shap= pd.DataFrame(
            [np.mean(np.abs(x), axis=0) for x in shap_values],
            columns=self.features_col, index=class_name_avail)
        file_name = os.path.join(
            self.path_export,
            f'{self.file_prefix}_feature_importance_tree_shap_count{self.train_count_str}.txt')
        df_shap.to_csv(
            path_or_buf=file_name, sep='\t', lineterminator='\n',
            header=True)
        shap_values_lst = [x.get() for x in shap_values]

        # !!! COLORING IS DONE ACCORDING TO HIGHEST IMPORTANCE
        importance_rank = df_shap.sum(axis=1).sort_values(ascending=False).index.tolist()
        color = [self.dict_assign[self.sorted_class[self.sorted_class_names.index(x)]][0]
                 for x in importance_rank]
        cmap_inp = LinearSegmentedColormap.from_list(
            'class', color, N=len(color))
        plt.close('all')
        plt.figure()
        shap.summary_plot(
            shap_values_lst, X_train_inp,
            feature_names=self.features_col,
            class_names=class_name_avail, show=False,
            color=cmap_inp, max_display=n_features)
        file_name = os.path.join(
            self.path_export,
            f'{self.file_prefix}_feature_importance_tree_shap_count{self.train_count_str}.pdf')
        plt.savefig(file_name, format='pdf')
        plt.close('all')

        self.classifier.n_streams = stream_orig

        return

    def shap_feature_importance(
            self, X_train_inp, X_test_inp):
        '''
        must be run after model.fit() thus self.train_all()
        '''

        stream_orig = self.classifier.n_streams
        self.classifier.n_streams = 1

        # https://docs.rapids.ai/api/cuml/stable/api/#cuml.explainer.PermutationExplainer
        explainer = cu_expl.PermutationExplainer(
            model=self.classifier.predict_proba,  # self.classifier.predict_proba???
            data=X_train_inp, random_state=42)

        # get true classes (since e.g. for SPOT ther is no snow)
        classes_avail = self.classifier.classes_
        class_name_avail = [
            self.sorted_class_names[self.sorted_class.index(x)]
            for x in classes_avail]

        # Calculates the SHAP values - It takes some time
        shap_values = explainer.shap_values(X_test_inp)
        df_shap= pd.DataFrame(
            [np.mean(np.abs(x), axis=0) for x in shap_values],
            columns=self.features_col, index=class_name_avail)
        file_name = os.path.join(
            self.path_export,
            f'{self.file_prefix}_feature_importance_shap_count{self.train_count_str}.txt')
        df_shap.to_csv(
            path_or_buf=file_name, sep='\t', lineterminator='\n',
            header=True)

        # !!! COLORING IS DONE ACCORDING TO HIGHEST IMPORTANCE
        importance_rank = df_shap.sum(axis=1).sort_values(ascending=False).index.tolist()
        color = [self.dict_assign[self.sorted_class[self.sorted_class_names.index(x)]][0]
                 for x in importance_rank]
        cmap_inp = LinearSegmentedColormap.from_list(
            'class', color, N=len(color))

        n_features = len(self.features_col)
        plt.close('all')
        plt.figure()
        shap.summary_plot(
            shap_values, X_test_inp,  # self.X_test
            feature_names=self.features_col,
            class_names=class_name_avail, show=False,
            color=cmap_inp, max_display=n_features)
        file_name = os.path.join(
            self.path_export,
            f'{self.file_prefix}_feature_importance_shap_count{self.train_count_str}.pdf')
        plt.savefig(file_name, format='pdf')
        plt.close('all')

        self.classifier.n_streams = stream_orig
        return

    def feature_importance_sklearn(
            self, X_test, Y_test, scoring=None, feature_name=None,
            suffix=''):

        if feature_name is not None:
            X_test_inp = X_test.loc[:, feature_name].to_pandas()
            suffix = '_' + suffix
            clf = RandomForestClassifier(
                **self.classifier.get_params())
            clf.fit(self.X.loc[:, feature_name], self.Y)
            feature_col = feature_name
        else:
            X_test_inp = X_test.to_pandas()
            clf = self.classifier
            feature_col = self.features_col

        result = permutation_importance(
            clf, X_test_inp, Y_test.values.get(),
            n_repeats=10, random_state=42, n_jobs=self.n_jobs,
            scoring=scoring)

        if isinstance(scoring, list):
            result_dict = result
            fig_size = [8.3, 11.7*(len(scoring)/3)]
        else:
            result_dict = {}
            if scoring is None:
                scoring = 'default'
            result_dict[scoring] = result
            fig_size = [11.7, 8.3]

        forest_importances_lst = []
        cols = []
        fig, ax = plt.subplots(
            len(result_dict.keys()), 1, figsize=fig_size)
        if len(result_dict.keys()) == 1:
            ax = [ax]
        for e, i_score in enumerate(result_dict.keys()):
            cols.append(i_score)

            forest_importances = pd.DataFrame.from_dict(
                result_dict[i_score], orient='index',
                columns=feature_col).T
            forest_importances.sort_values(
                'importances_mean', inplace=True, ascending=False)

            forest_importances_lst.append(forest_importances)

            forest_importances['importances_mean'].plot.bar(
                yerr=forest_importances['importances_std'], ax=ax[e])
            ax[e].set_title(
                f'Feature importances using permutation on full model based on score: {i_score}')
            ax[e].set_ylabel('Mean accuracy decrease')

        fig.tight_layout()
        file_name = os.path.join(
            self.path_export,
            f'{self.file_prefix}_feature_importance_permutation_count{self.train_count_str}{suffix}')
        fig.savefig(file_name + '.pdf', format='pdf')
        plt.close('all')

        forest_importances_df = pd.concat(
            forest_importances_lst, keys=cols, axis=1)
        forest_importances_df.to_csv(
            path_or_buf=file_name + '.txt', sep='\t',
            lineterminator='\n', header=True)
        return

    def initialize_evaluation_dicts(self, phase_lst):
        '''
        input for phase list is: PARAM['PHASE_NAME']
        '''
        # per class jaccard index
        self.class_score_name_lst = [
            'jacc_class' + '{0:02d}'.format(x) + '_' + y for x, y in zip(
                self.sorted_class, self.sorted_class_names)] + ['count']
        self.class_score_dict = {x: [] for x in phase_lst}
        #{'train': [], 'test': []}

        self.score_name_lst = [
            'acc', 'f1_micro', 'f1_macro', 'f1_w',
            'precision_micro', 'precision_macro', 'precision_w',
            'recall_micro', 'recall_macro', 'recall_w',
            'jacc_micro', 'jacc_macro', 'jacc_w',
            'tm_f1_micro', #'tm_f1_micro_proba',
            'tm_dice_micro', # 'tm_dice_micro_proba',
            'tm_jacc_micro', 'tm_jacc_macro',
            'pipeline', 'count']
        self.score_dict = {x: [] for x in phase_lst}

        self.cm_name_lst = self.sorted_class_names.copy() + ['count']
        self.cm_dict = {x: [] for x in phase_lst}

        # contanes infos of saved models
        self.model_dict = {
            'model_file_name_pkl': [], 'pipeline_file_name_pkl': [],
            'model_file_name_tl': [],
            'path': [], 'count': [], 'model': [], 'model_param': [],
            'pipeline': [], 'pipeline_param': [], 'comment': []}
             #'pred_files_prefix_train': [], 'pred_files_prefix_test': []}
        self.pred_files_df = {x: {} for x in phase_lst}
        return

    def predict_all(self, df_xy_all, df_xy, phase='validate'):

        # --- run prediction on self.X or self.X_test (depending on phase)
        # get probabilities
        y_pred_probab = self.predict_proba(phase)
        # get predicted classes
        y_pred = self.predict_evaluate(phase)
        # process logging
        self.text_add.write('\n---- predict --- \n')
        self.text_add.write(str(self.classifier))
        self.text_add.write('\n')
        self.text_add.flush()

        # check classes present in file
        # this is only iportant if not all classes occur in dataset
        classes_avail = self.classifier.classes_
        class_nume_pred = [
            self.sorted_class_names[self.sorted_class.index(x)]
            for x in classes_avail]
        col = [x + '_proba' for x in class_nume_pred] + ['class_pred']

        y_pred_full = proc_ML_output(
            df_xy_all.set_index(['aoi_key', 'x', 'y'], drop=False),
            df_xy.set_index(['aoi_key', 'x', 'y'], drop=False),
            np.hstack([y_pred_probab, y_pred[:, np.newaxis]]), col).reset_index(drop=True)

        #pred_files_lst = []
        self.pred_files_df[phase][self.train_count] = {}
        for i_aoi in y_pred_full.aoi_key.unique():
            file_name = (
                i_aoi + '_' + self.file_prefix
                + '_count' + self.train_count_str + '_' + phase
                + '_proba_pred')
            gdf_to_img(
                y_pred_full.query('aoi_key == @i_aoi'), col[:-1],
                self.path_export, file_name, self.EPSG_target,
                nodata=np.nan)
            file_name = (
                i_aoi + '_' + self.file_prefix
                + '_count' + self.train_count_str + '_' + phase
                + '_class_pred')
            self.pred_files_df[phase][self.train_count][i_aoi] = [file_name, file_name[:-5]]
            # pred_files_lst.append(f'{i_aoi}:{file_name}')
            gdf_to_img(
                y_pred_full.query('aoi_key == @i_aoi'), col[-1:],
                self.path_export, file_name, self.EPSG_target,
                nodata=np.nan)
        #self.model_dict['pred_files_prefix_' + phase].append(
        #    ', '.join(pred_files_lst))
        return

    def predict_proba(self, phase='test'):
        '''
        predict probabilities
        '''
        if phase == 'train':
            X_inp = self.X
        else:
            X_inp = self.X_test

        if hasattr(self.classifier, "predict_proba"):
            predict_probab = self.classifier.predict_proba(X_inp)
        else:
            predict_probab = self.classifier.decision_function(X_inp)
        return predict_probab

    def predict_evaluate(self, phase='validate', plot_c_matrix=True):
        '''
        check accuracy and precision
        '''
        if phase == 'train':
            X_inp = self.X
            Y_inp = self.Y
        else:
            X_inp = self.X_test
            Y_inp = self.Y_test

        y_pred = self.classifier.predict(X_inp)
        y_pred_proba = self.classifier.predict_proba(X_inp)

        if self.GPU_proc:
            try:
                y_true_pd = Y_inp.compute().to_pandas()
            except:
                y_true_pd = Y_inp.to_pandas()
        else:
            y_true_pd = Y_inp

        text = f'\n------ classification report {phase} WITH classes defined----\n'
        text += sk_metrics.classification_report(
            y_true_pd, y_pred, labels=self.sorted_class,
            target_names=self.sorted_class_names)
        text += ' \n'
        text += f'\n------ classification report {phase} WITHOUT classes defined----\n'
        text += sk_metrics.classification_report(
            y_true_pd, y_pred)
        text += ' \n'

        # calculate different scores':
        # output shape [1 x n_classes]
        # here add sorted_class as input (class without label is set to 0)
        jacc = sk_metrics.jaccard_score(
            y_true_pd, y_pred, average=None, labels=self.sorted_class)
        self.class_score_dict[phase].append(
            np.hstack([jacc, [self.train_count]]))

        # ouput shape is one number
        score_lst = []
        score_lst.append(
            sk_metrics.accuracy_score(y_true_pd, y_pred))
        score_lst.append(
            sk_metrics.f1_score(y_true_pd, y_pred, average='micro'))
        score_lst.append(
            sk_metrics.f1_score(y_true_pd, y_pred, average='macro'))
        score_lst.append(
            sk_metrics.f1_score(
                y_true_pd, y_pred, average='weighted'))

        score_lst.append(
            sk_metrics.precision_score(
                y_true_pd, y_pred, average='micro'))
        score_lst.append(
            sk_metrics.precision_score(
                y_true_pd, y_pred, average='macro'))
        score_lst.append(
            sk_metrics.precision_score(
                y_true_pd, y_pred, average='weighted'))

        score_lst.append(
            sk_metrics.recall_score(
                y_true_pd, y_pred, average='micro'))
        score_lst.append(
            sk_metrics.recall_score(
                y_true_pd, y_pred, average='macro'))
        score_lst.append(
            sk_metrics.recall_score(
                y_true_pd, y_pred, average='weighted'))

        score_lst.append(
            sk_metrics.jaccard_score(
                y_true_pd, y_pred, average='micro'))
        score_lst.append(
            sk_metrics.jaccard_score(
                y_true_pd, y_pred, average='macro'))
        score_lst.append(
            sk_metrics.jaccard_score(
                y_true_pd, y_pred, average='weighted'))

        # add comparison to torchmetrics
        # f1 micro
        #n_classes_tm_inp = y_pred_proba.shape[1]
        tm_f1_micro = torchmetrics.F1Score(
            task="multiclass", num_classes=self.n_classes,
            average='micro')
        score_lst.append(
            tm_f1_micro(torch.from_numpy(y_pred - 1),
                         torch.from_numpy(y_true_pd.values - 1)))

        # f1 micro based on proba
        #tm_f1_micro_proba = torchmetrics.F1Score(
        #    task="multiclass", num_classes=n_classes_tm_inp,
        #    average='micro')
        #score_lst.append(
        #    tm_f1_micro_proba(torch.from_numpy(y_pred_proba),
        #                      torch.from_numpy(y_true_pd.values - 1)))  # for proba need to minus 1 !!!

        # dice micro
        tm_dice_micro = torchmetrics.Dice(
            num_classes=self.n_classes, average='micro')
        score_lst.append(
            tm_dice_micro(torch.from_numpy(y_pred - 1),
                          torch.from_numpy(y_true_pd.values - 1)))
        # dice micro based on proba
        #tm_dice_micro_proba = torchmetrics.Dice(
        #    num_classes=n_classes_tm_inp, average='micro')
        #score_lst.append(
        #    tm_dice_micro_proba(torch.from_numpy(y_pred_proba),
        #                        torch.from_numpy(y_true_pd.values - 1)))

        tm_jacc = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.n_classes,
            average='micro')
        score_lst.append(
            tm_jacc(torch.from_numpy(y_pred - 1),
                    torch.from_numpy(y_true_pd.values -1)))

        tm_jacc_macro = torchmetrics.JaccardIndex(
                task="multiclass", num_classes=self.n_classes,
                average='macro')
        score_lst.append(
            tm_jacc_macro(torch.from_numpy(y_pred - 1),
                          torch.from_numpy(y_true_pd.values -1)))

        score_lst.append(str(self.classifier))
        score_lst.append(self.train_count)
        self.score_dict[phase].append(score_lst)

        # ouput is numpy array [n_class x n_class]
        cm = sk_metrics.confusion_matrix(
            y_true_pd, y_pred, labels=self.sorted_class)
        self.cm_dict[phase].append(
            np.hstack([cm, np.array([self.train_count]*self.n_classes)[:, np.newaxis]]))
        #if plot_c_matrix:
        #    plot_confusion_matrix(
        #        confusion_m, self.sorted_class, self.path_export,
        #        self.proc_header,
        #        self.proj_name + '_' + self.proc_header_save,
        #        self.dict_assign)

        self.text_add.write(text)
        self.text_add.flush()

        if plot_c_matrix:
            df_plot = pd.DataFrame(cm,
                                   index=self.sorted_class_names,
                                   columns=self.sorted_class_names)
            df_plot_n_true = metrics_plotting_utils.get_cm_ratio(df_plot, axis=1)
            df_plot_n_pred = metrics_plotting_utils.get_cm_ratio(df_plot, axis=0)
            title1 = f'counts, epoch: {self.train_count}'
            title2 = f'normalized over true, count:  {self.train_count}'
            title3 = f'normalized over predicted, count:  {self.train_count}'
            metrics_plotting_utils.plot_cm_matrix(df_plot, df_plot_n_true, df_plot_n_pred,
                   self.path_export,
                   f'{self.file_prefix}_{phase}_cm_count{self.train_count_str}.pdf',
                   title1, title2, title3)
        return y_pred

    def summarize_save_metrics(self, out_suffix=''):

        if out_suffix != '':
            out_suffix = f'_{out_suffix}'

        # create confusion matrix data frame
        cm_d = {}
        score_d = {}
        class_score_d = {}
        sum_lst = [x for x, y in self.class_score_dict.items() if len(y) > 0]
        for i in sum_lst:
            class_score_d[i] = pd.DataFrame(
                np.vstack(self.class_score_dict[i]),
                columns=self.class_score_name_lst)
            class_score_d[i].set_index('count', inplace=True)

            # scores
            score_d[i] = pd.DataFrame(
                self.score_dict[i], columns=self.score_name_lst)
            score_d[i].set_index('count', inplace=True)

            # confusion matrix
            n_runs = len(self.cm_dict[i])
            cm_d[i] = pd.DataFrame(
                np.vstack(self.cm_dict[i]),
                columns=self.cm_name_lst,
                index=self.cm_name_lst[:-1]*n_runs)
            cm_d[i].index.name = 'class'
            cm_d[i] = cm_d[i].reset_index().set_index(['count', 'class'])

        class_score_df = pd.concat(class_score_d, axis=1)
        score_df = pd.concat(score_d, axis=1)
        cm_df = pd.concat(cm_d, axis=1)

        # model metadata
        model_df = pd.DataFrame.from_dict(
            self.model_dict)

        # save metrics as txt file
        path_file = os.path.join(
            self.path_export,
            f'{self.file_prefix}_class_IoU{out_suffix}.txt')
        class_score_df.to_csv(
            path_or_buf=path_file, sep='\t', lineterminator='\n',
            header=True)
        path_file = os.path.join(
            self.path_export,
            f'{self.file_prefix}_scores{out_suffix}.txt')
        score_df.to_csv(
            path_or_buf=path_file, sep='\t', lineterminator='\n',
            header=True)
        path_file = os.path.join(
            self.path_export,
            f'{self.file_prefix}_cm{out_suffix}.txt')
        cm_df.to_csv(
            path_or_buf=path_file, sep='\t', lineterminator='\n',
            header=True)

        # save model metadata as text file
        if model_df.shape[0] > 0:
            path_file = os.path.join(
                self.path_export,
                f'{self.file_prefix}_model_meta{out_suffix}.txt')
            model_df.to_csv(
                path_or_buf=path_file, sep='\t', lineterminator='\n',
                header=True)
        return

    def save_model_pkl(self, comment_inp):
        ''''''
        file_name = (
            f'{self.file_prefix}_model_classifier_count{self.train_count_str}.pkl')
        file_path = os.path.join(self.path_export, file_name)
        with open(file_path,'wb') as f:
            pickle.dump(self.classifier, f)
        self.model_dict['model_file_name_pkl'].append(file_name)

        file_name = (
            f'{self.file_prefix}_model_pipeline_count{self.train_count_str}.pkl')
        file_path = os.path.join(self.path_export, file_name)
        with open(file_path,'wb') as f:
            pickle.dump(self.pipeline, f)
        self.model_dict['pipeline_file_name_pkl'].append(file_name)
        # parameters of model can be checked with
        # tt.get_params()

        self.model_dict['path'].append(file_path)
        self.model_dict['count'].append(self.train_count)
        self.model_dict['model'].append(str(self.classifier))
        self.model_dict['model_param'].append(
            self.classifier.get_params())
        self.model_dict['pipeline'].append(str(self.pipeline))
        self.model_dict['pipeline_param'].append(
            self.pipeline.get_params())
        self.model_dict['comment'].append(comment_inp)
        return

    def save_model_cpu(self):
        '''
        see:
        https://docs.rapids.ai/api/cuml/stable/pickling_cuml_models/#Exporting-cuML-Random-Forest-models-for-inferencing-on-machines-without-GPUs
        '''
        file_name = (
            f'{self.file_prefix}_model_classifier_cpu_count{self.train_count_str}.tl')
        file_path = os.path.join(self.path_export, file_name)
        # Export cuML RF model as Treelite checkpoint
        if self.GPU_proc:
            self.classifier.convert_to_treelite_model().to_treelite_checkpoint(file_path)
        else:
            treelite.sklearn.import_model(self.classifier).serialize(file_path)
        self.model_dict['model_file_name_tl'].append(file_name)
        # !!! saving pipeline does not work !!!!
        return

    def load_model_pkl(self, count=None):

        ''''''
        if count is None:
            self.train_count += 1
            self.train_count_str = '{0:02d}'.format(
                self.train_count)
            count_str = self.train_count_str
        else:
            count_str = '{0:02d}'.format(count)
            self.train_count_str = count_str
            self.train_count = count

        file_name = (
            f'{self.file_prefix}_model_classifier_count{count_str}.pkl')
        file_path = os.path.join(self.path_export, file_name)
        with open(file_path, 'rb') as f:
            self.classifier = pickle.load(f)

        file_name = (
            f'{self.file_prefix}_model_pipeline_count{count_str}.pkl')
        file_path = os.path.join(self.path_export, file_name)
        with open(file_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # but !!! would need to preprocess data beforehand !!!
        # can then use:
        # self.classifier.predict(self.X_test)
        return

    def load_model_tl(self, count=None):
        ''''''
        if count is None:
            count_str = self.train_count_str
        else:
            count_str = '{0:02d}'.format(count)

        file_name = (
            f'{self.file_prefix}_model_classifier_cpu_count{self.train_count_str}.tl')
        file_path = os.path.join(self.path_export, file_name)
        tl_model = treelite.Model.deserialize(file_path)

        # to predict can do following
        # out_prob = treelite.gtil.predict(
        #    tl_model, self.X.to_pandas().values, pred_margin=True)
        return tl_model

    def run_full_hyper_tuning_wf(self, param_dict):
        ''' this runs all required steps for hyprtparameter tuning apart
        from preprrocessing this should not be prarallelized since summary
        dataframes are shared one should better create a new class instance
        for parallization. However this might not be required since grid
        search etc are alrady higly parallelized. Main advantage of
        parallelization would be due to shap feature importances.

        param dict is set in parameter file e.g.
        param_dict = {
            'tune_type': 'grid_search',  # 'grid_search' or 'randomized_search'
            'scoring': ['recall_macro', 'jaccard_macro', 'f1_macro'],  # can be None, single string or list
            'refit_inp': 'jaccard_macro',  # or can be True (if scoring is not list)
            'shuffle_split': True,
            'shap_sample_num': 100,
            }

        # !!! preprocessing is done outsede since otherwise wil be readjusted !!!


        '''
        self.text_add.write('\n----- start new workflow run with following parameters ------\n')
        self.text_add.write(str(param_dict) + '\n')
        self.text_add.flush()

        # ------ resetup classifier ------
        self.set_up_classification_algorithms()

        # ------ hyperparam tuing ------
        count_inp = '{0:02d}'.format(self.train_count + 1)
        start_time = time.time()
        if param_dict['tune_type'] == 'grid_search':
            model_test = self.run_grid_search(
                count_inp, **param_dict)
            time_txt = '{:0.3f}'.format(time.time() - start_time)
            self.text_add.write('\n -- sec for hyperparam tune: ' + time_txt + '\n')
            self.text_add.flush()

            # ------ assign best param ------
            self.assign_best_estimator(model_test, param_dict['tune_type'])

        # ------ retrain all ------
        start_time = time.time()
        self.train_all()
        time_txt = '{:0.3f}'.format(time.time() - start_time)
        self.text_add.write('\n -- sec for re-training: ' + time_txt + '\n')
        self.text_add.flush()

        self.save_model_pkl(str(param_dict))
        self.save_model_cpu()

        return

    def resetup_with_param(self, param_set):
        '''
        resetup classifier with just parameters
        param_set e.g.
        self.classifier.get_params()
        '''
        self.set_classification_param(param_set)
        self.set_up_classification_algorithms()

        return

    def run_feature_importance(self, param_dict, selected_feature_names):
        '''
        need to have been readin trained model before or
        or run run_full_hyper_tuning_wf()
        '''
        # reduce amount of sampes for feature importance
        X_train_inp, X_test_inp, Y_train_inp, Y_test_inp = train_test_split(
            self.X.compute(), self.Y.compute(),
            train_size=param_dict['shap_sample_num'][0],
            test_size=param_dict['shap_sample_num'][1],
            random_state=42, stratify=self.Y.compute(), shuffle=True)

        # ------- check feature importance -------
        '''
        if torch.cuda.is_available():
            start_time = time.time()
            self.shap_feature_importance(X_train_inp, X_test_inp)
            time_txt = '{:0.3f}'.format(time.time() - start_time)
            self.text_add.write('\n -- sec for feature importance test: ' + time_txt + '\n')
            self.text_add.flush()
        '''
        if torch.cuda.is_available():
            start_time = time.time()
            self.shap_feature_importance_TreeExpl(X_train_inp)
            time_txt = '{:0.3f}'.format(time.time() - start_time)
            self.text_add.write(
                '\n -- sec for feature importance test 1: ' + time_txt + '\n')
            self.text_add.flush()

        start_time = time.time()
        self.feature_importance_sklearn(
            X_test_inp, Y_test_inp, scoring=param_dict['scoring'])
        time_txt = '{:0.3f}'.format(time.time() - start_time)
        self.text_add.write(
            '\n -- sec for feature importance sklear: ' + time_txt + '\n')
        self.text_add.flush()

        for e_feat, i_feat in enumerate(selected_feature_names):
            # get index number
            feature_name = [
                self.features_col[self.features_col_orig.index(x)]
                 for x in i_feat]
            self.feature_importance_sklearn(
                X_test_inp, Y_test_inp, scoring=param_dict['scoring'],
                feature_name=feature_name,
                suffix='set_' + '{0:02d}'.format(e_feat))
        return

    def run_full_prediction(
            self, df_xy_all_train, df_xy_train,
            df_xy_all_test, df_xy_test,
            train_phase_name='train',
            test_phase_name='validate'):
        # ----------- Predictions -----------
        start_time = time.time()
        # --- predict train ---
        self.predict_all(
            df_xy_all_train, df_xy_train, phase=train_phase_name)
        # --- predict train ---
        self.predict_all(
            df_xy_all_test, df_xy_test, phase=test_phase_name)
        time_txt = '{:0.3f}'.format(time.time() - start_time)
        self.text_add.write('\n -- sec for predicting: ' + time_txt + '\n')
        self.text_add.flush()

        # -------- save metrics and info --------
        #self.summarize_save_metrics()
        return


# --- subfunctions
def proc_ML_output(full_df, df_ind_inp, arr_inp, new_col_name):
    '''
    Adds the prediction output to the original DataFrame (full_df,
    where NaN values are still included). The output (out_df) can then be
    used to create a GeoTiff containing the predictions (with gdf_to_img()).

    Parameters
    ----------
    full_df : pd.DataFrame
        The full dataframe containing the indices (x, y, aoi_key)
        of the full dataset/image (no deleted NaNs).
    df_ind_inp : pd.DataFrame
        The index dataframe containing the indices (e.g. x, y, aoi_key)
        which were used in the training/prediction (thus after nans were dropped).
    arr_inp : np.ndarray or cupy.ndarray (!!! with row order corresponding
        to df_ind_inp) and columns corresponding to the output (e.g.
        probabilities, predicted classes).
    new_col_name : str
        The name of the new columns (from arr_inp) to be added to the
        index dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe with a value or nan for each x, y, aoi_key of the original
        GeoTiff.

    '''
    df_ind = df_ind_inp.copy()
    # assign ouput (predictions) to the coordinates
    df_ind.loc[:, new_col_name] = arr_inp
    # concatenate the predictions to the original DataFrame (before NaNs
    # were dropped)
    out_df = pd.concat(
        [full_df, df_ind.loc[:, new_col_name]], axis=1)

    return out_df


def gdf_to_img(
        gdf_inp, col_export, path_export, file_name,
        EPSG_target, nodata=np.nan):
    '''
    convert cudf DataFrame to image
    e.g. from clustering output

    img_name: e.g. standardised, or pca
    '''
    out_lst = ['y', 'x'] + col_export

    df_out = gdf_inp.loc[:, out_lst].set_index(['y', 'x'])
    attrs = {'AREA_OR_POINT': 'Area',
             'scale_factor': 1.0, 'add_offset': 0.0}

    img = conversion_utils.df_to_img(df_out, EPSG_target, attrs=attrs,
                    nodata=nodata)

    path_file = os.path.join(path_export, file_name + '.tif')
    img.rio.to_raster(path_file, driver='GTiff')
    return


def prepare_df_sep(df_inp, features_col, class_col, fmt_df=True,
                   GPU_proc=True, df_shuffle=True):
    '''
    Prepare a DataFrame for classification
    get features (X) and categories (y)
    Extracted are the following:
    - A DataFrame with the pixel coordinates (x, y) and the aoi_key.
       This is required later in proc_ML_output() to be able to assign
       the ouput to the correct pixel since the nans are dropped here.
       In addition to x, y the aoi_key (e.g. 'BLyaE_HEX1979_A01train-02')
       is required to correctly assign imagery from differeny years.
    - The input features and labels for training

    Parameters
    ----------
    df_inp : pandas.DataFrame
        Input DataFrame containing the pixel coordinates x and y, the
        features and class labels.
    features_col : list of str
        Column name(s) of the features.
    class_col : str or list of str
        Column name(s) of the class labels.
    fmt_df : bool, optional (default=True)
        Whether to format the output as a list of DataFrames (if True)
        or as a list of arrays (if False).
    GPU_proc : bool, optional (default=False)
        Whether to use GPU processing (requires cupy and cudf libraries).
    df_shuffle : bool, optional (default=True)
        Whether to shuffle the DataFrame before processing.

    Returns
    -------
    list
        A list containing:
            1. A DataFrame with the pixel coordinates (x, y), and the aoi_key
            2. A DataFrame or array with the features (X)
            3. A Series or array with the class labels (y)
    """
    '''
    if not isinstance(class_col, list):
        class_col = [class_col]

    col_all = (
        np.array([features_col]).ravel().tolist()
        + np.setdiff1d(class_col, features_col).tolist())
    # df_index = np.squeeze(df_inp.index.names).tolist()
    df = df_inp.reset_index().dropna(
        subset=col_all, axis=0, how='any').copy()
    # df_out = df.set_index(df_index)
    df.index.name = 'id'
    if df_shuffle:
        df = sk_utils.shuffle(df, random_state=42)
    if fmt_df:
        if GPU_proc:
            out = [df.loc[:, ['x', 'y', 'aoi_key']],
                   cudf.DataFrame.from_pandas(
                       df[features_col].reset_index(drop=True)),
                   cudf.Series.from_pandas(
                       df[class_col].reset_index(drop=True).squeeze())
                   ]
        else:
            out = [df.loc[:, ['x', 'y', 'aoi_key']],
                   df[features_col].reset_index(drop=True),
                   df[class_col].reset_index(drop=True).squeeze()]
    else:
        if GPU_proc:
            out = [df.loc[:, ['x', 'y', 'aoi_key']],
                   cupy.array(df[features_col].values),
                   cupy.array(df[class_col].values.ravel())]
        else:
            out = [df.loc[:, ['x', 'y', 'aoi_key']],
                   df[features_col].values,
                   df[class_col].values.ravel()]
    return out


def get_feature_correlation(X, path_name_export, feature_lst,
                            GPU_proc=True):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    if GPU_proc:
        corr = spearmanr(X.values.get()).correlation
    else:
        corr = spearmanr(X.values).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=feature_lst, ax=ax1, leaf_rotation=90)
    ax1.grid(True)
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    cm = ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.colorbar(cm, ax=ax2)
    _ = fig.tight_layout()
    fig.savefig(path_name_export + '.pdf', format='pdf')
    plt.close('all')
    return dist_linkage


def extract_correlated_groups(
        dist_linkage, feature_names, correl_threshold=1,
        path_export=None):
    cluster_ids = hierarchy.fcluster(
        dist_linkage, correl_threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    feature_groups = [v for v in cluster_id_to_feature_ids.values()]
    selected_features_names = [
        [feature_names[vv] for vv in v] for v in feature_groups]

    if path_export is not None:
        with open(path_export, 'w', errors='ignore') as file_out:
            txt = '\n'.join([str(x) for x in selected_features_names])
            file_out.write(txt)

    return selected_features_names


def distribute(X_inp, y_inp, n_partitions, client, workers, use_GPU=True):
    '''
    n_partitions shpould be n_workers
    https://github.com/rapidsai/cuml/blob/main/notebooks/random_forest_mnmg_demo.ipynb
    '''
    if use_GPU:
        # Partition with Dask
        # In this case, each worker will train on 1/n_partitions fraction of the data
        X_dask = dask_cudf.from_cudf(X_inp, npartitions=n_partitions)
        y_dask = dask_cudf.from_cudf(y_inp, npartitions=n_partitions)
        # Persist to cache the data in active memory
        X_dask, y_dask = \
            dask_utils.persist_across_workers(client, [X_dask, y_dask],
                                            workers=workers)
    else:
        X_dask = dask_df.from_pandas(X_inp, npartitions=n_partitions)
        y_dask = dask_array.from_array(y_inp, chunks=n_partitions)
        X_dask, y_dask = client.persist([X_dask, y_dask])

    return X_dask, y_dask
