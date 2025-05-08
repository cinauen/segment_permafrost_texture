'''
Functions to create plots comparing different segmentation frameworks

TODO: merge very similar functions to read DL and ML metrics into one
'''



import os
import numpy as np
import pandas as pd
import datetime as dt
import glob
import xarray
import geopandas
from rasterio.enums import Resampling

import seaborn as sns
import rioxarray
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcdefaults() # restore [global] defaults
# adjust fonttype for pdfs
matplotlib.rcParams['pdf.fonttype'] = 42

# ---- import specific utils ----
import utils.plotting_utils as plotting_utils
import utils.geo_utils as geo_utils


# --------------- extract metrics comparison --------------
def get_results_folder_DL(folder_df, exclude_folder,
                          SITES_DL, PATH, PATH_EXPORT,
                          old_naming=False,
                          merge_id_prefix='v', train_id_prefix='t'):
    '''
    ## ======= Find results folders DL ==========

    folder_df: pandas DataFrame
        dictionary with the list if folders/DL Fraework to be analysed
        plot_param_sets.extract_folder_from_param(site_sets_dict)

    exclude_folder: list
        list with folders to be excluded from analysis

    SITES_DL: dictionary
        Dictionary relating the id_name (from site_sets_dict) of a
        framework set with a specific folder

    PATH: path
        path to 02_train

    PATH_EXPORT
        path to export list

    old_naming: bool
        if old naming type was used

    merge_id_prefix: str
        'v' (for 'v079t16onl')
    train_id_prefix: str
        't' (for 'v079t16onl')
    '''
    # ### ======= Find results folders DL ==========
    TRAIN_PARAM_DL = {}
    TRAIN_INFO_DL = []
    count = 0
    for i_key, i_folder in SITES_DL.items():
        folder_search = os.path.join(PATH, i_folder)
        if folder_df is None:
            folder_found = [
                os.path.basename(
                    os.path.dirname(x)) for x in glob.glob(folder_search + '/*/')]
            TRAIN_PARAM_DL[i_key] = sorted(
                [x for x in folder_found
                if not np.any([x.find(y) > -1 for y in exclude_folder])])
        else:
            if i_key in folder_df.query('seg_type == "DL"').index.get_level_values(level='site'):
                train_dl_inp = folder_df.loc[(i_key, 'DL'), 'folder']
                TRAIN_PARAM_DL[i_key] = check_convert_to_lst(train_dl_inp)
            elif i_key in folder_df.query('seg_type == "ensemble"').index.get_level_values(level='site'):
                train_dl_inp = folder_df.loc[(i_key, 'ensemble'), 'folder']
                TRAIN_PARAM_DL[i_key] = check_convert_to_lst(train_dl_inp)
            else:
                TRAIN_PARAM_DL[i_key] = []

        for ii_folder in TRAIN_PARAM_DL[i_key]:
            folder_split = ii_folder.split('_')

            if folder_search.find('ensemble') == -1:
                # for new_naming:
                file_prefix = folder_split[0] + '_'.join(folder_split[1:])
                ii_folder_search = os.path.join(
                    folder_search, ii_folder, file_prefix + '*_model_save_df.txt')
                glob_found = glob.glob(ii_folder_search)
                if len(glob_found) == 0:
                    file_prefix = '_'.join(folder_split[1:])
                    ii_folder_search = os.path.join(
                        folder_search, ii_folder, file_prefix + '*_model_save_df.txt')
                    glob_found = glob.glob(ii_folder_search)
                if len(glob_found) == 0:
                    file_prefix = '_'.join(folder_split)
                    ii_folder_search = os.path.join(
                        folder_search, ii_folder, file_prefix + '*_model_save_df.txt')
                    glob_found = glob.glob(ii_folder_search)
                if len(glob_found) == 0:
                    continue

                basename = os.path.basename(glob_found[0])
                name_prefix = basename[:basename.find('_model_save_df.txt')]
                name_param = name_prefix[basename.find(file_prefix) + len(file_prefix) + 1:]
                name_param_lst = name_param.split('_')

                if not old_naming:
                    train_info_dict = {
                        'prefix': name_prefix,
                        'site': i_key,
                        'folder': ii_folder, 'index':[count],
                        'seg_type': 'DL',
                        'folder_sep': '_'.join(ii_folder.split('_')[:-1])
                        }

                    ind_t = folder_split[0].find(train_id_prefix)
                    train_info_dict.update({
                        'model': '',
                        'nchannels': '',
                        'n_classes': name_param.split('ncla')[-1],
                        'merge_id': folder_split[0][:ind_t],
                        'train_id': folder_split[0][ind_t:]})
                else:
                    train_info_dict = {
                        'prefix': name_prefix,
                        'site': i_key,
                        'folder': ii_folder, 'index':[count],
                        'seg_type': 'DL',
                        'folder_sep': '_'.join(ii_folder.split('_')[:-1])
                        }

                    train_info_dict.update({
                        'model': '',
                        'nchannels': '',
                        'n_classes': name_param_lst[1],
                        'merge_id': folder_split[0],
                        'train_id': '_'.join(folder_split[1:4])})
            else:
                ii_folder_search = os.path.join(
                    folder_search, '*_model_save_df.txt')
                glob_found = glob.glob(ii_folder_search)

                if len(glob_found) > 0:
                    #name_prefix = os.path.basename(glob_found[0])
                    basename = os.path.basename(glob_found[0])
                    name_prefix = basename[:basename.find('_model_save_df.txt')]
                    #name_param = name_prefix[basename.find(file_prefix) + len(file_prefix) + 1:]
                    #name_param_lst = name_param.split('_')
                else:
                    name_prefix = os.path.basename(folder_search)

                train_info_dict = {
                    'prefix': ii_folder,
                    'site': i_key,
                    'folder': ii_folder, 'index':[count],
                    'folder_sep': ii_folder,
                    'seg_type': 'ensemble'}

                train_info_dict.update({
                    'model': 'ensemble', 'nchannels': '',
                    'n_classes': '', 'merge_id': '', 'train_id': ''})


            TRAIN_INFO_DL.append(pd.DataFrame.from_dict(train_info_dict, orient='columns'))
            count += 1

    TRAIN_INFO_DL = pd.concat(TRAIN_INFO_DL).set_index(
        ['site', 'folder', 'prefix'])
    TRAIN_INFO_DL = TRAIN_INFO_DL.query('folder not in @exclude_folder')
    # adjust index
    TRAIN_INFO_DL.reset_index(inplace=True)
    TRAIN_INFO_DL['CV_NUM'] = TRAIN_INFO_DL.folder.apply(lambda x: x.split('_')[-1] if len(x.split('_')) > 1 else 'cv00')
    #TRAIN_INFO_DL['folder_sep'] = TRAIN_INFO_DL.folder.apply(lambda x: '_'.join(x.split('_')[:-1]))
    TRAIN_INFO_DL.set_index(['site', 'folder_sep', 'CV_NUM', 'folder', 'prefix', 'merge_id', 'train_id'], inplace=True)

    # save trained settings overview
    file_name = f"trained_settings_{dt.datetime.now().strftime('%Y-%m-%d')}.txt"
    TRAIN_INFO_DL.to_csv(
        path_or_buf=os.path.join(PATH_EXPORT, file_name),
        sep='\t', lineterminator='\n', header=True)

    return TRAIN_INFO_DL


def check_convert_to_lst(inp):
    '''
    Checks selection and converts to list
    e.g. after folder_df.loc[(i_key, 'ensemble'), 'folder']
    as this provides as output either single string (in only one) or series
    '''
    return inp.tolist() if isinstance(inp, (pd.Series, list, np.ndarray)) else [inp]


def extract_metrics_train_DL(TRAIN_INFO_DL, PATH, SITES_DL,
                       metrics_lst, join_char_settings=':',
                       metrics_rename=None):
    '''
    Extracts DL tain metrics

    TRAIN_INFO_DLL DataFrame with folder info

    fSITES_DL: dictionary
        Dictionary relating the id_name (from site_sets_dict) of a
        framework set with a specific folder

    PATH: path
        path to 02_train

    metrics_lst list
        metrics to read and concatenate

    '''
    # =========== Read all DL train metrics files
    if metrics_rename is None:
        metrics_rename = {}
    metrics_rename_dict = {'jacc': 'jacc_micro', 'dice': 'dice_micro',
                           'acc': 'acc_micro'}
    metrics_rename_dict.update(metrics_rename)
    classIoU = []
    metrics = []
    cm = []
    model_save = []
    index_keys = []
    # here only use DL from ensemble ther is no "training metrics" as they are
    # merged predoctions
    for i_site, i_folder_sep, i_CV_NUM, i_folder, i_prefix, i_merge_id, i_train_id in TRAIN_INFO_DL.query('seg_type == "DL"').index.tolist():
        # ======= train and validation
        path_folder = os.path.join(PATH, SITES_DL[i_site], i_folder)
        i_split = i_folder_sep.split('_')
        i_folder_sep_name = join_char_settings.join(i_split)
        i_site_settings = f'{i_site}{join_char_settings}{i_folder_sep_name}'
        if 'ensemble' in i_folder_sep_name:
            # ensemble testing has no trainig metrics (because is mixture model)
            continue
        index_keys.append((i_site, i_folder_sep_name, i_site_settings,
                           i_CV_NUM, i_folder, i_merge_id, i_train_id))

        # --- read train loss overall metrics
        file_name = i_prefix + '_summary_trainRun_tm.txt'
        path_read = os.path.join(path_folder, file_name)
        metrics.append(pd.read_csv(
            path_read, delimiter='\t', header=[0, 1], index_col=[0, 1]))

        # --- read train class IoU
        file_name = i_prefix + '_summary_class_IoU_trainRun_tm.txt'
        path_read = os.path.join(path_folder, file_name)
        classIoU.append(pd.read_csv(
            path_read, delimiter='\t', header=[0, 1], index_col=[0, 1]))

        # --- read train cm
        file_name = i_prefix + '_summary_cm_trainRun_tm.txt'
        path_read = os.path.join(path_folder, file_name)
        cm.append(pd.read_csv(
            path_read, delimiter='\t', header=[0, 1],
            index_col=[0, 1, 2]))

        file_name = i_prefix + '_model_save_df.txt'
        path_read = os.path.join(path_folder, file_name)
        model_save.append(pd.read_csv(
            path_read, delimiter='\t', header=[0], index_col=[0]))


    # =========  concatenate to DataFrame
    index_cols = ['site', 'settings', 'site_settings',
                  'cv_num', 'folder', 'merge_id', 'train_id']
    train_metrics_DL_all = pd.concat(
        metrics, keys=index_keys, axis=0,
        names=index_cols).reset_index('count', drop=True)
    train_classIoU_DL = pd.concat(
        classIoU, keys=index_keys, axis=0,
        names=index_cols).reset_index('count', drop=True)
    train_cm_DL = pd.concat(
        cm, keys=index_keys, axis=0,
        names=index_cols).reset_index('count', drop=True)
    train_cm_DL = train_cm_DL.drop(
        [('train', 'nan'), ('validate', 'nan')], axis=1).reset_index(
            'class_name').dropna(subset=[('class_name', '')], axis=0).set_index(
              'class_name', append=True)
    train_model_save_DL = pd.concat(
        model_save, keys=index_keys, axis=0,
        names=index_cols)
    train_metrics_DL = pd.concat(
        [train_metrics_DL_all.drop('add', axis=1, level=0),
         train_classIoU_DL], axis=1)
    train_metrics_DL.columns.names = ['phase', 'metrics']

    # rename to have same metrics names as for ML results
    train_metrics_DL.rename(
        columns=metrics_rename_dict, level='metrics', inplace=True)
    train_metrics_DL = train_metrics_DL.loc[:, (slice(None), metrics_lst)]

    return train_metrics_DL, train_model_save_DL, train_cm_DL, train_classIoU_DL


def extract_metrics_test_DL(
        TRAIN_INFO_DL, epoch_best, model_merge_dict,
        test_dict_inp, PATH, TEST_path, SITES_DL,
        TEST_dict_DL_patch_prefix, metrics_lst,
        class_weight_eval, join_char_settings=':', metrics_rename=None,
        old_naming=False):
    '''
    Extracts DL tain metrics

    TRAIN_INFO_DLL DataFrame with folder info

    fSITES_DL: dictionary
        Dictionary relating the id_name (from site_sets_dict) of a
        framework set with a specific folder

    PATH: path
        path to 02_train

    metrics_lst list
        metrics to read and concatenate

        PARAM['TEST_path']

    '''
    # ## ====== Read DL test results for best metrics and get file names for geom files
    # read all test metrics files
    if metrics_rename is None:
        metrics_rename = {}
    metrics_rename_dict = {'jacc': 'jacc_micro', 'dice': 'dice_micro',
                           'acc': 'acc_micro', 'f1_micro': 'dice_micro'}
    metrics_rename_dict.update(metrics_rename)

    classIoU = []
    classCM = []
    metrics = []
    geom_files = []
    metrics_geom_pred = []
    metrics_geom_true = []
    metrics_TP_FP = []
    index_keys = []


    index_cols = ['site', 'settings', 'site_settings', 'cv_num',
                  'folder', 'merge_id', 'train_id', 'phase']
    for i_site, i_folder_sep, i_CV_NUM, i_folder, i_prefix, i_merge_id, i_train_id, i_seg_type in TRAIN_INFO_DL.set_index('seg_type', append=True).index.tolist():
        # ======= test
        path_folder = os.path.join(PATH, SITES_DL[i_site], i_folder)
        i_split = i_folder_sep.split('_')
        i_folder_sep_name = join_char_settings.join(i_split)
        i_site_settings = f'{i_site}{join_char_settings}{i_folder_sep_name}'

        if 'ensemble' in i_seg_type:
            # for ensemble testing have no trainng data where need to evaluate
            # best epoch
            epoch_sel = None
        else:
            epoch_sel = epoch_best.reset_index(
                'epoch').loc[(i_site, i_folder_sep_name, i_site_settings, i_CV_NUM, i_folder, i_merge_id, i_train_id), 'epoch']

        if i_site in model_merge_dict.keys():
            i_site_index = model_merge_dict[i_site][1]
            i_test_dict_inp = {x: test_dict_inp[x] for x in model_merge_dict[i_site][0]}
            i_site_settings_inp = f'{i_site_index}:{model_merge_dict[i_site][2]}'
        else:
            i_site_index = i_site
            i_test_dict_inp = test_dict_inp
            i_site_settings_inp = i_site_settings

        #_BLyaE_HEX1979_test_summary_test_cummul_tm.txt
        for i_test, i_patch_lst in i_test_dict_inp.items():
            test_folder = i_test

            index_keys.append(
                (i_site_index, i_folder_sep_name, i_site_settings_inp,
                 i_CV_NUM, i_folder, i_merge_id, i_train_id, i_test))
            # --- read train loss overall metrics
            file_name = f'{i_prefix}_summary_{i_test}_cummul_tm.txt'
            path_read = os.path.join(path_folder, test_folder, file_name)
            if len(glob.glob(path_read)) == 0:
                # !!!! work around due to different naming of old and new files!!!
                i_prefix = f"{i_folder.split('_')[0]}{'_'.join(i_prefix.split('_')[:4])}_{'_'.join(i_prefix.split('_')[-2:])}"
                file_name = f'{i_prefix}_summary_{i_test}_cummul_tm.txt'
                path_read = os.path.join(path_folder, test_folder, file_name)

            metrics_r = pd.read_csv(
                path_read, delimiter='\t', header=[0, 1], index_col=[0, 1])
            if epoch_sel is None:
                # for ensemble testing
                epoch_sel = metrics_r.index.get_level_values(level='epoch')[0]
            metrics_r.rename(
                columns={i_test: 'test'}, level=0, inplace=True)
            metrics_r.rename(
                columns=metrics_rename_dict,
                level=1, inplace=True)
            metrics.append(metrics_r.loc[([epoch_sel], [9999]), :])

            # --- read test class IoU
            file_name = f'{i_prefix}_summary_class_IoU_{i_test}_cummul_tm.txt'
            path_read = os.path.join(path_folder, test_folder, file_name)
            classIoU_r = pd.read_csv(
                path_read, delimiter='\t', header=[0, 1], index_col=[0, 1])
            classIoU_r.rename(columns={i_test: 'test'}, level=0, inplace=True)
            classIoU.append(classIoU_r.loc[([epoch_sel], [9999]), :])

            # --- read test cm
            file_name = f'{i_prefix}_summary_cm_{i_test}_cummul_tm.txt'
            path_read = os.path.join(path_folder, test_folder, file_name)
            classCM_r = pd.read_csv(
                path_read, delimiter='\t', header=[0, 1], index_col=[0, 1, 2])
            classCM_r.rename(columns={i_test: 'test'}, level=0, inplace=True)
            classCM.append(classCM_r.loc[([epoch_sel], [9999]), :])

            # --- read geom specs
            for i_patch in i_patch_lst:
                patch_num = i_patch.split('-')[-1]
                # -- read predicition
                if not old_naming:
                    file_name_pred = f"{i_prefix}_{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_ep{epoch_sel:02d}_class_pred"
                else:
                    file_name_pred = f"{i_prefix}_{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_epoch{epoch_sel:02d}_class_pred"
                path_read_pred = os.path.join(
                    path_folder, test_folder, file_name_pred)
                metrics_geom_pred_r = pd.read_csv(
                    path_read_pred + '.txt', delimiter='\t', header=[0], index_col=[0])
                metrics_geom_pred_r['patch'] = i_patch
                metrics_geom_pred_r.loc[:, index_cols] = index_keys[-1]
                metrics_geom_pred.append(metrics_geom_pred_r)

                # -- read true
                if not old_naming:
                    file_name_true = f"{i_prefix}_{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_ep{epoch_sel:02d}_class_true"
                else:
                    file_name_true = f"{i_prefix}_{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_epoch{epoch_sel:02d}_class_true"
                path_read_true = os.path.join(
                    path_folder, test_folder, file_name_true)
                metrics_geom_true_r = pd.read_csv(
                    path_read_true + '.txt', delimiter='\t', header=[0], index_col=[0])
                metrics_geom_true_r['patch'] = i_patch
                metrics_geom_true_r.loc[:, index_cols] = index_keys[-1]
                metrics_geom_true.append(metrics_geom_true_r)

                # get file names of raw input image
                file_name_raw = f"{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_data"
                path_read_raw = os.path.join(
                    TEST_path[i_test], file_name_raw)

                # dataframe with folder and file names
                geom_files_r = pd.DataFrame(
                    [[path_read_pred, file_name_pred, path_read_true, file_name_true,
                    path_read_raw, file_name_raw]],
                    columns=['path_pred', 'file_pred', 'path_true', 'file_true',
                            'path_raw', 'file_raw'])
                geom_files_r['patch'] = i_patch
                geom_files_r['epoch'] = epoch_sel
                geom_files_r.loc[:, index_cols] = index_keys[-1]
                geom_files.append(geom_files_r)

                # read summary of true and false negative
                if not old_naming:
                    file_name = f"{i_prefix}_{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_ep{epoch_sel:02d}_class_TP_FN_class{class_weight_eval}.txt"
                else:
                    file_name = f"{i_prefix}_{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_epoch{epoch_sel:02d}_class_TP_FN.txt"
                path_read = os.path.join(path_folder, test_folder, file_name)
                metrics_TP_FP_r = pd.read_csv(
                    path_read, delimiter='\t', header=[0], index_col=[0])
                metrics_TP_FP_r['patch'] = i_patch
                metrics_TP_FP_r.loc[:, index_cols] = index_keys[-1]
                metrics_TP_FP.append(metrics_TP_FP_r)


    test_classIoU_DL = pd.concat(
        classIoU, keys=index_keys, axis=0, names=index_cols).reset_index(
            'count', drop=True)
    test_classCM_DL = pd.concat(
        classCM, keys=index_keys, axis=0, names=index_cols).reset_index(
            'count', drop=True)
    test_classCM_DL = test_classCM_DL.drop(('test', 'nan'), axis=1).reset_index(
        'class_name').dropna(subset=[('class_name', '')], axis=0).set_index(
            'class_name', append=True).reset_index('epoch', drop=True)
    test_metrics_DL_all = pd.concat(
        metrics, keys=index_keys, axis=0, names=index_cols).reset_index(
            'count', drop=True)
    test_metrics_DL = pd.concat(
        [test_metrics_DL_all, test_classIoU_DL], axis=1).loc[:, (slice(None), metrics_lst)]
    test_metrics_DL.columns.names = ['phase', 'metrics']


    test_metrics_geom_pred_DL = pd.concat(
        metrics_geom_pred, axis=0).reset_index().set_index(
            index_cols + ['patch', 'index'])
    test_metrics_geom_true_DL = pd.concat(
        metrics_geom_true, axis=0).reset_index().set_index(
            index_cols + ['patch', 'index'])
    test_geom_files_DL = pd.concat(geom_files, axis=0).reset_index().set_index(
        index_cols + ['patch', 'index'])
    test_metrics_TP_FP_DL = pd.concat(
        metrics_TP_FP, axis=0).reset_index().set_index(
            index_cols + ['patch', 'index'])

    test_metrics_TP_FP_sum_DL = test_metrics_TP_FP_DL.groupby(index_cols).sum()

    # calculate true positives percentage and precision and recall
    df_FN_TN_perc_DL = add_TP_FP_perc(
        test_metrics_TP_FP_sum_DL)
    df_FN_TN_f1_DL = add_recall_precision(
        test_metrics_TP_FP_sum_DL)

    return (test_metrics_DL, test_classCM_DL,
            test_metrics_geom_pred_DL, test_metrics_geom_true_DL,
            test_geom_files_DL, df_FN_TN_perc_DL, df_FN_TN_f1_DL)


def get_results_folder_ML(folder_df, exclude_folder,
                          SITES_ML, PATH, old_naming=False,
                          merge_id_prefix='vML', train_id_prefix='tML'):
    '''
    ## ======= Find results folders ML ==========

    folder_df: pandas DataFrame
        dictionary with the list if folders/ML Fraework to be analysed
        plot_param_sets.extract_folder_from_param(site_sets_dict)

    exclude_folder: list
        list with folders to be excluded from analysis

    SITES_ML: dictionary
        Dictionary relating the id_name (from site_sets_dict) of a
        framework set with a specific folder

    PATH: path
        path to 02_train

    merge_id_prefix: str
        'vML' (for 'vML017tML02')
    train_id_prefix: str
        'tML' (for 'vML017tML02')
    '''
    # ### =========== Find results folders ML
    TRAIN_PARAM_ML = {}
    TRAIN_INFO_ML = []
    count = 0
    search_file_str = '_scores_train_validate.txt'
    for i_key, i_folder in SITES_ML.items():
        folder_search = os.path.join(PATH, i_folder)
        if folder_df is None:
            folder_found = [
                os.path.basename(
                    os.path.dirname(x)) for x in glob.glob(folder_search + '/*/')]
            TRAIN_PARAM_ML[i_key] = sorted(
                [x for x in folder_found
                if not np.any([x.find(y) > -1 for y in exclude_folder])])
        else:
            if i_key in folder_df.query('seg_type == "ML"').index.get_level_values(level='site'):
                train_ml_inp = folder_df.loc[(i_key, 'ML'), 'folder']
                TRAIN_PARAM_ML[i_key] = check_convert_to_lst(train_ml_inp)
            else:
                TRAIN_PARAM_ML[i_key] = []

        for ii_folder in TRAIN_PARAM_ML[i_key]:
            if not old_naming:
                ind_t = ii_folder.find(train_id_prefix)
                file_prefix = ii_folder
                merge_id_inp = ii_folder[:ind_t]
                train_id_inp = ii_folder.split('_')[0][ind_t:]

            else:
                merge_id_inp = folder_split[0]
                train_id_inp = folder_split[1]
                folder_split = ii_folder.split('_')
                file_prefix = folder_split[0] + '_'.join(folder_split[1:])
            ii_folder_search = os.path.join(
                folder_search, ii_folder, file_prefix + '*' + search_file_str)
            glob_found = glob.glob(ii_folder_search)
            if len(glob_found) == 0:
                continue
            for i in glob_found:
                basename = os.path.basename(i)
                dirname = os.path.dirname(i)
                name_prefix = basename[:basename.find(search_file_str)]
                cv_num = name_prefix.split('_')[-1]
                TRAIN_INFO_ML.append(pd.DataFrame.from_dict({
                    'path': dirname,
                    'score_file_name': basename,
                    'prefix': name_prefix,
                    'CV_NUM': cv_num,
                    'merge_id': merge_id_inp,
                    'train_id': train_id_inp, 'site': i_key, 'folder': ii_folder,
                    'folder_sep': '_'.join(ii_folder.split('_')[:-1]),
                    'index':[count],
                    'seg_type': 'ML'}, orient='columns'))
            count += 1

    if len(TRAIN_INFO_ML) > 0:
        TRAIN_INFO_ML = pd.concat(TRAIN_INFO_ML).set_index(
            ['site', 'folder_sep', 'CV_NUM', 'folder',
             'prefix', 'merge_id', 'train_id'])
    else:
        TRAIN_INFO_ML = pd.DataFrame()

    return TRAIN_INFO_ML


def extract_metrics_train_ML(
        TRAIN_INFO_ML, PATH, SITES_ML, metrics_lst,
        metrics_rename=None):
    '''
    Extracts DL tain metrics

    TRAIN_INFO_DLL DataFrame with folder info

    fSITES_DL: dictionary
        Dictionary relating the id_name (from site_sets_dict) of a
        framework set with a specific folder

    PATH: path
        path to 02_train

    metrics_lst list
        metrics to read and concatenate

    '''
    # =========== Read all ML train metrics files
    # ======= read train metrics
    if metrics_rename is None:
        metrics_rename = {}
    metrics_rename_dict = {'jacc': 'jacc_micro', 'dice': 'dice_micro',
                           'acc': 'acc_micro', 'f1_micro': 'dice_micro'}
    metrics_rename_dict.update(metrics_rename)

    classIoU = []
    classCM = []
    metrics = []
    index_keys = []

    count_num = 1
    for i_site, i_folder_sep, i_CV_NUM, i_folder, i_prefix, i_merge_id, i_train_id in TRAIN_INFO_ML.index.tolist():
        # ======= train and validation
        path_folder = os.path.join(PATH, SITES_ML[i_site], i_folder)
        i_split = i_folder_sep.split('_')
        i_folder_sep_name = ':'.join(i_split)  # ['_'.join(i_split[:2]), '_'.join(i_split[2:])])
        i_site_settings = f'{i_site}:{i_folder_sep_name}'
        index_keys.append((i_site, i_folder_sep_name, i_site_settings, i_CV_NUM, i_folder, i_merge_id, i_train_id))

        # --- read train loss overall metrics
        file_name = i_prefix + '_scores_train_validate.txt'
        path_read = os.path.join(path_folder, file_name)
        metrics.append(pd.read_csv(
            path_read, delimiter='\t', header=[0, 1], index_col=[0]).query(
                'count == @count_num'))

        # --- read train class IoU
        file_name = i_prefix + '_class_IoU_train_validate.txt'
        path_read = os.path.join(path_folder, file_name)
        classIoU.append(pd.read_csv(
            path_read, delimiter='\t', header=[0, 1], index_col=[0]).query(
                'count == @count_num'))

        # --- read train cm
        file_name = i_prefix + '_cm_train_validate.txt'
        path_read = os.path.join(path_folder, file_name)
        classCM.append(pd.read_csv(
            path_read, delimiter='\t', header=[0, 1], index_col=[0, 1]).query(
                'count == @count_num'))

    # concatenate
    index_cols = ['site', 'settings', 'site_settings', 'cv_num', 'folder',
                  'merge_id', 'train_id']
    train_metrics_ML_all = pd.concat(
        metrics, keys=index_keys, axis=0, names=index_cols).query('count == 1').reset_index('count', drop=True)
    train_classIoU_ML = pd.concat(
        classIoU, keys=index_keys, axis=0, names=index_cols).query('count == 1').reset_index('count', drop=True)
    train_metrics_ML = pd.concat(
        [train_metrics_ML_all, train_classIoU_ML], axis=1).sort_index(axis=1)
    train_metrics_ML.columns.names = ['phase', 'metrics']
    # !!! rename f1_micro to dice_micro (is same here)
    train_metrics_ML.rename(columns=metrics_rename_dict,
                            level='metrics', inplace=True)
    train_metrics_ML = train_metrics_ML.loc[:, (slice(None), metrics_lst)].sort_index(axis=1)

    train_classCM_ML = pd.concat(
        classCM, keys=index_keys, axis=0, names=index_cols).query('count == 1').reset_index('count', drop=True)
    train_classCM_ML.columns.names = ['phase', 'metrics']
    cm_index = [x if x != 'class' else 'class_name' for x in train_classCM_ML.index.names]
    train_classCM_ML.index.names = cm_index

    return train_metrics_ML, train_classIoU_ML, train_classCM_ML


def extract_metrics_test_ML(
        TRAIN_INFO_ML,
        test_dict_inp, PATH, TEST_path, SITES_ML,
        TEST_dict_DL_patch_prefix, metrics_lst,
        class_weight_eval, metrics_rename=None):
    '''
    Extracts ML test metrics

    TRAIN_INFO_ML DataFrame with folder info

    SITES_ML: dictionary
        Dictionary relating the id_name (from site_sets_dict) of a
        framework set with a specific folder

    PATH: path
        path to 02_train

    metrics_lst list
        metrics to read and concatenate

        PARAM['TEST_path']

    '''
    # ## ====== Read ML test results for best metrics and get file names for geom files
    # read all test metrics files
    if metrics_rename is None:
        metrics_rename = {}
    metrics_rename_dict = {'jacc': 'jacc_micro', 'dice': 'dice_micro',
                           'acc': 'acc_micro', 'f1_micro': 'dice_micro'}
    metrics_rename_dict.update(metrics_rename)

    # ======= read ML TEST metrics
    classIoU = []
    classCM = []
    metrics = []
    index_keys = []
    metrics_geom_pred = []
    metrics_geom_true = []
    metrics_TP_FP = []
    geom_files = []

    count_num = 1
    index_cols = ['site', 'settings', 'site_settings',
                  'cv_num', 'folder', 'merge_id', 'train_id', 'phase']

    for i_site, i_folder_sep, i_CV_NUM, i_folder, i_prefix, i_merge_id, i_train_id in TRAIN_INFO_ML.index.tolist():
        # ======= train and validation
        #path_folder = os.path.join(PATH, PARAM['SITES_ML'][i_site], i_folder)
        i_split = i_folder_sep.split('_')
        i_folder_sep_name = ':'.join(i_split)  # ['_'.join(i_split[:2]), '_'.join(i_split[2:])])
        i_site_settings = f'{i_site}:{i_folder_sep_name}'
        #index_keys.append((i_site, i_folder_sep_name, i_site_settings, i_CV_NUM, i_folder))


        # --- read geom specs
        for i_test, i_patch_lst in test_dict_inp.items():
            path_folder = os.path.join(
                PATH, SITES_ML[i_site], i_folder, i_test)

            index_keys.append((i_site, i_folder_sep_name, i_site_settings,
                               i_CV_NUM, i_folder, i_merge_id, i_train_id,
                               i_test))

            # --- read train loss overall metrics
            file_name = i_prefix + '_scores_train_validate_test.txt'
            path_read = os.path.join(path_folder, file_name)
            metrics_r = pd.read_csv(
                path_read, delimiter='\t', header=[0, 1], index_col=[0]).query(
                    'count == @count_num')
            metrics_r = metrics_r.loc[:, (i_test, slice(None))]
            metrics_r.rename(columns={i_test: 'test'}, level=0, inplace=True)
            metrics_r.rename(
                columns=metrics_rename_dict, level=1, inplace=True)
            metrics.append(metrics_r)

            # --- read train class IoU
            file_name = i_prefix + '_class_IoU_train_validate_test.txt'
            path_read = os.path.join(path_folder, file_name)
            classIoU_r = pd.read_csv(
                path_read, delimiter='\t', header=[0, 1], index_col=[0]).query(
                    'count == @count_num')
            classIoU_r = classIoU_r.loc[:, (i_test, slice(None))]
            classIoU_r.rename(columns={i_test: 'test'}, level=0, inplace=True)
            classIoU.append(classIoU_r)

            # --- read train cm
            file_name = i_prefix + '_cm_train_validate_test.txt'
            path_read = os.path.join(path_folder, file_name)
            classCM_r = pd.read_csv(
                path_read, delimiter='\t', header=[0, 1], index_col=[0, 1]).query(
                    'count == @count_num')
            classCM_r = classCM_r.loc[:, (i_test, slice(None))]
            classCM_r.rename(columns={i_test: 'test'}, level=0, inplace=True)
            classCM.append(classCM_r)


            for i_patch in i_patch_lst:
                patch_num = i_patch.split('-')[-1]
                # -- read predicition
                file_name_pred = f"{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_{i_prefix}_count{count_num:02d}_{i_test}_class_pred"
                path_read_pred = os.path.join(path_folder, file_name_pred)
                metrics_geom_pred_r = pd.read_csv(
                    path_read_pred + '.txt', delimiter='\t', header=[0],
                    index_col=[0])
                metrics_geom_pred_r['patch'] = i_patch
                metrics_geom_pred_r.loc[:, index_cols] = list(index_keys[-1])# + [i_test]
                metrics_geom_pred.append(metrics_geom_pred_r)

                # -- read true
                file_name_true = f"{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_{i_prefix}_count{count_num:02d}_{i_test}_class_true"
                path_read_true = os.path.join(path_folder, file_name_true)
                metrics_geom_true_r = pd.read_csv(
                    path_read_true + '.txt', delimiter='\t', header=[0],
                    index_col=[0])
                metrics_geom_true_r['patch'] = i_patch
                metrics_geom_true_r.loc[:, index_cols] = list(index_keys[-1])# + [i_test]
                metrics_geom_true.append(metrics_geom_true_r)

                # file name of raw image file
                file_name_raw = f"{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_data"
                path_read_raw = os.path.join(
                    TEST_path[i_test], file_name_raw)

                # DataFrame with file info
                geom_files_r = pd.DataFrame(
                    [[path_read_pred, file_name_pred, path_read_true, file_name_true,
                    path_read_raw, file_name_raw]],
                    columns=['path_pred', 'file_pred', 'path_true', 'file_true',
                            'path_raw', 'file_raw'])
                geom_files_r['patch'] = i_patch
                geom_files_r['epoch'] = -1
                geom_files_r.loc[:, index_cols] = list(index_keys[-1])# + [i_test]

                geom_files.append(geom_files_r)

                # -- read summary of true and false negative
                file_name = f"{TEST_dict_DL_patch_prefix[i_test]}-{patch_num}_{i_prefix}_count{count_num:02d}_{i_test}_class_TP_FN_class{class_weight_eval}.txt"
                path_read = os.path.join(path_folder, file_name)
                metrics_TP_FP_r = pd.read_csv(
                    path_read, delimiter='\t', header=[0], index_col=[0])
                metrics_TP_FP_r['patch'] = i_patch
                metrics_TP_FP_r.loc[:, index_cols] = list(index_keys[-1])# + [i_test]
                metrics_TP_FP.append(metrics_TP_FP_r)

    # concatenate
    # merge test metrics
    test_metrics_ML_all = pd.concat(
        metrics, keys=index_keys, axis=0, names=index_cols).query('count == 1').reset_index('count', drop=True)

    test_classIoU_ML = pd.concat(
        classIoU, keys=index_keys, axis=0, names=index_cols).query('count == 1').reset_index('count', drop=True)

    test_metrics_ML = pd.concat(
        [test_metrics_ML_all, test_classIoU_ML], axis=1).sort_index(axis=1)
    test_metrics_ML.columns.names = ['phase', 'metrics']

    # !!! rename f1_micro to dice_micro (is same here)
    test_metrics_ML.rename(columns={'f1_micro': 'dice_micro'},
                            level='metrics', inplace=True)
    test_metrics_ML = test_metrics_ML.loc[:, (slice(None), metrics_lst)].sort_index(axis=1)

    test_classCM_ML = pd.concat(
        classCM, keys=index_keys, axis=0, names=index_cols).query('count == 1').reset_index('count', drop=True)
    test_classCM_ML.columns.names = ['phase', 'metrics']
    cm_index = [x if x != 'class' else 'class_name' for x in test_classCM_ML.index.names]
    test_classCM_ML.index.names = cm_index

    # ==================================
    # merge test geom metrics
    test_metrics_geom_pred_ML = pd.concat(
        metrics_geom_pred, axis=0).reset_index().set_index(
            index_cols + ['patch', 'index'])
    test_metrics_geom_true_ML = pd.concat(
        metrics_geom_true, axis=0).reset_index().set_index(
            index_cols + ['patch', 'index'])

    test_geom_files_ML = pd.concat(
        geom_files, axis=0).reset_index().set_index(
            index_cols + ['patch', 'index'])

    test_metrics_TP_FP_ML = pd.concat(
        metrics_TP_FP, axis=0).reset_index().set_index(
            index_cols + ['patch', 'index'])
    test_metrics_TP_FP_sum_ML = test_metrics_TP_FP_ML.groupby(index_cols).sum()

    # calculate true positives percentage and precision and recall
    df_FN_TN_perc_ML = add_TP_FP_perc(
        test_metrics_TP_FP_sum_ML)
    df_FN_TN_f1_ML = add_recall_precision(
        test_metrics_TP_FP_sum_ML)

    return (test_metrics_ML, test_classCM_ML, test_metrics_geom_pred_ML,
              test_metrics_geom_true_ML, test_geom_files_ML,
              df_FN_TN_perc_ML, df_FN_TN_f1_ML)


def add_TP_FP_perc(df_FN_TN):
    '''
    add percentage True Positive and False negatives
    (per amount of cells of class 1 and per weight)
    '''
    df_FN_TN_perc = df_FN_TN.loc[:, ['px_sum']]
    for i_w in [0, 1, 2]:
        df_FN_TN_perc[f'TP\nW{i_w}'] = (df_FN_TN[f'TP_W{i_w}']/df_FN_TN[f'weight{i_w}_cts_cls1'])*100
        df_FN_TN_perc[f'FN\nW{i_w}'] = (df_FN_TN[f'FN_W{i_w}']/df_FN_TN[f'weight{i_w}_cts_cls1'])*100
    df_FN_TN_perc['TN'] = (df_FN_TN['TN']/df_FN_TN['px_sum_no_cls1'])*100
    df_FN_TN_perc['FP'] = (df_FN_TN['FP']/df_FN_TN['px_sum_no_cls1'])*100
    return df_FN_TN_perc


def add_recall_precision_perc(FN_TN_perc):
    '''
    this does not provide correct precision
    '''
    df_FN_TN_f1 = FN_TN_perc.loc[:, ['px_sum']]
    for i_w in [0, 1, 2]:
        df_FN_TN_f1[f'prec_TP\nW{i_w}'] = FN_TN_perc[f'TP\nW{i_w}']/(FN_TN_perc[f'TP\nW{i_w}'] + FN_TN_perc[f'FP'])
        df_FN_TN_f1[f'recall_TP\nW{i_w}'] = FN_TN_perc[f'TP\nW{i_w}']/(FN_TN_perc[f'TP\nW{i_w}'] + FN_TN_perc[f'FN\nW{i_w}'])
    return df_FN_TN_f1


def add_recall_precision(df_FN_TN):
    '''
    recalss is same as TP perc
    '''
    df_FN_TN_f1 = df_FN_TN.loc[:, ['px_sum']]
    for i_w in [0, 1, 2]:
        df_FN_TN_f1[f'recall_TP\nW{i_w}'] = df_FN_TN[f'TP_W{i_w}']/(df_FN_TN[f'TP_W{i_w}'] + df_FN_TN[f'FN_W{i_w}'])
    TP_sum = np.sum(df_FN_TN[f'TP_W{i_w}'] for i_w in [0, 1, 2])
    df_FN_TN_f1[f'prec_TN'] = TP_sum/(TP_sum + df_FN_TN[f'FP'])
    return df_FN_TN_f1


def get_FP_TP_recall_prec_from_CM(df_cm, class_label_dict_rename):
    '''
    df_cm is dataframe with several CM per different setups etc...
        is grouped pr index and then recall etc are calculated

    class_label_dict_rename: dictionary to rename the classes
    '''

    def get_FP(x_inp):
        '''
        x_inp is single confusion matrix
        '''
        x = x_inp.copy()
        x = x.reset_index('class_name').reset_index(drop=True).set_index('class_name')
        x = x.sort_index(axis=1).sort_index(axis=0)
        TP = x.values.diagonal()
        FN = x.sum(axis=1) - x.values.diagonal()
        FN = FN.to_frame(name='FN')
        #ind_names = FP.index.names.copy()
        #FP = FP.reset_index().set_index('class_name')
        FP = x.sum(axis=0) - x.values.diagonal()
        TN = np.sum(x.values) - x.sum(axis=0) - x.sum(axis=1) + x.values.diagonal()
        FP = FP.to_frame(name='FP')
        FP.index.names = ['class_name']
        FN['FP'] = FP
        FN['TP'] = TP
        FN['TN'] = TN
        FN['prec'] = FN['TP']/(FN['TP'] + FN['FP'])
        FN['recall'] = FN['TP']/(FN['TP'] + FN['FN'])
        #FP = FP.unstack('class_name')
        return FN.unstack('class_name')

    grouped = df_cm.groupby(list(df_cm.index.names)[:-1])
    df_out = grouped.apply(get_FP)
    df_out = df_out.stack('class_name', future_stack=True)
    df_out = df_out.rename(
            class_label_dict_rename, level='class_name', axis=0)
    ind_index = df_out.index.names
    df_out.reset_index(inplace=True)
    df_out['true_pred_set'] = df_out.loc[:, ['site_settings']].apply(
        lambda x: 'pred ' + x['site_settings'].replace(':', '_'), axis=1)
    df_out.set_index(ind_index + ['true_pred_set'], inplace=True)
    # add summed up TP FN and FP as well as percentages for bar plots
    df_out['TRUE'] = df_out.loc[:, ['TP', 'FN']].sum(axis=1)

    # True area plus false predicted area
    df_out['TRUE_FP'] = df_out.loc[:, ['TP', 'FN', 'FP']].sum(axis=1)
    df_out['TRUE_perc'] = 100

    # this is TPR true positive rate (TP/true_area), same as recall
    df_out['TRUE_TP_perc'] = (
        df_out['TP']/df_out['TRUE'])*100

    # this is an unknow measure ((true area + FP)/true_area)
    df_out['TRUE_FP_perc'] = (
        df_out['TRUE_FP']/df_out['TRUE'])*100
    df_out['TRUE_FP_perc_neg'] = (
        100 - df_out['TRUE_FP_perc'])


    df_out['PRED'] = df_out.loc[:, ['TP', 'FP']].sum(axis=1)
    df_out['PRED_FN'] = df_out.loc[:, ['TP', 'FP', 'FN']].sum(axis=1)
    df_out['PRED_perc'] = 100
    df_out['PRED_TP_perc'] = (
        df_out['TP']/df_out['PRED'])*100
    df_out['PRED_FN_perc'] = (
        df_out['PRED_FN']/df_out['PRED'])*100

    # this is false discovery rate (FDR = FP/(FP + TP) = FP/predicted_area)
    df_out['FDR_perc'] = (
        df_out['FP']/df_out['PRED'])*100
    df_out['FDR_perc_neg'] = - df_out['FDR_perc']

    # false positive rate
    df_out['FPR_perc'] = (
        df_out['FP']/(df_out.loc[:, ['FP', 'TN']].sum(axis=1)))*100
    df_out['FPR_perc_neg'] = - df_out['FPR_perc']

    df_out['PRED'] = -df_out['PRED']
    df_out['PRED_FN'] = -df_out['PRED_FN']
    df_out['PRED_perc'] = -df_out['PRED_perc']
    df_out['PRED_TP_perc'] = -df_out['PRED_TP_perc']
    df_out['PRED_FN_perc'] = -df_out['PRED_FN_perc']

    return df_out


def extract_inputs_for_metrics_plots(
        train_metrics_DL, train_metrics_best, test_metrics_all,
        test_TP_FP, test_TP_FP_f1, test_metrics_geom,
        validate_metrics_prec_recall_inp,
        test_metrics_prec_recall_inp, site_settings_index,
        metrics_plot_lst, set_relabel, min_epoch, test_phases):
    '''
    metrics_plot_lst: metrics_plot_lst
            (which metrics should be extracted for plotting)

    set_relabel: dict to rename site_settings_index to shorter labels
        for plot

    min_epoch: above whoch epoch metrics is used for distribution plot
    '''

    # extract input for boxplots resp stripplots
    # for boxplots or distribution plots, can use data from DL only
    # since we hae no distribution for the ML results
    sets = train_metrics_DL.index.get_level_values(level='site_settings').unique()
    site_settings_index_DL = np.intersect1d(site_settings_index, sets)
    box_plot_inp = train_metrics_DL.loc[(slice(None), slice(None), site_settings_index_DL), :].stack('phase', future_stack=True).query('epoch >= @min_epoch')
    box_plot_inp = box_plot_inp.loc[:, metrics_plot_lst].stack('metrics', future_stack=True)
    box_plot_inp = pd.DataFrame(box_plot_inp, columns=['val']).reset_index()
    box_plot_inp['cv_num_phase'] = box_plot_inp.loc[:, ['cv_num', 'phase']].sum(axis=1)

    box_plot_inp['set_label'] = box_plot_inp.loc[:, ['site_settings']].apply(
        lambda x: set_relabel[x['site_settings']], axis=1)

    # extract best epoch input for plots
    # following two lines are required as for example for ensemble run ther
    # if no training metrics
    sets = train_metrics_best.index.get_level_values(level='site_settings').unique()
    site_settings_index_train = np.intersect1d(site_settings_index, sets)
    epoch_best_inp = train_metrics_best.loc[(slice(None), slice(None), site_settings_index_train), metrics_plot_lst].stack('metrics', future_stack=True)
    epoch_best_inp = pd.DataFrame(epoch_best_inp, columns=['val']).reset_index()
    epoch_best_inp['cv_num_phase'] = epoch_best_inp.loc[:, ['cv_num', 'phase']].sum(axis=1)
    epoch_best_inp['set_label'] = epoch_best_inp.loc[:, ['site_settings']].apply(
        lambda x: set_relabel[x['site_settings']], axis=1)

    cols_to_keep = ['prec', 'recall', 'TRUE_TP_perc', 'TRUE_perc',
                    'TRUE_FP_perc', 'FDR_perc_neg',
                    'PRED_TP_perc', 'PRED_perc', 'PRED_FN_perc']
    validate_metrics_prec_recall = validate_metrics_prec_recall_inp.loc[(slice(None), slice(None), site_settings_index_train), cols_to_keep].copy()
    #test_metrics_prec_recall = test_metrics_prec_recall.loc[(slice(None), slice(None), site_settings_index), cols_to_keep]
    ind_names = validate_metrics_prec_recall.index.names
    validate_metrics_prec_recall.reset_index(inplace=True)
    #validate_metrics_prec_recall['cv_num_phase'] = validate_metrics_prec_recall.loc[:, ['cv_num', 'phase']].sum(axis=1)
    validate_metrics_prec_recall['set_label'] = validate_metrics_prec_recall.loc[:, ['site_settings']].apply(
        lambda x: set_relabel[x['site_settings']], axis=1)
    validate_metrics_prec_recall.set_index(list(ind_names) + ['set_label'], inplace=True)
    #test_metrics_prec_recall.columns.names = ['metrics']
    #test_metrics_prec_recall = test_metrics_prec_recall.loc[:, cols_to_keep].stack('metrics', dropna=False).to_frame('val')


    cols_to_keep = ['prec', 'recall']
    test_metrics_prec_recall = test_metrics_prec_recall_inp.loc[(slice(None), slice(None), site_settings_index), cols_to_keep].copy()
    #test_metrics_prec_recall = test_metrics_prec_recall.loc[(slice(None), slice(None), site_settings_index), cols_to_keep]
    ind_names = test_metrics_prec_recall.index.names
    test_metrics_prec_recall.reset_index(inplace=True)
    test_metrics_prec_recall['cv_num_phase'] = test_metrics_prec_recall.loc[:, ['cv_num', 'phase']].sum(axis=1)
    test_metrics_prec_recall['set_label'] = test_metrics_prec_recall.loc[:, ['site_settings']].apply(
        lambda x: set_relabel[x['site_settings']], axis=1)
    test_metrics_prec_recall.set_index(list(ind_names) + ['cv_num_phase'] + ['set_label'], inplace=True)
    test_metrics_prec_recall.columns.names = ['metrics']
    test_metrics_prec_recall = test_metrics_prec_recall.loc[:, cols_to_keep].stack('metrics', future_stack=True).to_frame('val')  # dropna=False,

    # extract test results as input for plot
    #test_plot_inp = test_metrics_DL.loc[(site_index, settings_index), :].droplevel(axis=1, level='phase')
    test_plot_inp = test_metrics_all.loc[(slice(None), slice(None), site_settings_index), metrics_plot_lst].stack('metrics', future_stack=True)
    test_plot_inp = pd.DataFrame(test_plot_inp, columns=['val']).reset_index()
    test_plot_inp['cv_num_phase'] = test_plot_inp.loc[:, ['cv_num', 'phase']].sum(axis=1)
    test_plot_inp = test_plot_inp.query('phase in @test_phases')
    test_plot_inp['set_label'] = test_plot_inp.loc[:, ['site_settings']].apply(
        lambda x: set_relabel[x['site_settings']], axis=1)

    test_TP_FP_plot_inp = test_TP_FP.loc[(slice(None), slice(None), site_settings_index), ['TP\nW0', 'TP\nW1', 'TP\nW2', 'TN']]
    test_TP_FP_plot_inp.columns.name = 'cts_name'
    test_TP_FP_plot_inp = test_TP_FP_plot_inp.stack(future_stack=True).to_frame('cts_perc').reset_index()
    test_TP_FP_plot_inp['set_label'] = test_TP_FP_plot_inp.loc[:, ['site_settings']].apply(
        lambda x: set_relabel[x['site_settings']], axis=1)

    # set negative to 100% (for stacked barplot)
    test_TN_FN_plot_inp_neg = test_TP_FP_plot_inp.copy()
    test_TN_FN_plot_inp_neg['cts_perc'] = 100

    cols_f1 = [f'recall_TP\nW{x}' for x in range(3)] + [f'prec_TN']
    test_TP_FP_f1_plot_inp = test_TP_FP_f1.loc[(slice(None), slice(None), site_settings_index), cols_f1]
    test_TP_FP_f1_plot_inp.columns.name = 'score_name'
    test_TP_FP_f1_plot_inp = test_TP_FP_f1_plot_inp.stack(future_stack=True).to_frame('score').reset_index()
    test_TP_FP_f1_plot_inp['set_label'] = test_TP_FP_f1_plot_inp.loc[:, ['site_settings']].apply(
        lambda x: set_relabel[x['site_settings']], axis=1)
    test_TP_FP_f1_plot_inp['score_type'] = test_TP_FP_f1_plot_inp.loc[:, ['score_name']].apply(
        lambda x: x['score_name'].split('_')[0], axis=1)
    test_TP_FP_f1_plot_inp['score_xlabel'] = test_TP_FP_f1_plot_inp.loc[:, ['score_name']].apply(
        lambda x: x['score_name'].split('_')[1], axis=1)

    # extract geom as input for plot
    test_metrics_geom_plot_inp = test_metrics_geom.loc[(slice(None), slice(None), site_settings_index, slice(None), slice(None), slice(None), slice(None), test_phases), :]
    test_metrics_geom_plot_inp.set_index('true_pred_set', append=True, inplace=True)

    # get geom counts
    #!!! counts of different patches are summed up !!!
    test_geom_counts_plot_inp1 = test_metrics_geom_plot_inp.groupby(
        ['site', 'site_settings', 'cv_num', 'phase', 'true_pred', 'class', 'true_pred_set']).count()
    # take only true counts and rename set as true (this is used for barplots)
    test_geom_true_counts = test_geom_counts_plot_inp1.query('true_pred == "true"').groupby(
        ['site', 'cv_num', 'phase', 'true_pred', 'class']).first()
    test_geom_true_counts['site_settings'] = 'true'
    test_geom_true_counts['true_pred_set'] = 'true'
    test_geom_true_counts = test_geom_true_counts.reset_index().set_index(
        test_geom_counts_plot_inp1.index.names)
    test_geom_counts_plot_inp = pd.concat(
        [test_geom_true_counts,
        test_geom_counts_plot_inp1.query('true_pred == "pred"')], axis=0)


    # get geom counts
    #!!! counts of different patches are summed up !!!
    test_metrics_geom_plot_inp2 = test_metrics_geom_plot_inp.query('area_m >= (1.5*1.5)*3')
    test_geom_counts_plot_inp1 = test_metrics_geom_plot_inp2.groupby(
        ['site', 'site_settings', 'cv_num', 'phase', 'true_pred', 'class', 'true_pred_set']).count()
    # take only true counts and rename set as true (this is used for barplots)
    test_geom_true_counts = test_geom_counts_plot_inp1.query('true_pred == "true"').groupby(
        ['site', 'cv_num', 'phase', 'true_pred', 'class']).first()
    test_geom_true_counts['site_settings'] = 'true'
    test_geom_true_counts['true_pred_set'] = 'true'
    test_geom_true_counts = test_geom_true_counts.reset_index().set_index(
        test_geom_counts_plot_inp1.index.names)
    test_geom_counts_plot_inp_filt = pd.concat(
        [test_geom_true_counts,
        test_geom_counts_plot_inp1.query('true_pred == "pred"')], axis=0)


    # get geom sum
    #!!! different test patches are summed up !!!
    test_geom_sum_plot_inp1 = test_metrics_geom_plot_inp.groupby(
        ['site', 'site_settings', 'cv_num', 'phase', 'true_pred', 'class', 'true_pred_set']).sum()
    # take only true sums and rename set as true (this is used for barplots)
    test_geom_true_sum = test_geom_sum_plot_inp1.query('true_pred == "true"').groupby(
        ['site', 'cv_num', 'phase', 'true_pred', 'class']).first()
    test_geom_true_sum['site_settings'] = 'true'
    test_geom_true_sum['true_pred_set'] = 'true'
    test_geom_true_sum = test_geom_true_sum.reset_index().set_index(
        test_geom_sum_plot_inp1.index.names)

    test_geom_sum_plot_inp = pd.concat(
        [test_geom_true_sum,
        test_geom_sum_plot_inp1.query('true_pred == "pred"')], axis=0)

    cols_incl = ['TRUE_TP_perc', 'TRUE_perc', 'TRUE_FP_perc', 'FDR_perc_neg',
                 'PRED_TP_perc', 'PRED_perc', 'PRED_FN_perc']
    test_metrics_prec_recall_sel = test_metrics_prec_recall_inp.loc[(slice(None), slice(None), site_settings_index), :].copy()
    test_metrics_prec_recall_sel.reset_index(inplace=True)
    test_metrics_prec_recall_sel['true_pred'] = test_metrics_prec_recall_sel['true_pred_set'].apply(
        lambda x: x.split(' ')[0])
    #test_geom_sum_plot_inp.loc[:, cols_incl] = test_metrics_prec_recall1.rename(
    #    {'class_name': 'class'}, axis=1).set_index(test_geom_sum_plot_inp1.index.names).loc[:, cols_incl]
    concat_inp = test_metrics_prec_recall_sel.rename(
        {'class_name': 'class'}, axis=1).set_index(test_geom_sum_plot_inp1.index.names).loc[:, cols_incl]
    test_geom_sum_plot_inp = pd.concat([test_geom_sum_plot_inp, concat_inp], axis=1)
    test_geom_sum_plot_inp.loc[test_geom_sum_plot_inp.index.get_level_values(level='true_pred') == "true", ['TRUE_perc']] = 100
    test_geom_sum_plot_inp.loc[test_geom_sum_plot_inp.index.get_level_values(level='true_pred') == "true", ['PRED_perc']] = -100

    return (box_plot_inp, epoch_best_inp, test_plot_inp, test_TP_FP_plot_inp,
            test_TN_FN_plot_inp_neg, test_TP_FP_f1_plot_inp,
            test_metrics_geom_plot_inp, test_geom_counts_plot_inp,
            test_geom_sum_plot_inp, test_metrics_prec_recall,
            test_geom_counts_plot_inp_filt,
            validate_metrics_prec_recall)


def extract_pond_counts(test_metrics_geom, site_settings_index, test_phases, EPSG_num):
    '''
    extract pond counts per size range for plotting
    '''
        # extract geom as input for plot
    test_metrics_geom_plot_inp = test_metrics_geom.loc[(slice(None), slice(None), site_settings_index, slice(None), slice(None), slice(None), slice(None), test_phases), :].copy()
    test_metrics_geom_plot_inp.set_index('true_pred_set', append=True, inplace=True)
    test_metrics_geom_plot_inp.sort_index(inplace=True)

    ponds = test_metrics_geom_plot_inp.loc[test_metrics_geom_plot_inp['class']=="ponds", :]
    ponds_true = ponds.loc[ponds.true_pred=="true", :].copy()#.reset_index(['patch', 'index', 'true_pred_set'])
    ponds_pred = ponds.loc[ponds.true_pred!="true", :].copy()#.reset_index(['patch','index', 'true_pred_set'])
    gdf_ponds_true = geo_utils.convert_txt_to_gdf(ponds_true, EPSG_num, geom_col='geometry')
    gdf_ponds_pred = geo_utils.convert_txt_to_gdf(ponds_pred, EPSG_num, geom_col='geometry')

    grouped_true = gdf_ponds_true.groupby(['site', 'settings', 'site_settings', 'cv_num', 'folder', 'phase', 'patch'])
    grouped_pred = gdf_ponds_pred.groupby(['site', 'settings', 'site_settings', 'cv_num', 'folder', 'phase', 'patch'])

    ponds_true['aoi_intersect_area'] = 0.0
    ponds_pred['aoi_intersect_area'] = 0.0
    ponds_true['aoi_intersect_perc'] = 0.0
    ponds_pred['aoi_intersect_perc'] = 0.0

    for i_true, i_pred in zip(grouped_true, grouped_pred):
        # print(i_true)
        true_index = i_true[1].index
        for i in true_index:
            # print(i)
            geom = i_true[1].loc[i, 'geometry']
            geo_utils.add_intersection_area(i_pred[1], geom, 32654)
            if np.all(i_pred[1]['aoi_intersect_area'] == 0):
                ponds_true.loc[i, 'aoi_intersect_area'] = 0
                ponds_true.loc[i, 'aoi_intersect_perc'] = 0
            else:
                ponds_found = i_pred[1].query('aoi_intersect_area > 0')
                ponds_max_area = ponds_found['aoi_intersect_area'].max(axis=0)
                ponds_true.loc[i, 'aoi_intersect_area'] = ponds_max_area
                ponds_pred.loc[ponds_found.index, 'aoi_intersect_area'] = ponds_found['aoi_intersect_area']

                ponds_max_perc = ponds_found['aoi_intersect_perc'].max(axis=0)
                ponds_true.loc[i, 'aoi_intersect_perc'] = ponds_max_perc
                ponds_pred.loc[ponds_found.index, 'aoi_intersect_perc'] = ponds_found['aoi_intersect_perc']

    return ponds_true, ponds_pred


def get_last_epoch_metrics(train_metrics_DL, suffix_out, PATH_EXPORT,
                           prefix_out=''):

    max_epoch = train_metrics_DL.index.get_level_values(
        level='epoch').max()
    epoch_99 = train_metrics_DL.query('epoch == @max_epoch').groupby(
        ['site', 'settings', 'site_settings', 'epoch']).mean()
    path_file = os.path.join(
        PATH_EXPORT,
        f"{prefix_out}_{suffix_out}_metrics_summary_epoch99_CVmean.txt")

    epoch_99.stack('phase', future_stack=True).to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n',
        header=True)

    epoch_99['epoch_diff'] = (
        epoch_99.loc[:, ('train', 'jacc_macro')]
        - epoch_99.loc[:, ('validate', 'jacc_macro')])

    return epoch_99


def save_metrics_summary(
        metrics_df_inp, suffix_out, PATH_EXPORT,
        epoch_diff=None, site_sets_dict_with_label=None,
        prefix_out=''):
    '''
    metrics_df can either be:
    - train_metrics_best
    - test_metrics_all

    epoch_diff: epoch_99

    site_sets_dict_with_label: this can be site_sets_dict if
        run from MAIN_create_paper_plot_summary. Then the site_sets_dict
        contains labels which can be added to the dataframe.

    '''
    metrics_df = metrics_df_inp.copy()
    if site_sets_dict_with_label is not None:
        for i_set_name, i_set_lst_s in site_sets_dict_with_label.items():
            for i in i_set_lst_s:
                metrics_df.loc[(slice(None), slice(None), i[0]), 'label'] = i[-1]
        add_group_id = ['label']
    else:
        add_group_id = []

    if epoch_diff is not None:
        # this is just for trainnig metrics
        metrics_df['epoch100_diff'] = epoch_diff.reset_index('epoch')['epoch_diff']

    # --- save non-averaged
    path_file = os.path.join(PATH_EXPORT,
                f"{prefix_out}_{suffix_out}_metrics_summary_best_epoch.txt")
    metrics_df.reset_index().to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n',
        header=True)

    # --- save averaged
    metrics_df_cv_mean = metrics_df.groupby(
        ['site', 'settings', 'site_settings', 'phase'] + add_group_id).mean()
    metrics_df_cv_mean['epoch'] = metrics_df.reset_index('epoch').groupby(
        ['site', 'settings', 'site_settings', 'phase'] + add_group_id)['epoch'].agg(list)
    path_file = os.path.join(PATH_EXPORT,
                f"{prefix_out}_{suffix_out}_metrics_summary_best_epoch_CVmean.txt")
    metrics_df_cv_mean.reset_index().to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n',
        header=True)

    return


def save_TP_geom_metrics(
        validate_metrics_prec_recall, test_geom_sum_plot_inp,
        test_TP_FP_f1_plot_inp, test_TP_FP_plot_inp,
        PATH_EXPORT, prefix_out):
    '''
    prefix_out = f'{i_set_name}_{suffix_add}'
    '''

    # -- save validation precision and recall
    # save non-averaged
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_validation_TPR_FDR.txt')
    validate_metrics_prec_recall.to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n', header=True)
    # save averaged
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_validation_TPR_FDR_CVmean.txt')
    groupby_lst = ['site', 'settings', 'site_settings',
                   'class_name', 'true_pred_set']
    validate_metrics_prec_recall.groupby(groupby_lst).mean().to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n', header=True)

    # -- save geom and TPR and FDR
    # save non-averaged
    index_order = ['site', 'true_pred', 'true_pred_set', 'phase', 'class']
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_test_geom_sum_TPR_FDR.txt')
    test_geom_sum_plot_inp.reset_index('site_settings', drop=True).drop(
        'geometry', axis=1).reset_index().set_index(index_order).sort_index().to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n', header=True)
    # save averaged
    groupby_lst = [
        'site', 'phase', 'true_pred', 'class', 'true_pred_set']
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_test_geom_sum_TPR_FDR_CVmean.txt')
    test_geom_sum_plot_inp.reset_index('site_settings', drop=True).drop('geometry', axis=1).groupby(groupby_lst).mean().reset_index().set_index(index_order).sort_index().to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n', header=True)

    index_lst_save =  ['site', 'settings', 'site_settings', 'cv_num', 'folder',
       'merge_id', 'train_id', 'phase']
    # -- save baydzherakh dependent recall and precision
    # save non-averaged
    df_save = test_TP_FP_f1_plot_inp.drop(['score_type', 'score_xlabel'], axis=1).set_index(
        index_lst_save + ['score_name', 'set_label']).unstack('score_name')
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_test_baydz_recall_prec.txt')
    df_save.to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n', header=True)
    # save averaged
    groupby_lst = [
        'site', 'settings', 'site_settings', 'phase', 'set_label']
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_test_baydz_recall_prec_CVmean.txt')
    df_save.groupby(groupby_lst).mean().to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n',
        header=True)

    # --- save baydzherakh dependent TPR and FNR
    # save non-averaged
    df_save = test_TP_FP_plot_inp.set_index(
        index_lst_save + ['cts_name', 'set_label']).unstack('cts_name')
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_test_baydz_TPR_FNR.txt')
    df_save.to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n', header=True)
    # save averaged
    groupby_lst = [
        'site', 'settings', 'site_settings', 'phase', 'set_label']
    path_file = os.path.join(PATH_EXPORT,
            f'{prefix_out}_test_baydz_TPR_FNR_CVmean.txt')
    df_save.groupby(groupby_lst).mean().to_csv(
        path_or_buf=path_file, sep='\t', lineterminator='\n',
        header=True)

    return


# ---------------- plotting -------------------
def get_metrics_colormap(n_settings):

    col_map = [sns.color_palette("flare", n_colors=7, as_cmap=False)[0::3]]
    col_map += [sns.color_palette("Blues", n_colors=10, as_cmap=False)[3::3]]
    col_map += [sns.color_palette("Greys", n_colors=10, as_cmap=False)[3::3]]
    col_map_inp = np.moveaxis(col_map, 0, 1)

    col_map_combined = sns.color_palette("flare", n_colors=7, as_cmap=False)[0::3][:2]
    col_map_combined += sns.color_palette("Blues", n_colors=10, as_cmap=False)[3::3][:2]
    col_map_combined += sns.color_palette("Greys", n_colors=10, as_cmap=False)[3::3][:2]

    col_1 = sns.color_palette("Blues", n_colors=11, as_cmap=False)[3::3][:3]
    col_1 += sns.color_palette("Purples", n_colors=11, as_cmap=False)[3::3][:3]
    col_2 = sns.color_palette("RdPu", n_colors=11, as_cmap=False)[3::3][:3]
    col_2 += sns.color_palette("Reds", n_colors=11, as_cmap=False)[3::3][:3]
    col_tt = [col_1] + [col_2]

    col_viridis = sns.color_palette("RdPu", n_colors=n_settings, as_cmap=False)[::-1]

    col_s1 = sns.color_palette("PiYG", n_colors=11, as_cmap=False)[1:-1]  # set 1 valid, ttrain
    col_s2 = sns.color_palette("RdBu_r", n_colors=11, as_cmap=False)[1:-1]
    col_s3 = sns.color_palette("PuOr", n_colors=11, as_cmap=False)[1:-1]  # BrBG_r  # PuOr_r  # Wistia
    col_s4 = sns.color_palette("BrBG_r", n_colors=11, as_cmap=False)[1:-1]  # BrBG_r  # PuOr_r  # Wistia
    col_s5 = sns.color_palette("Greys", n_colors=11, as_cmap=False)[1:-1]
    col_s6 = sns.color_palette("Greys_r", n_colors=6, as_cmap=False)[2:-1]
    col_s7 = sns.color_palette("PRGn", n_colors=11, as_cmap=False)[1:-1]  # BrBG_r  # PuOr_r  # Wistia
    col_s8 = sns.color_palette("seismic", n_colors=7, as_cmap=False)[1:-1]
    col_s9 = sns.color_palette("seismic_r", n_colors=7, as_cmap=False)[1:-1]

    col_scatter = [col_s1, col_s2, col_s3, col_s4, col_s6, col_s7]
    col_set = [col_s5[4], col_s1[0], col_s2[0], col_s3[0], col_s4[0], col_s7[0]]
    col_bright = [col_s5[2], col_s1[1], col_s2[1], col_s3[1], col_s4[1], col_s7[1]]

    col_scatter2 = [col_s9,
                    col_s3,
                    col_s7, col_s2, col_s8,
                    col_s4, col_s6, col_s1]
    col_set2 = [col_s5[4], col_s9[0], col_s3[0],
                col_s7[0], col_s2[0], col_s8[0],
                col_s4[0], col_s1[0]]
    col_bright2 = [col_s5[2], col_s9[1], col_s3[1],
                   col_s7[1], col_s2[1], col_s8[1],
                   col_s4[1], col_s1[1]]
    col_scaling = [col_scatter2, col_set2, col_bright2]

    col_scatter3 = [col_s2, col_s8, col_s3, col_s7,
                    #col_s6,
                    col_s4]
    col_set3 = [
        col_s5[4], col_s2[0], col_s8[0], col_s3[0], col_s7[0],
        #col_s6[0],
        col_s4[0]]
    col_bright3 = [
        col_s5[2], col_s2[1], col_s8[1], col_s3[1], col_s7[1],
        #col_s6[1],
        col_s4[1]]
    col_fine_tune = [col_scatter3, col_set3, col_bright3]


    return col_scatter, col_set, col_bright, col_s6, col_scaling, col_fine_tune


def plot_metrics_test_patch_stats_prec_recall_TP_violon(
        test_plot_inp, test_TP_FP_plot_inp, test_metrics_geom_plot_inp,
        test_geom_counts_plot_inp, test_TP_FP_f1_plot_inp,
        test_prec_recall_plot_inp,
        site_settings_index, metrics_plot, test_phases,
        set_annot_text, metrics_title, site_label,
        cat_order, alpha_bar, alpha_violon, col_set, col_set_bright,
        col_scatter, n_cv_nums, font_size, font_size_leg,
        PATH_EXPORT, file_suffix, fig_size=None,
        plot_baydz_TPFP=False, wspace=2.5, hspace=0.8, err_bar=None,
        average_test=False, markers_size=None, markers_test=None):

    if fig_size is None:
        fig_size = (9.5, 4)
    if err_bar is None:
        err_bar = ("pi", 100)  # min max

    cv_hue_order = [f'cv{x:02d}' for x in range(n_cv_nums)]

    n_settings = len(site_settings_index)
    marker_size_add = (4-n_settings)*0.5
    axg = []
    # gridspec inside gridspec
    fig_v1_11 = plt.figure(figsize=fig_size)  # (8.27, 2.5)
    gs0 = gridspec.GridSpec(
        2 + 2*len(test_phases), 17, figure=fig_v1_11, wspace=wspace, hspace=hspace)

    # subplot 0 to 2
    gs00 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs0[:2, :5], wspace=0)
    axg.extend([fig_v1_11.add_subplot(gs00[:, x]) for x in range(2)])

    # subplot 2 to 7
    gs00 = gridspec.GridSpecFromSubplotSpec(
        2, 6, subplot_spec=gs0[:2, 5:], wspace=0)
    axg.extend([fig_v1_11.add_subplot(gs00[:, x]) for x in range(6)])

    # subplot 8 to 9
    for e, i in enumerate(test_phases):
        n_y = [2*(e+1), 2*(e+2)]
        gs03 = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs0[n_y[0]:n_y[1], :4], wspace=0)
        axg.extend([fig_v1_11.add_subplot(gs03[:2, :])])

    # subplot 10 to 13
    for e, i in enumerate(test_phases):
        n_y = [2*(e+1), 2*(e+2)]
        gs02 = gridspec.GridSpecFromSubplotSpec(
            2, 6, subplot_spec=gs0[n_y[0]:n_y[1], 5:],
            wspace=3.5, hspace=0.1)
        axg.extend(
            [fig_v1_11.add_subplot(gs02[0, :]),
             fig_v1_11.add_subplot(gs02[1, :]),
             ])

    start_subp = 8+len(test_phases)
    plotting_utils.format_axes_general(
        axg, [0, 8+len(test_phases)-1, start_subp + len(test_phases)*2 - 1],
        dict(labeltop=False, labelbottom=True,
             labelleft=True, labelright=False, labelsize=font_size))

    plotting_utils.format_axes_general(
        axg, list(range(1, 7)),
        dict(labeltop=False, labelbottom=True,
             labelleft=False, labelright=False, labelsize=font_size))

    plotting_utils.format_axes_general(
        axg, [7],
        dict(labeltop=False, labelbottom=True,
             labelleft=False, labelright=True,
             left=True, right=True, bottom=True, top=False, labelsize=font_size))

    plotting_utils.format_axes_general(
        axg, list(range(8, 8+len(test_phases)-1)),
        dict(labeltop=False, labelbottom=False,
             labelleft=True, labelright=False,
             #top=True, bottom=False, left=False, right=True,
             labelsize=font_size))

    plotting_utils.format_axes_general(
        axg, list(range(start_subp, start_subp + len(test_phases)*2-1)),
        dict(labeltop=False, labelbottom=False,
             labelleft=True, labelright=False,
             labelsize=font_size), ax_excl_lst=['bottom'])

    sns.set_style(
        "whitegrid",
        {'axes.grid.axis' : 'both', "axes.ticks.visible": True,
         'legend.fontsize':6})

    if markers_test is None:
        markers_test = ['+', 'x', 'o', 'D', '.', '*']
    if markers_size is None:
        markers_size = [4.5, 3.5, 2.0, 2.0, 2.0, 2.0]
    leg_count_lst = []
    leg_label_lst = []
    count_leg = 0
    for e_ax, i_metrics in enumerate(metrics_plot):
        scat_p = test_plot_inp.query('metrics == @i_metrics')
        for e_phase, i_phase in enumerate(test_phases):
            for e_set, i_set in enumerate(site_settings_index):
                col_pal = [col_scatter[e_set][0]]
                if not average_test:
                    sns.pointplot(
                        data=scat_p.query('phase == @i_phase and site_settings == @i_set'),
                        x="set_label", y="val", hue='cv_num',
                        hue_order=cv_hue_order,
                        dodge=.8 - .8 / n_cv_nums, #label=f'{i_phase}',
                        palette=col_pal*n_cv_nums,
                        errorbar=None,
                        markers=markers_test[e_phase],
                        markersize=markers_size[e_phase] + marker_size_add,
                        linewidth=0.75, linestyle="none",
                        ax=axg[e_ax], legend=True)
                    mul_leg = n_cv_nums
                else:
                    sns.pointplot(
                        data=scat_p.query('phase == @i_phase and site_settings == @i_set'),
                        x="set_label", y="val",# hue='cv_num',
                        #hue_order=cv_hue_order,
                        #dodge=.8 - .8 / 1,
                        label=f'{i_phase}', color=col_pal[0],
                        errorbar=err_bar, capsize=0.1,
                        err_kws={'color': col_pal[0], 'linewidth': 0.3,
                                 'zorder':1.5, 'alpha':0.8},
                        markers=markers_test[e_phase],
                        markersize=markers_size[e_phase] + marker_size_add + 1,
                        linewidth=0.75, linestyle="none",
                        ax=axg[e_ax], zorder=2, legend=True)
                    mul_leg = 1
                if e_set == 0 and e_ax == 0:
                    leg_count_lst.append(count_leg*mul_leg)
                    leg_label_lst.append(i_phase)
                count_leg += 1
        if e_ax != len(metrics_plot) - 1:
            sns.despine(ax=axg[e_ax])  # removes top and right axis
        else:
            sns.despine(ax=axg[e_ax], top=True, right=False)

    matplotlib.rcParams['legend.fontsize'] = font_size
    for e_ax, i_metrics in enumerate(metrics_plot):
        axg[e_ax].grid(True, axis='both', lw=0.25, zorder=1.0)
        axg[e_ax].set_xlabel('', fontsize=font_size, labelpad=1)
        axg[e_ax].set_ylabel('metrics', fontsize=font_size,
                             labelpad=1)
        axg[e_ax].set_title(metrics_title[e_ax],
                            fontsize=font_size, pad=2.0)
        axg[e_ax].set_ylim(0.0, 1.0)
        try:
            axg[e_ax].get_legend().set_visible(False)
        except:
            pass
    axg[e_ax].yaxis.set_label_position("right")

    leg_handles, leg_label = axg[0].get_legend_handles_labels()
    axg[0].legend(
        [leg_handles[x] for x in leg_count_lst],
        leg_label_lst)

    sns.move_legend(axg[0], loc="upper left",
                    bbox_to_anchor=(0.0, -(1.5 + len(test_phases))),
                                    fontsize=font_size_leg)

    axg[0].annotate(
        set_annot_text, xy=[0, -(1.2 + len(test_phases))],
        xycoords='axes fraction', fontsize=font_size_leg)

    # =================== violon plot
    hue_order_set = ['true'] + ['pred ' + x.replace(':', '_')
                                for x in site_settings_index]
    width_v = [1.4]*n_settings
    # leg_text = ['true']
    inner = 'quart'
    gap_shift = [0.1]*n_settings
    marker_f1 = ['D', 'o']

    axg2_lower = []  # twin axis for metrics
    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n*2 + len(metrics_plot) + len(test_phases)
        axg2_lower.append(axg[e_ax + 1].twinx())
    leg_text = []
    leg_handles_lst = []
    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n*2 + len(metrics_plot) + len(test_phases)
        geom_plt = test_metrics_geom_plot_inp.query('phase ==@i_phase').reset_index()
        counts_plt = test_geom_counts_plot_inp.query('phase ==@i_phase')
        prec_plt = test_prec_recall_plot_inp.query('phase ==@i_phase').reset_index()
        for e_set, i_set in enumerate(site_settings_index):
            if e_set > 0:
                geom_plt_filt = geom_plt.query(
                    'site_settings==@i_set and true_pred != "true"')
                col_set_inp = [col_set[e_set + 1]]
            else:
                geom_plt_filt = geom_plt.query('site_settings==@i_set')
                col_set_inp = [col_set[0]] + [col_set[e_set + 1]]
                if e_ax_n == 0:
                    leg_text.append('True')

            sns.violinplot(
                data=geom_plt_filt, x="class", y="area_m", hue="true_pred",
                hue_order=['true', 'pred'], order=cat_order,
                split=True, gap=gap_shift[e_set], inner=inner, log_scale=True,
                alpha=alpha_violon, dodge=True, width=width_v[e_set],
                fill=True, linewidth=0.25,
                palette=col_set_inp, ax=axg[e_ax], legend=False, zorder=1)
            # for line
            sns.violinplot(
                data=geom_plt_filt, x="class", y="area_m", hue="true_pred",
                hue_order=['true', 'pred'], order=cat_order,
                split=True, gap=gap_shift[e_set], inner=inner,
                log_scale=True, alpha=1.,
                dodge=True, width=width_v[e_set], fill=False,
                linewidth=0.75, palette=col_set_inp, ax=axg[e_ax],
                legend=True, zorder=1.1)
            # for fine line
            sns.violinplot(
                data=geom_plt_filt, x="class", y="area_m", hue="true_pred",
                        hue_order=['true', 'pred'], order=cat_order,
                        split=True, gap=gap_shift[e_set], inner=inner,
                        log_scale=True,
                        alpha=1.0, dodge=True, width=width_v[e_set],
                        fill=False, linewidth=0.1,
                        color='k', linecolor='k', ax=axg[e_ax],
                        legend=False, zorder=1.2)
            axg[e_ax].axhline(1.5*1.5*3, lw=0.5, color="k", clip_on=False,
                              linestyle='--', dashes=(2, 4, 2, 4))
            if e_ax_n == 0:
                leg_text.append('pred ' + i_set.replace(':', '_'))
        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax].get_legend_handles_labels()
            leg_handles_lst.extend(leg_handles[:1] + leg_handles[1::2])

        sns.despine(ax=axg[e_ax])

        # plot TRUE VALS as barplot (as positive percentage values)
        # true as 100% with annotated area value
        sns.barplot(data=counts_plt.query('true_pred == "true"'),
                    x="class", y='TRUE_perc', hue="true_pred_set",
                    hue_order=['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:],
                    order=cat_order,
                    gap=.25, alpha=alpha_bar, dodge=True, #width=0.5,
                    linewidth=0.0, palette=[col_set[0]]*4 + col_set[1:],
                    fill=True,
                    errorbar=err_bar,
                    err_kws={'color': 'k', 'linewidth': 0.25},
                    ax=axg[e_ax + 1], legend=False, zorder=1)


        patch_x_pos = [x.get_x()  + x.get_width() / 2. for x in axg[e_ax + 1].patches][:len(cat_order)]
        if 'area_km' not in counts_plt.columns:
            counts_plt.rename({'area': 'area_km'}, axis=1, inplace=True)
        area_query = counts_plt.query(
                'true_pred == "true" and cv_num=="cv00"').reset_index().loc[:, ['class', 'area_km']]
        # ther might be duplicates if the model was taken from different "sites" (e.g. HEX_A01_A02, HEX_SPOT_A01_A02)
        # (site here correesponds to the site that has been used for the trainng. Ant path and phase reflect the
        # predicted test patch)
        # also round areas in case there are small missmatches (which is
        # for example tha case for BLyakh_v3 versus v4 as traning area
        # of v3 was initially lipped too much)
        area_query = area_query.round(3)
        area_query.drop_duplicates(inplace=True)
        area_query.set_index('class', inplace=True)

        # patch_area = [area_query.loc[x, 'area_km']  if x in area_query.index else 0 for x in cat_order]
        patch_area = [
            area_query.loc[x, 'area_km'] for x in cat_order if x in area_query.index]
        # drop duplicates which come form the three differeent sets

        patch_annotate = pd.DataFrame(
                [patch_x_pos, patch_area],
                index=['x_pos', 'area_km']).T.drop_duplicates('x_pos').set_index('x_pos')

        for i_pos in patch_annotate.index:
            area_num = patch_annotate.loc[i_pos, "area_km"]
            axg[e_ax + 1].annotate(
                    f'{area_num:.3f}', (i_pos, 10),
                     ha='center', va='bottom', rotation=90,
                     fontsize=font_size-1)

        #for e_set, i_set in enumerate(site_settings_index):
        # ['TRUE_TP_perc', 'TRUE_FP_perc_neg']
        for e_perc, i_perc in enumerate(['TRUE_TP_perc', 'FDR_perc_neg']):
            if e_perc == 0:
                col_s = col_set

            else:
                col_s = col_set_bright
            sns.barplot(
                data=counts_plt.query('true_pred != "true"'),
                x="class", y=i_perc, hue="true_pred_set",
                hue_order=['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:],
                order=cat_order,
                gap=.25, alpha=alpha_bar, dodge=True, #width=0.5,
                linewidth=0.0, palette=[col_s[0]]*4 + col_s[1:],
                fill=True,
                errorbar=err_bar,
                err_kws={'color': 'k', 'linewidth': 0.25},
                ax=axg[e_ax + 1], legend=True, zorder=4-e_perc)
        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax + 1].get_legend_handles_labels()
            leg_text.extend(leg_label)

        for e_score, i_score in enumerate(['recall', 'prec']):
            hue_order_inp = ['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:]
            sns.pointplot(
                data=prec_plt.rename({'class_name': 'class'}, axis=1).query('metrics==@i_score'),
                x="class", y="val", hue="true_pred_set",
                hue_order=hue_order_inp,
                order=cat_order,
                log_scale=False, alpha=1,
                dodge=.8 - .8 / len(hue_order_inp), #width=0.5,
                markers=marker_f1[e_score],
                fillstyle='full',
                markersize=3.5 + marker_size_add,
                markeredgewidth=0.25, markeredgecolor='k',
                linewidth=0.5, linestyle="none",
                palette=[col_set[0]]*4 + col_set[1:],
                errorbar=None,
                ax=axg2_lower[e_ax_n], legend=True, zorder=1.5)

            if e_ax_n == 0:
                # start from 1 because first label is 'True'
                leg_text.extend([f'{i_score} {x}' for x in  leg_label[1:n_settings+1]])

        sns.despine(ax=axg[e_ax + 1])

        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax + 1].get_legend_handles_labels()
            leg_handles2, leg_label = axg2_lower[e_ax_n].get_legend_handles_labels()
            # start from 1 because first label is 'True'
            leg_handles_lst.extend(
                leg_handles + leg_handles2[1:n_settings+1]
                + leg_handles2[-n_settings:])
        axg[e_ax + 1].get_legend().set_visible(False)

    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n*2 + len(metrics_plot) + len(test_phases)
        ax_param = dict(labeltop=False, labelbottom=False,
                        labelleft=False, labelright=True,
                        left=False, labelsize=font_size)
        plotting_utils.format_axes_general(
            axg2_lower, [e_ax_n], ax_param)

        ax_param = dict(labeltop=False, labelbottom=False,
                        labelleft=True, labelright=False,
                        left=True, right=False, bottom=False, top=False,
                        labelsize=font_size)
        plotting_utils.format_axes_general(axg, [e_ax], ax_param)

        title_p = ' '.join(i_phase.split('_')[:-1])
        axg[e_ax].grid(True, axis='both', lw=0.25)
        axg[e_ax].set_ylim(0.01, 10**7)
        axg[e_ax].set_ylabel(
            r'area [$\mathregular{m^2}$]', fontsize=font_size,
            labelpad=1)
        axg[e_ax].get_legend().set_visible(False)

        axg[e_ax + 1].grid(True, axis='both', lw=0.25)
        axg[e_ax + 1].set_ylim(-100, 100)
        axg[e_ax + 1].set_ylabel(
             'FDR | TPR [%]', fontsize=font_size, labelpad=0.3)
        axg[e_ax + 1].set_xlabel(
             '', fontsize=font_size, labelpad=1)

        axg2_lower[e_ax_n].grid(False)
        axg2_lower[e_ax_n].set_ylim(-1.0, 1.0)
        # keep only ticks which are between 0 and 1
        yticks = axg2_lower[e_ax_n].get_yticks()
        filtered_xticks = [x for x in yticks if x <=1 and x >= 0]
        axg2_lower[e_ax_n].set_yticks(filtered_xticks)
        axg2_lower[e_ax_n].set_ylabel(
            'metrics', fontsize=font_size,
            labelpad=0.1, rotation=270)
        axg2_lower[e_ax_n].yaxis.set_label_coords(1.12, (0.5/1.2/2 + 0.5))

        axg2_lower[e_ax_n].get_legend().set_visible(False)
        axg[e_ax].set_title(title_p, fontsize=font_size, pad=2.0)
        axg2_lower[e_ax_n].spines['top'].set_linewidth(0.0)
        axg2_lower[e_ax_n].spines['left'].set_linewidth(0.0)
        axg2_lower[e_ax_n].spines['bottom'].set_linewidth(0.0)
        axg[e_ax].spines['bottom'].set_linewidth(0.0)
        axg[e_ax + 1].spines['bottom'].set_linewidth(0.0)
        axg[e_ax].spines['right'].set_linewidth(0.0)
        axg[e_ax + 1].axhline(0, lw=0.4, color="k", clip_on=False)

    axg[e_ax + 1].legend(leg_handles_lst, leg_text)

    sns.move_legend(axg[e_ax + 1], loc="upper left",
                    bbox_to_anchor=(0.5, -1.1), fontsize=font_size_leg)

    # ================= plot barplot
    marker_f1 = ['D', 'o']
    marker_fill = ['none', 'full']
    axf2 = []
    score_label = []
    bar_order = ['TP\nW0', 'TP\nW1', 'TP\nW2', 'TN']
    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n + len(metrics_plot) #+ len(test_phases)*2

        TP_FP_plt = test_TP_FP_plot_inp.query('phase ==@i_phase and cts_name!="TN"')
        sns.barplot(data=TP_FP_plt, x="cts_name", y="cts_perc", hue="set_label",
                    order=bar_order,
                    log_scale=False, alpha=alpha_bar, dodge=True, gap=.25,  #width=0.8,
                    linewidth=0.0, palette=col_set[1:],
                    fill=True,
                    errorbar=err_bar,
                    err_kws={'color': 'k', 'linewidth': 0.25},
                    ax=axg[e_ax], legend=True, zorder=2)

        # plot true negatives with different alpha
        TP_FP_plt = test_TP_FP_plot_inp.query('phase ==@i_phase and cts_name=="TN"')
        sns.barplot(data=TP_FP_plt, x="cts_name", y="cts_perc", hue="set_label",
                    order=bar_order,
                    log_scale=False, alpha=alpha_bar, dodge=True, gap=.25,  #width=0.8,
                    linewidth=0.0, palette=col_set_bright[1:],
                    fill=True,
                    errorbar=err_bar,
                    err_kws={'color': 'k', 'linewidth': 0.25},
                    ax=axg[e_ax], legend=True, zorder=2)

        if plot_baydz_TPFP:
            axf2.append(axg[e_ax].twinx()) # applies twinx to ax2, which is the second y axis.
            TP_FP_plt_f1 = test_TP_FP_f1_plot_inp.query('phase ==@i_phase')
            for e_score, i_score in enumerate(['recall', 'prec']):
                sns.pointplot(
                    data=TP_FP_plt_f1.query('score_type ==@i_score'),
                    x="score_xlabel", y="score", hue='set_label',
                    dodge=.8 - .8 / n_settings, legend=True,
                    palette=col_set[1:], errorbar=None,
                    markers=marker_f1[e_score],
                    fillstyle=marker_fill[e_score],
                    markersize=2.5 + marker_size_add,
                    markeredgewidth=0.25, markeredgecolor='k',
                    linewidth=0.5, linestyle="none",
                    ax=axf2[e_ax_n])
                score_label.extend([i_score]*n_settings)

        axg[e_ax].grid(True, axis='both', lw=0.25)
        axg[e_ax].set_xlabel('', fontsize=font_size, labelpad=1)
        axg[e_ax].set_ylabel('occurrence [%]', fontsize=font_size, labelpad=1)
        title_p = ' '.join(i_phase.split('_')[:-1])
        axg[e_ax].set_title(title_p, fontsize=font_size, pad=2.0)
        axg[e_ax].set_ylim(0, 100)
        axg[e_ax].get_legend().set_visible(False)
        sns.despine(ax=axg[e_ax])

    leg_handles, leg_label = axg[e_ax].get_legend_handles_labels()
    rem_line_b = [site_settings_index[site_label.index(x)].replace(':', ' ')
                   for x in leg_label]
    leg_label = [
        f"{x}: {y}" for x, y in zip(leg_label, rem_line_b)]

    if plot_baydz_TPFP:
        leg_handles2, leg_label2 = axf2[e_ax_n].get_legend_handles_labels()
        leg_label2 = [
            f'{x}: {y}' for x, y in zip(leg_label2, score_label)]
        leg_handles.expand(leg_handles2)
        leg_label.expand(leg_label2)

    axg[e_ax].legend(leg_handles, leg_label)
    sns.move_legend(
        axg[e_ax], loc="upper left", bbox_to_anchor=(0.0, -2.2),
        fontsize=font_size_leg)

    fig_v1_11.savefig(
        os.path.join(PATH_EXPORT,
                     f"{file_suffix}_metrics_TPperc_shapes.pdf"),
        format='pdf', bbox_inches="tight")

    return


def plot_metrics_test_patch_stats_prec_recall_TP(
        test_plot_inp, test_TP_FP_plot_inp,
        test_geom_counts_plot_inp, test_TP_FP_f1_plot_inp,
        test_prec_recall_plot_inp,
        site_settings_index, metrics_plot, test_phases,
        set_annot_text, metrics_title, site_label,
        cat_order, alpha_bar, col_set, col_set_bright,
        col_scatter, n_cv_nums,
        font_size, font_size_leg,
        PATH_EXPORT, file_suffix, fig_size=None,
        plot_baydz_TPFP=False, wspace=2.5, hspace=0.8, err_bar=None,
        average_test=False, markers_size=None, markers_test=None):

    if fig_size is None:
        fig_size = (9.5, 4)
    if err_bar is None:
        err_bar = ("pi", 100)  # min max

    cv_hue_order = [f'cv{x:02d}' for x in range(n_cv_nums)]

    n_settings = len(site_settings_index)
    marker_size_add = (4-n_settings)*0.5
    axg = []
    # gridspec inside gridspec
    fig_v1_11 = plt.figure(figsize=fig_size)  # (8.27, 2.5)
    gs0 = gridspec.GridSpec(
        2 + 2*len(test_phases), 17, figure=fig_v1_11, wspace=wspace, hspace=hspace)

    # subplot 0 to 2 (micro)
    gs00 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs0[:2, :5], wspace=0)
    axg.extend([fig_v1_11.add_subplot(gs00[:, x]) for x in range(2)])

    # subplot 2 to 7 (class IoU)
    gs00 = gridspec.GridSpecFromSubplotSpec(
        2, 6, subplot_spec=gs0[:2, 5:], wspace=0)
    axg.extend([fig_v1_11.add_subplot(gs00[:, x]) for x in range(6)])

    # subplot 8 to 9 (Baydz bar plot)
    for e, i in enumerate(test_phases):
        n_y = [2*(e+1), 2*(e+2)]
        gs03 = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs0[n_y[0]:n_y[1], :4], wspace=0)
        axg.extend([fig_v1_11.add_subplot(gs03[:2, :])])

    # subplot 10 to 11
    # (bar plot True and false positives and recall, pecision)
    for e, i in enumerate(test_phases):
        n_y = [2*(e+1), 2*(e+2)]
        gs02 = gridspec.GridSpecFromSubplotSpec(
            2, 6, subplot_spec=gs0[n_y[0]:n_y[1], 5:],
            wspace=3.5, hspace=0.1)
        axg.extend(
            [fig_v1_11.add_subplot(gs02[:2, :]),
            ])

    start_subp = 8+len(test_phases)
    # top (IoU) left axis, bottom TP TN baydz, bottom FP TP all classes
    plotting_utils.format_axes_general(
        axg, [0, 8+len(test_phases)-1, start_subp + len(test_phases) - 1],
        dict(labeltop=False, labelbottom=True,
             labelleft=True, labelright=False, labelsize=font_size))

    # top (IoU) no left axis
    plotting_utils.format_axes_general(
        axg, list(range(1, 7)),
        dict(labeltop=False, labelbottom=True,
             labelleft=False, labelright=False, labelsize=font_size))

    # top (IoU) right axis
    plotting_utils.format_axes_general(
        axg, [7],
        dict(labeltop=False, labelbottom=True,
             labelleft=False, labelright=True,
             left=True, right=True, bottom=True, top=False, labelsize=font_size))

    # Bar TP TN baydzherakhs top plots
    plotting_utils.format_axes_general(
        axg, list(range(8, 8+len(test_phases)-1)),
        dict(labeltop=False, labelbottom=False,
             labelleft=True, labelright=False,
             #top=True, bottom=False, left=False, right=True,
             labelsize=font_size))

    # Bar TP FP all classes top plots
    plotting_utils.format_axes_general(
        axg, list(range(start_subp, start_subp + len(test_phases)-1)),
        dict(labeltop=False, labelbottom=False,
             labelleft=True, labelright=False,
             #top=True, bottom=False, left=False, right=True,
             labelsize=font_size), ax_excl_lst=['bottom'])

    sns.set_style(
        "whitegrid",
        {'axes.grid.axis' : 'both', "axes.ticks.visible": True,
         'legend.fontsize':6})

    if markers_test is None:
        markers_test = ['+', 'x', 'o', 'D', '.', '*']
    if markers_size is None:
        markers_size = [4.5, 3.5, 2.0, 2.0, 2.0, 2.0]
    leg_count_lst = []
    leg_label_lst = []
    count_leg = 0
    for e_ax, i_metrics in enumerate(metrics_plot):
        scat_p = test_plot_inp.query('metrics == @i_metrics')
        for e_phase, i_phase in enumerate(test_phases):
            for e_set, i_set in enumerate(site_settings_index):
                col_pal = [col_scatter[e_set][0]]
                if not average_test:
                    sns.pointplot(
                        data=scat_p.query('phase == @i_phase and site_settings == @i_set'),
                        x="set_label", y="val", hue='cv_num',
                        hue_order=cv_hue_order,
                        dodge=.8 - .8 / n_cv_nums,
                        palette=col_pal*n_cv_nums, #label=f'{i_phase}',
                        errorbar=None,
                        markers=markers_test[e_phase],
                        markersize=3.5 + marker_size_add,
                        linewidth=0.75, linestyle="none",
                        ax=axg[e_ax], legend=True)
                    mul_leg = n_cv_nums
                else:
                    sns.pointplot(
                        data=scat_p.query('phase == @i_phase and site_settings == @i_set'),
                        x="set_label", y="val",# hue='cv_num',
                        #hue_order=cv_hue_order,
                        #dodge=.8 - .8 / 1,
                        palette=col_pal, label=f'{i_phase}',
                        errorbar=err_bar, capsize=0.1,
                        err_kws={'color': col_pal[0], 'linewidth': 0.3,
                                 'zorder':1.5, 'alpha':0.8},
                        markers=markers_test[e_phase],
                        markersize=markers_size[e_phase] + marker_size_add + 1,
                        linewidth=0.75, linestyle="none",
                        ax=axg[e_ax], zorder=2, legend=True)
                    mul_leg = 1
                if e_set == 0 and e_ax == 0:
                    leg_count_lst.append(count_leg*mul_leg)
                    leg_label_lst.append(i_phase)
                count_leg += 1
        if e_ax != len(metrics_plot) - 1:
            sns.despine(ax=axg[e_ax])  # removes top and right axis
        else:
            sns.despine(ax=axg[e_ax], top=True, right=False)

    matplotlib.rcParams['legend.fontsize'] = font_size
    for e_ax, i_metrics in enumerate(metrics_plot):
        axg[e_ax].grid(True, axis='both', lw=0.25, zorder=1.0)
        axg[e_ax].set_xlabel('', fontsize=font_size, labelpad=1)
        axg[e_ax].set_ylabel('metrics', fontsize=font_size,
                             labelpad=1)
        axg[e_ax].set_title(metrics_title[e_ax],
                            fontsize=font_size, pad=2.0)
        axg[e_ax].set_ylim(0.0, 1.0)
        try:
            axg[e_ax].get_legend().set_visible(False)
        except:
            pass
    axg[e_ax].yaxis.set_label_position("right")

    leg_handles, leg_label = axg[0].get_legend_handles_labels()
    axg[0].legend(
        [leg_handles[x] for x in leg_count_lst],
        leg_label_lst)

    sns.move_legend(axg[0], loc="upper left",
                    bbox_to_anchor=(0.0, -(1.5 + len(test_phases))),
                                    fontsize=font_size_leg)

    axg[0].annotate(
        set_annot_text, xy=[0, -(1.2 + len(test_phases))],
        xycoords='axes fraction', fontsize=font_size_leg)

    # =================== TP FP
    hue_order_set = ['true'] + ['pred ' + x.replace(':', '_')
                                for x in site_settings_index]
    marker_f1 = ['D', 'o']

    axg2_lower = []  # twin axis for metrics
    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n + len(metrics_plot) + len(test_phases)
        axg2_lower.append(axg[e_ax].twinx())
    leg_text = []
    leg_handles_lst = []
    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n + len(metrics_plot) + len(test_phases)
        counts_plt = test_geom_counts_plot_inp.query('phase ==@i_phase')
        prec_plt = test_prec_recall_plot_inp.query('phase ==@i_phase').reset_index()

        # plot TRUE VALS as barplot (as positive percentage values)
        # true as 100% with annotated area value
        sns.barplot(
            data=counts_plt.query('true_pred == "true"'),
            x="class", y='TRUE_perc', hue="true_pred_set",
            hue_order=['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:],
            order=cat_order, gap=.25, alpha=alpha_bar, dodge=True,
            linewidth=0.0,
            palette=[col_set_bright[0]]*4 + col_set_bright[1:],
            fill=True,
            errorbar=err_bar, err_kws={'color': 'k', 'linewidth': 0.25},
            ax=axg[e_ax], legend=False, zorder=1)


        patch_x_pos = [
            x.get_x()  + x.get_width() / 2. for x in axg[e_ax].patches][:len(cat_order)]
        if 'area_km' not in counts_plt.columns:
            counts_plt.rename({'area': 'area_km'}, axis=1, inplace=True)
        area_query = counts_plt.query('true_pred == "true" and cv_num=="cv00"').reset_index().loc[:, ['class', 'area_km']]
        # there might be duplicates if the model was taken from different "sites" (e.g. HEX_A01_A02, HEX_SPOT_A01_A02)
        # (site here correesponds to the site that has been used for the trainng. Ant path and phase reflect the
        # predicted test patch)
        # also round areas in case there are small missmatches (which is
        # for example tha case for BLyakh_v3 versus v4 as traning area
        # of v3 was initially lipped too much)
        area_query = area_query.round(3)
        area_query.drop_duplicates(inplace=True)
        area_query.set_index('class', inplace=True)
        area_query['perc_fract'] = area_query/area_query.sum()*100
        area_query['true_pred_set'] = 'true'

        patch_area = [
            area_query.loc[x, 'area_km'] for x in cat_order if x in area_query.index]
        # drop duplicates which come form the three differeent sets
        patch_annotate = pd.DataFrame(
            [patch_x_pos, patch_area],
            index=['x_pos', 'area_km']).T.drop_duplicates('x_pos').set_index('x_pos')

        for i_pos in patch_annotate.index:
            area_num = patch_annotate.loc[i_pos, "area_km"]
            axg[e_ax].annotate(
                    f'{area_num:.3f}', (i_pos, 10),
                     ha='center', va='bottom', rotation=90,
                     fontsize=font_size-1)

        #for e_set, i_set in enumerate(site_settings_index):
        # was before ['TRUE_TP_perc', 'TRUE_FP_perc_neg']
        for e_perc, i_perc in enumerate(['TRUE_TP_perc', 'FDR_perc_neg']):
            if e_perc == 0:
                col_s = col_set
            else:
                col_s = col_set_bright
            sns.barplot(
                data=counts_plt.query('true_pred != "true"'),
                x="class", y=i_perc, hue="true_pred_set",
                hue_order=['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:],
                order=cat_order,
                gap=.25, alpha=alpha_bar, dodge=True,
                linewidth=0.0, palette=[col_s[0]]*4 + col_s[1:],
                fill=True,
                errorbar=err_bar,
                err_kws={'color': 'k', 'linewidth': 0.25},
                ax=axg[e_ax], legend=True, zorder=4-e_perc)
        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax].get_legend_handles_labels()
            leg_text.extend(leg_label)
        sns.barplot(
            data=area_query, x="class", y='perc_fract',
            hue="true_pred_set",
            hue_order=['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:],
            order=cat_order,
            gap=.25, alpha=alpha_bar, dodge=True, linewidth=0.0,
            palette=[col_set[0]]*4 + col_set[1:],
            fill=True,
            errorbar=err_bar, err_kws={'color': 'k', 'linewidth': 0.25},
            ax=axg[e_ax], legend=False, zorder=1.2)

        for e_score, i_score in enumerate(['recall', 'prec']):
            hue_order_inp = ['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:]
            sns.pointplot(
                data=prec_plt.rename({'class_name': 'class'}, axis=1).query('metrics==@i_score'),
                x="class", y="val", hue="true_pred_set",
                hue_order=hue_order_inp, order=cat_order,
                log_scale=False, alpha=1,
                dodge=.8 - .8 / len(hue_order_inp), #width=0.5,
                markers=marker_f1[e_score], fillstyle='full',
                markersize=3.5 + marker_size_add,
                markeredgewidth=0.25, markeredgecolor='k',
                linewidth=0.5, linestyle="none",
                palette=[col_set[0]]*4 + col_set[1:],
                errorbar=None,
                ax=axg2_lower[e_ax_n], legend=True, zorder=1.5)

            if e_ax_n == 0:
                # start from 1 because first label is 'True'
                leg_text.extend(
                    [f'{i_score} {x}' for x in  leg_label[1:n_settings+1]])

        sns.despine(ax=axg[e_ax])

        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax].get_legend_handles_labels()
            leg_handles2, leg_label = axg2_lower[e_ax_n].get_legend_handles_labels()
            # start from 1 because first label is 'True'
            leg_handles_lst.extend(
                leg_handles + leg_handles2[1:n_settings+1]
                + leg_handles2[-n_settings:])
        axg[e_ax].get_legend().set_visible(False)

    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n + len(metrics_plot) + len(test_phases)
        ax_param = dict(labeltop=False, labelbottom=False,
                        labelleft=False, labelright=True,
                        left=False, labelsize=font_size)
        plotting_utils.format_axes_general(
            axg2_lower, [e_ax_n], ax_param)

        title_p = ' '.join(i_phase.split('_')[:-1])

        axg[e_ax].grid(True, axis='both', lw=0.25)
        axg[e_ax].set_ylim(-100, 100)
        axg[e_ax].set_ylabel(
            'FDR | TPR [%]', fontsize=font_size, labelpad=0.3)
        axg[e_ax].set_xlabel(
             '', fontsize=font_size, labelpad=1)

        axg2_lower[e_ax_n].grid(False)
        axg2_lower[e_ax_n].set_ylim(-1.0, 1.0)
        # keep only ticks which are between 0 and 1
        yticks = axg2_lower[e_ax_n].get_yticks()
        filtered_xticks = [x for x in yticks if x <=1 and x >= 0]
        axg2_lower[e_ax_n].set_yticks(filtered_xticks)
        axg2_lower[e_ax_n].set_ylabel(
            'metrics', fontsize=font_size,
            labelpad=0.1, rotation=270)
        axg2_lower[e_ax_n].yaxis.set_label_coords(1.12, (0.5/1.2/2 + 0.5))

        axg2_lower[e_ax_n].get_legend().set_visible(False)

        axg[e_ax].set_title(title_p, fontsize=font_size, pad=2.0)
        axg2_lower[e_ax_n].spines['top'].set_linewidth(0.0)
        axg2_lower[e_ax_n].spines['left'].set_linewidth(0.0)
        axg2_lower[e_ax_n].spines['bottom'].set_linewidth(0.0)
        axg[e_ax].spines['bottom'].set_linewidth(0.0)
        axg[e_ax].axhline(0, lw=0.4, color="k", clip_on=False)

    axg[e_ax].legend(leg_handles_lst, leg_text)

    sns.move_legend(
        axg[e_ax], loc="upper left",
        bbox_to_anchor=(0.5, -1.1), fontsize=font_size_leg)

    # ================= plot barplot
    marker_f1 = ['D', 'o']
    marker_fill = ['none', 'full']
    axf2 = []
    score_label = []
    bar_order = ['TP\nW0', 'TP\nW1', 'TP\nW2', 'TN']
    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n + len(metrics_plot) #+ len(test_phases)*2

        TP_FP_plt = test_TP_FP_plot_inp.query('phase ==@i_phase and cts_name!="TN"')
        sns.barplot(data=TP_FP_plt, x="cts_name", y="cts_perc", hue="set_label",
                    order=bar_order,
                    log_scale=False, alpha=alpha_bar, dodge=True, gap=.25,  #width=0.8,
                    linewidth=0.0, palette=col_set[1:],
                    fill=True,
                    errorbar=err_bar,
                    err_kws={'color': 'k', 'linewidth': 0.25},
                    ax=axg[e_ax], legend=True, zorder=2)

        # plot true negatives with different alpha
        TP_FP_plt = test_TP_FP_plot_inp.query('phase ==@i_phase and cts_name=="TN"')
        sns.barplot(data=TP_FP_plt, x="cts_name", y="cts_perc", hue="set_label",
                    order=bar_order,
                    log_scale=False, alpha=alpha_bar, dodge=True, gap=.25,  #width=0.8,
                    linewidth=0.0, palette=col_set_bright[1:],
                    fill=True,
                    errorbar=err_bar,
                    err_kws={'color': 'k', 'linewidth': 0.25},
                    ax=axg[e_ax], legend=True, zorder=2)

        if plot_baydz_TPFP:
            axf2.append(axg[e_ax].twinx()) # applies twinx to ax2, which is the second y axis.
            TP_FP_plt_f1 = test_TP_FP_f1_plot_inp.query('phase ==@i_phase')
            for e_score, i_score in enumerate(['recall', 'prec']):
                sns.pointplot(
                    data=TP_FP_plt_f1.query('score_type ==@i_score'),
                    x="score_xlabel", y="score", hue='set_label',
                    dodge=.8 - .8 / n_settings, legend=True,
                    palette=col_set[1:], errorbar=None,
                    markers=marker_f1[e_score],
                    fillstyle=marker_fill[e_score],
                    markersize=2.5 + marker_size_add,
                    markeredgewidth=0.25, markeredgecolor='k',
                    linewidth=0.5, linestyle="none",
                    ax=axf2[e_ax_n])
                score_label.extend([i_score]*n_settings)

        axg[e_ax].grid(True, axis='both', lw=0.25)
        axg[e_ax].set_xlabel('', fontsize=font_size, labelpad=1)
        axg[e_ax].set_ylabel('occurrence [%]', fontsize=font_size, labelpad=1)
        title_p = ' '.join(i_phase.split('_')[:-1])
        axg[e_ax].set_title(title_p, fontsize=font_size, pad=2.0)
        axg[e_ax].set_ylim(0, 100)
        axg[e_ax].get_legend().set_visible(False)
        sns.despine(ax=axg[e_ax])

    leg_handles, leg_label = axg[e_ax].get_legend_handles_labels()
    rem_line_b = [site_settings_index[site_label.index(x)].replace(':', ' ')
                   for x in leg_label]
    leg_label = [
        f"{x}: {y}" for x, y in zip(leg_label, rem_line_b)]

    if plot_baydz_TPFP:
        leg_handles2, leg_label2 = axf2[e_ax_n].get_legend_handles_labels()
        leg_label2 = [
            f'{x}: {y}' for x, y in zip(leg_label2, score_label)]
        leg_handles.expand(leg_handles2)
        leg_label.expand(leg_label2)

    axg[e_ax].legend(leg_handles, leg_label)
    sns.move_legend(
        axg[e_ax], loc="upper left", bbox_to_anchor=(0.0, -2.2),
        fontsize=font_size_leg)

    fig_v1_11.savefig(
        os.path.join(PATH_EXPORT,
                     f"{file_suffix}_test_metrics_TPperc.pdf"),
        format='pdf', bbox_inches="tight")

    return


def test_patch_plot(
    test_geom_files_all, train_metrics_best,
    site_settings_index, test_phases,
    PATH_EXPORT, file_suffix, cv_num_displ, dict_assign,
    n_test_patches=6, fontsize=8, metrics_name_title='jacc_micro',
    test_patch_lst=None):
    '''
    dict_assign = PARAM['dict_assign']
    '''
    label_kws = dict(
        labelbottom=False, labeltop=False, labelleft=False,
        labelright=False,
        bottom=True, top=True, left=True, right=True)

    if not isinstance(n_test_patches, list):
        n_test_patches_inp = [n_test_patches]
    else:
        n_test_patches_inp = n_test_patches


    # ===== plot tifs
    # initialize figure
    # columns are: raw: with labels, predicted with labels
    # twos are amount of test patches

    #test_patches = [f'test-{x:02d}' for x in range(1, n_test_patches + 1)]
    #n_patches = len(test_patches)
    #grid = [n_patches, len(site_settings_index) + 1]  # test patches x settings
    #count_p = np.ravel(np.arange(grid[0]*grid[1]).reshape(grid[1], grid[0]).T)
    #share_axis = None#
    #figsize = (2*grid[1], 2*grid[0])

    col_map_combined = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", [y[0] for x, y in dict_assign.items()])

    for e_phase, i_phase in enumerate(test_phases):
        if test_patch_lst is not None:
            test_patches = test_patch_lst[i_phase]
        else:
            test_patches = [f'test-{x:02d}' for x in range(1, n_test_patches_inp[e_phase] + 1)]
        n_patches = len(test_patches)
        grid = [n_patches, len(site_settings_index) + 1]  # test patches x settings
        count_p = np.ravel(np.arange(grid[0]*grid[1]).reshape(grid[1], grid[0]).T)
        share_axis = None

        figsize = (2*grid[1], 2*grid[0])

        ax, fig, plot_nr = plotting_utils.initialize_plot_custom(
            grid, [1]*n_patches, [1]*(grid[1]), count_p,
            fontsize=fontsize, share_axis=share_axis,
            figsize=figsize, aspect_ratio='equal',
            l_top=False, l_bot=False, l_left=False, l_right=False,
            e_left=0.05, e_right=0.95, e_bottom=0.05, e_top=0.95,
            e_wspace=0.1, e_hspace=0.1)
        plotting_utils.adjust_tick_params(
            fig, ax_lw=0.5, pad=3.0, font_size=8, axis_col='k',
            tick_spacing_maj=100, tick_spacing_min=None,
            tick_spacing_majy=100, tick_spacing_miny=None,
            tick_length=2, label_kws=label_kws)
        count = 0
        coun_patch = 0
        for i_patch in test_patches:
            ax[count].grid(False)
            # read raw image
            path_raw_true = test_geom_files_all.loc[(slice(None), slice(None), slice(None), slice(None), i_phase, i_patch), ['path_raw', 'path_true']].values[0, :]
            raw_img = rioxarray.open_rasterio(path_raw_true[0] + '.tif', mask_nan=True).squeeze()

            true_gdf = geopandas.read_file(path_raw_true[1] + '.geojson')
            # do not plot stabilized
            true_gdf = true_gdf.loc[true_gdf['class'] != 3, :]
            true_gdf.set_index('class', inplace=True)

            # plot raw file
            xarray.plot.imshow(
                raw_img, ax=ax[count], #aspect='equal',
                add_colorbar=False, vmin=0, vmax=255, add_labels=False,
                cmap='Greys_r', interpolation='none')

            # plot ground truth
            plotting_utils.plot_class_boundaries(
                true_gdf, dict_assign, ax[count], linewidth=0.75)
            count += 1

            for i_set in site_settings_index:
                ax[count].grid(False)
                if coun_patch == 0:
                    t_set = i_set.split(':')
                    epoch_num = test_geom_files_all.loc[(i_set, cv_num_displ, slice(None), slice(None), i_phase, i_patch), 'epoch'].values[0]
                    try:
                        metrics_val = train_metrics_best.loc[(slice(None), slice(None), i_set, cv_num_displ, slice(None), slice(None), slice(None), 'validate'), metrics_name_title].values[0]
                    except:
                        # e.g for ensemle tests have no training metrics
                        metrics_val = 99
                    try:
                        epoch_str = '{0:02d}'.format(epoch_num)
                    except:
                        epoch_str = epoch_num

                    title = f"{t_set[0]}\n{t_set[1]} epoch: {epoch_str}\nvalidation {metrics_name_title.replace('_', ' ')}: {metrics_val:0.2f}"
                    ax[count].set_title(title, fontsize=6)
                # get paths
                path_pred = test_geom_files_all.loc[(i_set, cv_num_displ, slice(None), slice(None), i_phase, i_patch), 'path_pred'].values[0]

                # read prediction
                pred_img = rioxarray.open_rasterio(
                    path_pred + '.tif', mask_nan=True).squeeze()
                pred_img = pred_img.rio.reproject_match(
                    raw_img, Resampling=Resampling.nearest)
                # reproject match t make sure that img fit on top of each other

                # plot raw as base
                xarray.plot.imshow(
                    raw_img, ax=ax[count], #aspect='equal',
                    add_colorbar=False, vmin=0, vmax=255,
                    add_labels=False, cmap='Greys_r', interpolation='none')

                # set stabiliyed to nan
                pred_img = pred_img.where(pred_img != 3, np.nan)
                xarray.plot.imshow(
                    pred_img, ax=ax[count], #aspect='equal',
                    add_colorbar=False, vmin=1, vmax=6, add_labels=False,
                    alpha=0.6, cmap=col_map_combined, interpolation='none')

                # plot snow and ponds only
                pred_img_p_s = pred_img.where(pred_img >= 4, np.nan)
                xarray.plot.imshow(
                    pred_img_p_s, ax=ax[count], #aspect='equal',
                    add_colorbar=False, vmin=1, vmax=6, add_labels=False, #alpha=1.0,
                    cmap=col_map_combined, interpolation='none')
                # plot ground truth on top
                plotting_utils.plot_class_boundaries(
                    true_gdf, dict_assign, ax[count],  # true_gdf.loc[true_gdf.index !=5, :]
                    col='k', linewidth=0.3)
                count += 1
            coun_patch += 1

        file_name_out = os.path.join(
            PATH_EXPORT, f'{file_suffix}_pred_on_{i_phase}')
        fig.savefig(file_name_out + '.pdf', format='pdf')
        fig.savefig(file_name_out + '.svg', format='svg')
        fig.savefig(file_name_out + '.png', format='png')

        del fig, ax

    return


def get_ponds_TP_FP(ponds_true, ponds_pred):

    ponds_max_px = np.ceil(max(ponds_pred.px.max(), ponds_true.px.max()))
    group_index = ['site', 'settings', 'site_settings', 'cv_num',
                   'folder', 'phase', 'px_bin']

    ponds_FN = ponds_true.query('aoi_intersect_area == 0')
    ponds_FN['px_bin'] = pd.cut(ponds_FN['px'], bins=[0, 3, 6, 10, 20, 50, 100, ponds_max_px])
    ponds_FN_counts = ponds_FN.groupby(group_index, observed=True).count()['class'].to_frame(name='counts')

    ponds_FP = ponds_pred.query('aoi_intersect_area == 0')
    ponds_FP['px_bin'] = pd.cut(ponds_FP['px'], bins=[0, 3, 6, 10, 20, 50, 100, ponds_max_px])
    ponds_FP_counts = ponds_FP.groupby(group_index, observed=True).count()['class'].to_frame(name='counts')

    ponds_TP = ponds_pred.query('aoi_intersect_area > 0')
    ponds_TP['px_bin'] = pd.cut(ponds_TP['px'], bins=[0, 3, 6, 10, 20, 50, 100, ponds_max_px])
    ponds_TP_counts = ponds_TP.groupby(group_index, observed=True).count()['class'].to_frame(name='counts')

    return ponds_FN_counts, ponds_FP_counts, ponds_TP_counts


def plot_metrics_geom_ponds(
        test_metrics_geom_plot_inp,
        test_geom_plot_inp,
        test_geom_counts_plot_inp, ponds_true, ponds_pred,
        site_settings_index, test_phases,
        set_annot_text,
        cat_order, alpha_bar, alpha_violon, col_set, col_set_bright,
        font_size, font_size_leg,
        PATH_EXPORT, file_suffix, fig_size=None,
        wspace=2.5, hspace=0.8,
        bar_plot='px',
        ylim_inp=None, log_scale=False,
        tick_interval=50, err_bar=None):
    '''
    plot area distribution as well as total area and pond distibution as
    counts (intersecting or non intersecting and depending on size)

    bar_plot can be:
        area_m, or px is input is test_geom_sum_plot_inp
    or it can be geometry if input is test_geom_counts_plot_inp
    '''

    if fig_size is None:
        fig_size = (9.5, 4)
    if err_bar is None:
        err_bar = ("pi", 100)  # min max

    ponds_FN_counts, ponds_FP_counts, ponds_TP_counts = get_ponds_TP_FP(
        ponds_true, ponds_pred)

    n_settings = len(site_settings_index)
    n_test = len(test_phases)
    marker_size_add = (4-n_settings)*0.5
    axg = []
    # gridspec inside gridspec
    fig_v1_11 = plt.figure(figsize=fig_size)  # (8.27, 2.5)
    gs0 = gridspec.GridSpec(
        len(test_phases), 8, figure=fig_v1_11, wspace=wspace,
        hspace=hspace)

    # subplots for violin plot
    for e, i in enumerate(test_phases):
        #n_y = [2*(e+1), 2*(e+2)]
        gs02 = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=gs0[e, :5],
            wspace=3.5, hspace=0.1)
        axg.extend(
            [fig_v1_11.add_subplot(gs02[:2, :]),
             fig_v1_11.add_subplot(gs02[2:, :])
             ])
    for e, i in enumerate(test_phases):
        gs04 = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=gs0[e, 5:], wspace=0)
        axg.extend([fig_v1_11.add_subplot(gs04[0, :]),
                    fig_v1_11.add_subplot(gs04[1, :]),
                    fig_v1_11.add_subplot(gs04[2, :])])

    # bottom
    plotting_utils.format_axes_general(
        axg, [n_test*2-1, n_test*(2+3)-1],
        dict(labeltop=False, labelbottom=True,
             labelleft=True, labelright=False, labelsize=font_size))

    # top (IoU) no left axis
    plotting_utils.format_axes_general(
        axg, list(range(0, n_test*2-1)) + list(range(n_test*2, n_test*(2+3)-1)),
        dict(labeltop=False, labelbottom=False,
             labelleft=True, labelright=False, labelsize=font_size))

    sns.set_style(
        "whitegrid",
        {'axes.grid.axis' : 'both', "axes.ticks.visible": True,
         'legend.fontsize':6})

    # =================== violon plot
    hue_order_set = ['true'] + ['pred ' + x.replace(':', '_')
                                for x in site_settings_index]

    width_v = [1.4]*n_settings
    # leg_text = ['true']
    inner = 'quart'
    gap_shift = [0.1]*n_settings
    marker_f1 = ['D', 'o']

    leg_text = []
    leg_handles_lst = []
    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n*2
        geom_plt = test_metrics_geom_plot_inp.query('phase ==@i_phase').reset_index()
        geom_plt1 = test_geom_plot_inp.query('phase ==@i_phase')
        counts_plt = test_geom_counts_plot_inp.query('phase ==@i_phase')

        for e_set, i_set in enumerate(site_settings_index):
            if e_set > 0:
                geom_plt_filt = geom_plt.query(
                    'site_settings==@i_set and true_pred != "true"')
                col_set_inp = [col_set[e_set + 1]]
            else:
                geom_plt_filt = geom_plt.query('site_settings==@i_set')
                col_set_inp = [col_set[0]] + [col_set[e_set + 1]]
                if e_ax_n == 0:
                    leg_text.append('True')

            sns.violinplot(
                data=geom_plt_filt, x="class", y="area_m", hue="true_pred",
                hue_order=['true', 'pred'], order=cat_order, split=True,
                gap=gap_shift[e_set], inner=inner, log_scale=True,
                alpha=alpha_violon, dodge=True, width=width_v[e_set],
                fill=True, linewidth=0.25, palette=col_set_inp,
                ax=axg[e_ax], legend=False, zorder=1)
            # for line
            sns.violinplot(
                data=geom_plt_filt, x="class", y="area_m", hue="true_pred",
                hue_order=['true', 'pred'], order=cat_order, split=True,
                gap=gap_shift[e_set], inner=inner, log_scale=True,
                alpha=1., dodge=True, width=width_v[e_set], fill=False,
                linewidth=0.75, palette=col_set_inp, ax=axg[e_ax],
                legend=True, zorder=1.1)
            # for fine line
            sns.violinplot(
                data=geom_plt_filt, x="class", y="area_m", hue="true_pred",
                hue_order=['true', 'pred'], order=cat_order,
                split=True, gap=gap_shift[e_set], inner=inner,
                log_scale=True, alpha=1.0, dodge=True,
                width=width_v[e_set], fill=False, linewidth=0.1,
                color='k', linecolor='k', ax=axg[e_ax], legend=False,
                zorder=1.2)
            # mark line of 3 pixel size
            axg[e_ax].axhline(1.5*1.5*3, lw=0.5, color="k", clip_on=False,
                              linestyle='--', dashes=(2, 4, 2, 4))

            if e_ax_n == 0:
                leg_text.append('pred ' + i_set.replace(':', '_'))
        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax].get_legend_handles_labels()
            leg_handles_lst.extend(leg_handles[:1] + leg_handles[1::2])

        sns.barplot(
            data=counts_plt, x="class", y=bar_plot, hue="true_pred_set",
            hue_order=['', ''] + [hue_order_set[0]] + [''] + hue_order_set[1:],
            order=cat_order, gap=.25, alpha=alpha_bar,
            dodge=True, linewidth=0.0,
            palette=[col_set[0]]*4 + col_set[1:], fill=True,
            errorbar=err_bar, err_kws={'color': 'k', 'linewidth': 0.25},
            ax=axg[e_ax + 1], legend=True, zorder=1)
        if log_scale:
            axg[e_ax + 1].set_yscale('log')
        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax + 1].get_legend_handles_labels()
            leg_text.extend(leg_label)

        if e_ax_n == 0:
            leg_handles, leg_label = axg[e_ax + 1].get_legend_handles_labels()
            leg_handles_lst.extend(leg_handles)
        axg[e_ax + 1].get_legend().set_visible(False)
        sns.despine(ax=axg[e_ax])
        sns.despine(ax=axg[e_ax + 1])

    axg[e_ax].annotate(
        set_annot_text, xy=[0, -3.2], xycoords='axes fraction',
        fontsize=font_size_leg)

    for e_ax_n, i_phase in enumerate(test_phases):
        e_ax = e_ax_n*2

        title_p = ' '.join(i_phase.split('_')[:-1])
        axg[e_ax].grid(True, axis='both', lw=0.25)
        axg[e_ax].set_ylim(0.01, 10**7)
        axg[e_ax].set_ylabel(
            r'area [$\mathregular{m^2}$]', fontsize=font_size,
            labelpad=1)
        axg[e_ax].get_legend().set_visible(False)

        axg[e_ax].set_xlabel(
             '', fontsize=font_size, labelpad=1)

        axg[e_ax].set_title(title_p, fontsize=font_size, pad=2.0)
        #axg[e_ax].spines['bottom'].set_linewidth(0.0)
        axg[e_ax].spines['right'].set_linewidth(0.0)
        #axg[e_ax].tick_params(labelright=False, which='both', width=0.0)

        axg[e_ax + 1].grid(True, axis='both', lw=0.25)
        if bar_plot == 'area_m':
            if ylim_inp is None:
                ylim_inp = (0.01, 10**7)

            axg[e_ax + 1].set_ylim(*ylim_inp)
            axg[e_ax + 1].set_ylabel(
                r'area [$\mathregular{m^2}$]', fontsize=font_size,
                labelpad=1)
        elif bar_plot == 'px':
            if ylim_inp is None:
                ylim_inp = (00, 500000)
            axg[e_ax + 1].set_ylim(*ylim_inp)
            axg[e_ax + 1].set_ylabel(
                'px count', fontsize=font_size, labelpad=0.5)
        else:
            axg[e_ax + 1].set_ylabel(
                'shape count', fontsize=font_size, labelpad=0.5)
        axg[e_ax + 1].set_xlabel(
            'classes', fontsize=font_size, labelpad=1)


    axg[e_ax].legend(leg_handles_lst, leg_text)
    #axg[e_ax + 1].legend(leg_handles_lst, leg_text)

    sns.move_legend(axg[e_ax], loc="upper left",
                    bbox_to_anchor=(0.0, -1.5), fontsize=font_size_leg)

    # --- counts plot
    max_cts = max(ponds_FN_counts['counts'].max(),
                  ponds_FP_counts['counts'].max(),
                  ponds_TP_counts['counts'].max())
    for e_ax_n, i_phase in enumerate(test_phases):
        title_p = ' '.join(i_phase.split('_')[:-1])
        e_ax = e_ax_n*3 + len(test_phases)*2

        ponds_FN_counts_p = ponds_FN_counts.query('phase ==@i_phase')
        ponds_FP_counts_p = ponds_FP_counts.query('phase ==@i_phase')
        ponds_TP_counts_p = ponds_TP_counts.query('phase ==@i_phase')

        # plot amount of shapes
        sns.barplot(
            data=ponds_TP_counts_p,
            x='px_bin', y='counts', hue='site_settings', dodge=True,
            gap=.25,
            errorbar=err_bar,
            err_kws={'linewidth': 0.5, 'color': 'k'},
            hue_order=site_settings_index,
            alpha=alpha_bar,
            linewidth=0.0, palette=col_set[1:],
            ax=axg[e_ax], legend=False, zorder=1)
        sns.barplot(
            data=ponds_FP_counts_p,
            x='px_bin', y='counts', hue='site_settings', dodge=True,
            gap=.25,
            errorbar=err_bar, err_kws={'linewidth': 0.5, 'color': 'k'},
            hue_order=site_settings_index,
            alpha=alpha_bar,
            linewidth=0.0, palette=col_set[1:],
            ax=axg[e_ax + 1], legend=False, zorder=1)
        sns.barplot(
            data=ponds_FN_counts_p,
            x='px_bin', y='counts', hue='site_settings', dodge=True,
            gap=.25,
            errorbar=err_bar, err_kws={'linewidth': 0.5, 'color': 'k'},
            hue_order=site_settings_index,
            alpha=alpha_bar,
            linewidth=0.0, palette=col_set[1:],
            ax=axg[e_ax + 2], legend=True, zorder=1)

        axg[e_ax].set_title(
                title_p, fontsize=font_size, pad=2.0)
        title_suffix = ['TP', 'FP', 'FN']
        for i in range(3):
            sns.despine(ax=axg[e_ax + i])
            axg[e_ax + i].set_yticks(np.arange(0, max_cts+tick_interval/2, tick_interval))
            axg[e_ax + i].set_ylim(0.0, max_cts)
            axg[e_ax + i].grid(True, axis='both', lw=0.25)
            # axg[e_ax + len(test_phases)].set_ylim(-100, 100)
            axg[e_ax + i].set_ylabel(
                f'{title_suffix[i]} cts', fontsize=font_size, labelpad=0.5)
            axg[e_ax + i].set_xlabel(
                'shape size range [px]', fontsize=font_size, labelpad=1)
        plt.setp(axg[e_ax + i].get_xticklabels(), rotation=90, ha='right')  # Rotate labels in-place

        axg[e_ax + 2].get_legend().set_visible(False)

    fig_v1_11.savefig(
        os.path.join(PATH_EXPORT,
                     f"{file_suffix}_test_shape_counts_TPFP.pdf"),
        format='pdf', bbox_inches="tight")

    return