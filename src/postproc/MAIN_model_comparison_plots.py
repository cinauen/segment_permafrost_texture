'''
# ===== Create comparison plots =======
Creates comparison plots on the test patch metrics.

Before running this script:
- adjust INPUT PARAMETERS further below
- edit the combination of segmentation frameworks to compare in:
    src/postproc/postproc/plot_param_sets.py in get_param_sets()

!!! SITE corresponds to the site on which the model was trained on!!!
(SITE does NOT correspond to the test area. The test area is defined by the
phase)

'''

import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = "2" # visible in this process
import sys
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# scatter plot new
plt.rcdefaults() # restore [global] defaults
matplotlib.rcParams['pdf.fonttype'] = 42



# ----------------- import specific modules -----------------
MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MODULE_PATH)

import postproc.model_comparison_plots as model_comparison_plots
import postproc.plot_param_sets as plot_param_sets

# ----------------------- define paths -----------------------
# set project paths (if required change here or change add_base_path.py)
BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, BASE_PATH)
import example.add_base_path
path_proc_inp, path_temp_inp = example.add_base_path.main()

PATH = os.path.join(path_proc_inp, '2_segment/02_train')
PATH_INP = os.path.join(path_proc_inp, '2_segment/01_input')


# ------------------- define INPUT PARAMETERS -----------------------
PATH_EXPORT = os.path.join(
    path_proc_inp, '2_segment/03_analyse/ML_DL_comparison')
if not os.path.isdir(PATH_EXPORT):
    os.makedirs(PATH_EXPORT)


# suffix to be added to output files
suffix_add = ''

# add parameters to dict
PARAM = {}
PARAM['PATH_EXPORT'] = PATH_EXPORT
PARAM['EPSG_TARGET'] = 32654

# path to input imagery of test patches
PATH_IMG = os.path.join(path_proc_inp, '1_site_preproc')
PARAM['TEST_path'] = {
    'BLyaE_HEX1979_test': os.path.join(
        PATH_IMG, 'BLyaE_v1', '03_train_inp',
        'BLyaE_HEX1979_test_perc0-2_g0-3_8bit_Lv01_untiled_v00'),
    'FadN_HEX1980_test': os.path.join(
        PATH_IMG, 'FadN_v1', '03_train_inp',
        'FadN_HEX1980_test_histm_b0-5aRv1_8bit_Lv01_untiled_v00'),
    }

# =========
# --- plot settings
site_sets_dict, model_merge_dict, final_sets_dict = plot_param_sets.get_param_sets()

metrics_plot = ['jacc_micro', 'dice_micro', 'jacc_class03_stable_areas',
                'jacc_class01_baydherakhs', 'jacc_class02_ridges_ponds',
                'jacc_class04_gully_base', 'jacc_class05_ponds',
                'jacc_class06_snow']

metrics_title = ['mIoU\nmicro', 'dice\nmicro', 'IoU\nstable_areas',
                 'IoU\nbaydzh', 'IoU\nponding\nareas',
                 'IoU\ngully\nbase', 'IoU\nponds', 'IoU\nsnow']

# !!!! do not use duplicates in the list below.
# otherwise will have duplicated entries !!!
test_phases = [
               'BLyaE_HEX1979_test',
               'FadN_HEX1980_test',
               #'FadN_SPOT2018_test',
               ]
cat_order = ['stable_areas', 'baydzh', 'ponding\nareas',
             'gully\nbase', 'ponds', 'snow']

# ==== other inputs
# overfit threshold to find best two epochs
overfit_threshold = 0.1
metrics_decide = 'jacc_macro'

PARAM['CLASS_LABELS'] = [
        [0, 1, 2, 3, 4, 5, 6],
        ['nan', 'baydherakhs', 'ridges_ponds', 'stable_areas',
         'gully_base', 'ponds', 'snow']]

class_label_dict = {
    1: 'baydzh', 2: 'ponding\nareas', 3: 'stable_areas',
    4: 'gully\nbase', 5: 'ponds', 6: 'snow'}  # for relabelling in dataframe
class_label_dict_rename = {
    'baydherakhs': 'baydzh', 'ridges_ponds': 'ponding\nareas',
    'stable_areas': 'stable_areas', 'gully_base': 'gully\nbase',
    'ponds': 'ponds', 'snow': 'snow'}

# min and max epoch for overview boxplot stats
min_epoch = 30  # from which epoch to use for scatter plot
max_epoch = 100  # not used currently

# class number used to evaluate weights with TN TP (are baydzherakhs)
class_weight_eval = 1

# cv num to use for test patch plot
cv_num_plot_test = 'cv00'

# exclude forders from file search
exclude_folder = [
    ]

# extract folders from site_sets_dict:
folder_df = plot_param_sets.extract_folder_from_param(site_sets_dict)

# folder name of CNN training results
# dictionary key is id_name_of_set from site_sets_dict,
#    value is the actual folder name
PARAM['SITES_DL'] = {
    'BLyaE_v1_HEX1979_A02':
        'BLyaE_v1_HEX1979_A02',
    'BLyaE_v1_HEX1979_A02_v079t16_finet_FadN':
        'BLyaE_v1_HEX1979_A02_v079t16_finet_FadN',
    }
# same as above for ML training
PARAM['SITES_ML'] = {
    'BLyaE_v1_HEX1979_A02': 'BLyaE_v1_HEX1979_A02_ML_rt',
    }

# define test patches for specific test phase (which is also the folder name)
PARAM['TEST_dict'] = {
    'BLyaE_HEX1979_test': [f'test-{x:02d}' for x in range(1, 5)],
    'FadN_HEX1980_test': [f'test-{x:02d}' for x in range(1, 5)],
    }

# in case the file name patch prefix differs from the folder name this
# can be added here {foldername: test patch prefix in file, ...}
# file naming of test patches is
# v079t16onl_cv00_ncla7_BLyaE_HEX1979_test-04_ep64_class_pred
# {identifier for framework setup}_{test patch prefix}-{test patch number}_ep{XX}_class_pred
PARAM['TEST_dict_DL_patch_prefix'] = {
    'BLyaE_HEX1979_test':'BLyaE_HEX1979_test',
    'FadN_HEX1980_test': 'FadN_HEX1980_test',
    }

test_dict_inp = {x: PARAM['TEST_dict'][x] for x in test_phases}
n_test_patches = [len(PARAM['TEST_dict'][x]) for x in test_phases]

# metrics to read and concatenate
metrics_lst = [
    'acc_micro', 'dice_micro', 'f1_macro',
    'jacc_micro', 'jacc_macro',
    'precision_macro', 'recall_macro',
    'jacc_class01_baydherakhs', 'jacc_class02_ridges_ponds',
    'jacc_class03_stable_areas', 'jacc_class04_gully_base',
    'jacc_class05_ponds', 'jacc_class06_snow']

PARAM['dict_assign'] = {
    1: ['#FFEA86', 'baydherakhs'],
    2: ['#882255', 'ridges_ponds'],
    3: ['#A7A3A6', 'stable_areas'],
    4: ['#2330D0', 'gully_base'],
    5: ['#A2F6FB', 'ponds'],
    6: ['#B157F5', 'snow']}


# ======= Find results folders DL ==========
TRAIN_INFO_DL = model_comparison_plots.get_results_folder_DL(
    folder_df, exclude_folder,
    PARAM['SITES_DL'], PATH, PATH_EXPORT)

# =========== Read all DL train metrics files
train_metrics_DL, train_model_save_DL, train_cm_DL, train_classIoU_DL = model_comparison_plots.extract_metrics_train_DL(
    TRAIN_INFO_DL, PATH, PARAM['SITES_DL'], metrics_lst)

# get diff between train and validate (as indicator for overfit)
epoch_diff = train_metrics_DL.loc[:, 'train'] - train_metrics_DL.loc[:, 'validate']
train_model_save_DL_checkp = train_model_save_DL.query('if_checkp == True')

# ====== extract metrics at checkpoint
train_metrics_DL_checkp = train_metrics_DL.loc[train_model_save_DL_checkp.index]
epoch_diff_DL_checkp = (train_metrics_DL_checkp.loc[:, 'train']
                        - train_metrics_DL_checkp.loc[:, 'validate'])

# ===== extract epochs with best metrics and overfit below overfit max
# overfit_threshold, metrics_decide
train_metrics_DL_checkp[('eval', metrics_decide + '_diff')] = epoch_diff_DL_checkp[metrics_decide]

# extract best epochs but below overfit
# best epochs will be used for scatter plot and for test
grouped = train_metrics_DL_checkp.groupby(['site', 'folder'], group_keys=False)
epoch_best = grouped.apply(
    lambda x: (x.loc[x[('eval', metrics_decide + '_diff')] < overfit_threshold, :]).nlargest(
        1, columns=('validate', metrics_decide)))
epoch_best.sort_index(inplace=True)

# check that have one good epoch per cv
print(epoch_best.loc[:, ('validate', metrics_decide)].groupby(
    ['site', 'settings'], group_keys=False).count())
print(epoch_best.reset_index('epoch').describe())

# =========== Read all DL test metrics files
(test_metrics_DL, test_classCM_DL, test_metrics_geom_pred_DL,
 test_metrics_geom_true_DL, test_geom_files_DL,
 df_FN_TN_perc_DL, df_FN_TN_f1_DL) = model_comparison_plots.extract_metrics_test_DL(
        TRAIN_INFO_DL, epoch_best, model_merge_dict,
        test_dict_inp, PATH, PARAM['TEST_path'], PARAM['SITES_DL'],
        PARAM['TEST_dict_DL_patch_prefix'], metrics_lst,
        class_weight_eval)

# =========== Find results folders ML
TRAIN_INFO_ML = model_comparison_plots.get_results_folder_ML(
    folder_df, exclude_folder,
    PARAM['SITES_ML'], PATH)


if len(TRAIN_INFO_ML) > 0:

    # =========== Read all ML train metrics files
    train_metrics_ML, train_classIoU_ML, train_classCM_ML = model_comparison_plots.extract_metrics_train_ML(
        TRAIN_INFO_ML, PATH, PARAM['SITES_ML'], metrics_lst)

    (test_metrics_ML, test_classCM_ML, test_metrics_geom_pred_ML,
              test_metrics_geom_true_ML, test_geom_files_ML,
              df_FN_TN_perc_ML, df_FN_TN_f1_ML) = model_comparison_plots.extract_metrics_test_ML(
        TRAIN_INFO_ML, test_dict_inp, PATH, PARAM['TEST_path'], PARAM['SITES_ML'],
        PARAM['TEST_dict_DL_patch_prefix'], metrics_lst,
        class_weight_eval)


# ========= Merge ML and DL metrics and geom
# Merge best train metrics for DL and ML
if len(TRAIN_INFO_ML) > 0:
    train_metrics_best = pd.concat(
        [epoch_best.drop('eval', axis=1).stack('phase', future_stack=True).reset_index('epoch'),
        train_metrics_ML.stack('phase', future_stack=True).query('phase in ["train", "validate"]')], axis=0)
    # replace nan epochs (to avoid problems if use groupby, nans in groupby indes seem to be dropped)
    train_metrics_best.loc[np.isnan(train_metrics_best.epoch), 'epoch'] = -1
    train_metrics_best.set_index('epoch', append=True, inplace=True)

    # merge traim metrics cm
    validate_CM_all = pd.concat(
        [train_cm_DL.loc[:, 'validate'].reset_index('epoch'),
         train_classCM_ML.loc[:, 'validate']], axis=0)
    # replace nan eppochs (to avoid problems if use groupby, nans in groupby indes seem to be dropped)
    validate_CM_all.loc[np.isnan(validate_CM_all.epoch), 'epoch'] = -1
    validate_CM_all = validate_CM_all.set_index('epoch', append=True).swaplevel('epoch', 'class_name')

    # merge test metrics of DL and ML
    test_metrics_all = pd.concat(
        [test_metrics_DL.loc[:, 'test'].reset_index('epoch'),
         test_metrics_ML.loc[:, 'test'].query('phase not in ["train", "validate"]')], axis=0)
    # replace nan eppochs (to avoid problems if use groupby, nans in groupby indes seem to be dropped)
    test_metrics_all.loc[np.isnan(test_metrics_all.epoch), 'epoch'] = -1
    #test_metrics_all.set_index('epoch', append=True, inplace=True)
    test_metrics_all.reset_index('phase', inplace=True)
    test_metrics_all['site_phase'] = test_metrics_all['phase'].apply(lambda x: x.split('_')[0])
    test_metrics_all.set_index(['site_phase', 'phase', 'epoch'], append=True, inplace=True)
    test_metrics_all.sort_index(inplace=True)

    # merge test CM of DL and ML
    # for test selected only best metrics Thus no need to care about epochs....
    test_CM_all = pd.concat(
        [test_classCM_DL.loc[:, 'test'],
         test_classCM_ML.loc[:, 'test'].query(
            'phase not in ["train", "validate"]')], axis=0).sort_index(axis=1).sort_index(axis=0)
    # test_CM_all = test_CM_all.set_index('epoch', append=True).swaplevel('epoch', 'class_name')

    test_geom_files_all = pd.concat(
        [test_geom_files_DL, test_geom_files_ML], axis=0).reset_index(
            ['site', 'settings', 'folder'])
else:
    train_metrics_best = epoch_best.drop('eval', axis=1).stack(
        'phase', future_stack=True).reset_index('epoch')
    train_metrics_best.set_index('epoch', append=True, inplace=True)

    # merge test metrics of DL and ML
    test_metrics_all = test_metrics_DL.loc[:, 'test'].reset_index(['phase', 'epoch'])
    test_metrics_all['site_phase'] = test_metrics_all['phase'].apply(
        lambda x: x.split('_')[0])
    test_metrics_all.set_index(
        ['site_phase', 'phase', 'epoch'], append=True, inplace=True)
    test_metrics_all.sort_index(inplace=True)

    validate_CM_all = train_cm_DL.loc[:, 'validate'].sort_index(axis=1).sort_index(axis=0)

    # merge test CM of DL and ML
    test_CM_all = test_classCM_DL.loc[:, 'test'].sort_index(axis=1).sort_index(axis=0)

    test_geom_files_all = test_geom_files_DL.reset_index(
            ['site', 'settings', 'folder'])

validate_prec_recall_all = model_comparison_plots.get_FP_TP_recall_prec_from_CM(
    validate_CM_all, class_label_dict_rename)

test_prec_recall_all = model_comparison_plots.get_FP_TP_recall_prec_from_CM(
    test_CM_all, class_label_dict_rename)

# ----- merge geom measures for DL and ML
if len(TRAIN_INFO_ML) > 0:
    test_metrics_geom_pred = pd.concat(
        [test_metrics_geom_pred_DL, test_metrics_geom_pred_ML], axis=0)
    test_metrics_geom_true = pd.concat(
        [test_metrics_geom_true_DL, test_metrics_geom_true_ML], axis=0)
else:
    test_metrics_geom_pred = test_metrics_geom_pred_DL
    test_metrics_geom_true = test_metrics_geom_true_DL

# ------ test_metrics_geom_pred
test_metrics_geom_pred['true_pred'] = 'pred'
ind_names = test_metrics_geom_pred.index.names
test_metrics_geom_pred.reset_index(inplace=True)
test_metrics_geom_pred['true_pred_set'] = test_metrics_geom_pred.loc[:, ['true_pred', 'site_settings']].apply(
    lambda x: x['true_pred'] + ' ' + x['site_settings'].replace(':', '_'), axis=1)
test_metrics_geom_pred.set_index(ind_names, inplace=True)

test_metrics_geom_true['true_pred'] = 'true'
test_metrics_geom_true['true_pred_set'] = 'true'

test_metrics_geom = pd.concat(
    [test_metrics_geom_true, test_metrics_geom_pred], axis=0)

# ------- rename class names
test_metrics_geom = test_metrics_geom.set_index(
    'class', append=True).rename(index=class_label_dict, level='class').reset_index('class')

# ------ merge TN FN files
if len(TRAIN_INFO_ML) > 0:
    test_TP_FP = pd.concat([df_FN_TN_perc_DL, df_FN_TN_perc_ML], axis=0)
    test_TP_FP_f1 = pd.concat([df_FN_TN_f1_DL, df_FN_TN_f1_ML], axis=0)
else:
    test_TP_FP = df_FN_TN_perc_DL
    test_TP_FP_f1 = df_FN_TN_f1_DL

if len(model_merge_dict) > 0:
    site_sets_dict = final_sets_dict


# ============= Loop throught dict sets for plotting ==============
for i_set_name, i_set_lst_subl in site_sets_dict.items():
    i_set_lst = [':'.join(x[:2]) for x in i_set_lst_subl]

    alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
    site_label = alphabet[:len(i_set_lst)]
    set_relabel = {y: x for x, y in zip(site_label, i_set_lst)}
    n_settings = len(i_set_lst)

    # Extract specific inputs for metrics plots
    (box_plot_inp, epoch_best_inp, test_plot_inp, test_TP_FP_plot_inp,
    test_TN_FN_plot_inp_neg, test_TP_FP_f1_plot_inp, test_metrics_geom_plot_inp,
    test_geom_counts_plot_inp, test_geom_sum_plot_inp, test_prec_recall_plot_inp,
    test_geom_counts_plot_inp_filt, validate_metrics_prec_recall) = model_comparison_plots.extract_inputs_for_metrics_plots(
            train_metrics_DL, train_metrics_best, test_metrics_all,
            test_TP_FP, test_TP_FP_f1, test_metrics_geom,
            validate_prec_recall_all, test_prec_recall_all,
            i_set_lst,
            metrics_plot, set_relabel, min_epoch, test_phases)

    ponds_true, ponds_pred = model_comparison_plots.extract_pond_counts(
        test_metrics_geom, i_set_lst, test_phases, EPSG_num=PARAM['EPSG_TARGET'])

    # for validation and recall take only the best epochs
    validate_metrics_prec_recall.reset_index(['class_name', 'true_pred_set', 'set_label'], inplace=True)
    validate_metrics_prec_recall = validate_metrics_prec_recall.loc[train_metrics_best.query(
        'phase=="validate" and site_settings in @i_set_lst').reset_index('phase').index, :]
    validate_metrics_prec_recall.set_index(['class_name', 'true_pred_set', 'set_label'], append=True, inplace=True)

    # -------- save comparison stats
    epoch_99 = model_comparison_plots.get_last_epoch_metrics(
        train_metrics_DL.query('site_settings in @i_set_lst'),
        'train', PATH_EXPORT, prefix_out=f'{i_set_name}_{suffix_add}')

    model_comparison_plots.save_metrics_summary(
        train_metrics_best.query('site_settings in @i_set_lst'),
        'train-validation', PATH_EXPORT,
        epoch_diff=epoch_99, prefix_out=f'{i_set_name}_{suffix_add}')

    model_comparison_plots.save_metrics_summary(
        test_metrics_all.query('site_settings in @i_set_lst'),
        'test', PATH_EXPORT, prefix_out=f'{i_set_name}_{suffix_add}')

    model_comparison_plots.save_TP_geom_metrics(
        validate_metrics_prec_recall, test_geom_sum_plot_inp,
        test_TP_FP_f1_plot_inp, test_TP_FP_plot_inp,
        PATH_EXPORT, f'{i_set_name}_{suffix_add}')


    # -- get epoch list
    best_epoch_df = epoch_best_inp.set_index(['site_settings', 'cv_num']).loc[:, 'epoch'].reset_index().drop_duplicates(subset=['site_settings', 'cv_num']).set_index('site_settings')
    # merged set is missing
    set_missing = np.setdiff1d(i_set_lst, best_epoch_df.reset_index().site_settings.unique())

    epoch_lst = []
    for i_set in i_set_lst:
        try:
            epo = best_epoch_df.loc[i_set, 'epoch'].tolist()
            epoch_lst.append(
                ' '.join([f'{xx:.0f}' if not np.isnan(xx) else '' for xx in epo]))
        except:
            # for merged sets e.g. for sets with fine tuning with different inputs
            # need to specify epoch per initial training set
            df_inp = test_metrics_all.reset_index('epoch').loc[(slice(None), slice(None), i_set), 'epoch'].reset_index().drop_duplicates(subset=['site_settings', 'cv_num', 'site_phase'])#['epoch'].tolist()
            lst_inp = [f'{x}: {y}' for x, y in zip(df_inp['site_phase'].tolist(),
                                                   df_inp['epoch'].tolist())]
            epoch_lst.append(' '.join(lst_inp))

    epoch_lst_annot = [f'| epochs {x}' for x in epoch_lst]
    # annotation text
    repl_line_b = [x.replace(':', ' ') for x in i_set_lst]
    set_annot_text = '\n'.join(
        [f"{x}: {y} {z}"
         for x, y, z in zip(site_label, repl_line_b, epoch_lst_annot)])

    # epoch_best_inp
    cv_nums = np.unique(epoch_best_inp.cv_num)
    n_cv_nums = len(cv_nums)

    # get colors
    col_scatter, col_set, col_set_bright, col_greys, col_scaling, col_fine_tune = model_comparison_plots.get_metrics_colormap(
        n_settings)
    if 'fine_tune' in suffix_add:
        col_scatter = col_fine_tune[0] + [col_fine_tune[0][-1]]
        col_set = col_fine_tune[1] + [col_fine_tune[1][-1]]
        col_set_bright = col_fine_tune[2] + [col_fine_tune[2][-1]]
    else:
        col_scatter = col_scaling[0]
        col_set = col_scaling[1]
        col_set_bright = col_scaling[2]

    # ### Stats plot with stripplot and non-overlapping viololin and bar
    # plot and f1 scores
    font_size = 8
    font_size_leg = 8
    alpha_bar = 0.7
    alpha_violon = 0.2  # 0.2
    model_comparison_plots.plot_metrics_test_patch_stats_prec_recall_TP_violon(
        test_plot_inp, test_TP_FP_plot_inp, test_metrics_geom_plot_inp,
        test_geom_sum_plot_inp, test_TP_FP_f1_plot_inp,
        test_prec_recall_plot_inp,
        i_set_lst, metrics_plot, test_phases,
        set_annot_text, metrics_title, site_label,
        cat_order, alpha_bar, alpha_violon, col_set,
        col_set_bright, col_scatter, n_cv_nums, font_size, font_size_leg,
        PATH_EXPORT, f'{i_set_name}_{suffix_add}_test_patches',
        fig_size=(9.0, 3.0 + len(test_phases)*2),
        wspace=1.0, hspace=0.8, average_test=True)

    model_comparison_plots.plot_metrics_test_patch_stats_prec_recall_TP_violon(
        test_plot_inp, test_TP_FP_plot_inp, test_metrics_geom_plot_inp,
        test_geom_sum_plot_inp, test_TP_FP_f1_plot_inp,
        test_prec_recall_plot_inp,
        i_set_lst, metrics_plot, test_phases,
        set_annot_text, metrics_title, site_label,
        cat_order, alpha_bar, alpha_violon, col_set,
        col_set_bright, col_scatter, n_cv_nums, font_size, font_size_leg,
        PATH_EXPORT, f'{i_set_name}_{suffix_add}_test_patches_averaged',
        fig_size=(9.0, 3.0 + len(test_phases)*2),
        wspace=1.0, hspace=0.8, average_test=False)


    # without violon plot
    model_comparison_plots.plot_metrics_test_patch_stats_prec_recall_TP(
        test_plot_inp, test_TP_FP_plot_inp,
        test_geom_sum_plot_inp, test_TP_FP_f1_plot_inp,
        test_prec_recall_plot_inp,
        i_set_lst, metrics_plot, test_phases,
        set_annot_text, metrics_title, site_label,
        cat_order, alpha_bar, col_set,
        col_set_bright, col_scatter, n_cv_nums, font_size,
        font_size_leg, PATH_EXPORT, f'{i_set_name}_{suffix_add}_test_patches',
        fig_size=(9.0, 3.0 + len(test_phases)*1.8),
        wspace=1.0, hspace=0.8, average_test=True)


    # plot area distribution as well as total area and pond distibution as
    # counts (intersecting or non intersecting and depending on size)
    model_comparison_plots.plot_metrics_geom_ponds(
        test_metrics_geom_plot_inp,
        test_geom_sum_plot_inp,
        test_geom_sum_plot_inp, ponds_true, ponds_pred,
        i_set_lst, test_phases,
        set_annot_text,
        cat_order, alpha_bar, alpha_violon, col_set,
        col_set_bright,
        font_size, font_size_leg,
        PATH_EXPORT, f'{i_set_name}_{suffix_add}', fig_size=(8, 6.0),
        wspace=1.8, hspace=0.2,
        bar_plot='area_m',
        ylim_inp=(0.1, 1.2*10**6), log_scale=False)


    model_comparison_plots.test_patch_plot(
        test_geom_files_all, train_metrics_best,
        i_set_lst, test_phases,
        PATH_EXPORT, f'{i_set_name}_{suffix_add}_test_patches',
        cv_num_plot_test, PARAM['dict_assign'],
        n_test_patches=n_test_patches, fontsize=8,
        metrics_name_title=metrics_decide, test_patch_lst=PARAM['TEST_dict'])
