"""
Input to select specific segmentation frameworks to be compared and plotted
(used in MAIN_model_comparison_plots.py)
"""
import numpy as np
import pandas as pd


def get_param_sets():
    '''
    Defines sets to compare

    structure of dictionary is:
    dict is {output_name: [[id_name_of_set, {PARAM_PREP_ID}{PARAM_TRAIN_ID}{aug_itentifier}, TYPE_NAME]]}
    e.g.: {'Test_summary_sets_ov0-1':
            [
                ['BLyakhE_v4_HEX_SPOT_A01_A02_set01a', 'v079t16onl', 'DL'],
            ]
            }

    '''
    site_sets_dict = {
        'BLyaE_v1_HEX_A02_metrics_ML17t16v079v158_ov0-1':
            [
                ['BLyaE_v1_HEX1979_A02', 'vML017tML02', 'ML'],
                ['BLyaE_v1_HEX1979_A02', 'v079t16onl', 'DL'],
                ['BLyaE_v1_HEX1979_A02', 'v158t16onl', 'DL'],
                ['BLyaE_v1_HEX1979_A02_v079t16_finet_FadN', 'v079t16onl', 'DL'],
                # ['BLyaE_v1_HEX1979_A02_v079t16_finet_BLyaS', 'v079t16onl', 'DL']
            ],
        }

    # if want to merge some site specific fine tuining
    # sublists {original_site_name:
    #                   [lst_test_phases_to_use,
    #                    new_site_name,
    #                    settings_name]  # settings_name must be the same for the ones that are merged
    model_merge_dict = {
        #'BLyaE_v1_HEX1979_A02_v079t16_finet_FadN':
        #    [['FadN_HEX1980_test'],
        #    'BLyaE_v1_HEX1979_A02_v079t16_finet',
        #    'v079t16onl'],
        #'BLyaE_v1_HEX1979_A02_v079t16_finet_BLyaS':
        #    [['BLyaS_HEX1979_test'],
        #    'BLyaE_v1_HEX1979_A02_v079t16_finet',
        #    'v079t16onl']
        }
    final_sets_dict = {
        #'BLyaE_v1_HEX_A02_test_summary_ML17t16v079v158_ov0-1':
        #   [
        #       ['BLyaE_v1_HEX1979_A02', 'vML017tML02'],
        #       ['BLyaE_v1_HEX1979_A02', 'v079t16onl'],
        #       ['BLyaE_v1_HEX1979_A02', 'v158t16onl'],
        #       ['BLyaE_v1_HEX1979_A02_v079t16_finet', 'v079t16onl'],
        #         ],
        }
    return site_sets_dict, model_merge_dict, final_sets_dict


def extract_folder_from_param(site_sets_dict):
    folder_all = [
        [x.split(':')[0], '_'.join(y.split(':')), str(z)]
            for x, y, z in np.hstack(list(site_sets_dict.values()))]
    folder_all = [x + [['cv00', 'cv01', 'cv02']]
                  if x[-1] != 'ensemble' else x + ['cv00']
                  for x in folder_all]
    folder_df = pd.DataFrame(
        folder_all, columns=['site', 'folder', 'seg_type', 'cv_num'])
    folder_df = folder_df.explode('cv_num')
    folder_df.loc[folder_df.seg_type != 'ensemble', 'folder'] = folder_df.loc[folder_df.seg_type != 'ensemble', ['folder', 'cv_num']].apply(
        lambda x: f"{x.folder}_{x.cv_num}", axis=1)
    folder_df.drop_duplicates(inplace=True)
    folder_df.set_index(['site', 'seg_type'], inplace=True)
    folder_df.sort_index(inplace=True)

    return folder_df