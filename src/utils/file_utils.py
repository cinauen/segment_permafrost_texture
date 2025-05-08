
'''
Module for file handling:
- save parameter files
- save dictionary to text
- change file names and remove extension

'''

import os
import pandas as pd
import datetime as dt



def write_param_file(
        file_path, file_name, param_dict, manual_inputs=None, how='a',
        close=False):
    '''
    manual_inputs: dictionary with separately defined input parameters

    param_dict: dict with subdict specifying different parameter types
        created e.g. with:
        keys = ['PROJ', 'PARAM', 'PLOT']
        param_lst = [PARAM.PROJ, PARAM.PARAM, PARAM.PLOT]
        param_dict = {x:y for x, y in zip(names, param_lst)}
    '''

    # --------------  save parameter values -----------------
    param_log = open(os.path.join(file_path, file_name), how)
    param_log.write(
        '# ==========================================================\n'
        + '#  Parameter Log: ' + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + '\n# ==========================================================\n')

    # save manual inputs
    if manual_inputs is not None:
        param_log.write('# ---- manual inputs ---\n')
        for i_key, i_val in manual_inputs.items():
            param_log.write(i_key + ': ' + str(i_val) + '\n')

    # save input from parametr files
    for ii, i in param_dict.items():
        param_log.write('# ----- ' + ii + ' ----- \n')
        param_log.write(save_dict_to_text(i))
        param_log.write('\n')

    param_log.flush()
    if close:
        param_log.close()

    return param_log


def save_dict_to_text(dict_data, keys_dict=None):
    '''
    Converts dictionaries to text
    (e.g. to save Parameters to file)

    keys_dict: list of specific keys which should be saved
    '''
    if not keys_dict:
        keys_dict = list(dict_data.keys())

    text = []
    for key, val in dict_data.items():
        if isinstance(val, dict):
            text.append('\n-- dict: ' + key + '\n')
            for key_s, val_s in val.items():
                text.append('%s: %s \n' % (key_s, val_s))
            text.append('--\n')
        else:
            text.append('%s: %s \n' % (key, val))
        if isinstance(val, pd.DataFrame):
            text.append('\n')
    text = ''.join(text)

    return text


def remove_file_extension(file_name):
    '''
    returns output without "."
    could also use file_name.split('.')[0]
    however, there would be a problem is there is another point in the
    path somewhere...
    '''
    return file_name[:-(file_name[::-1].find('.') + 1)]


def merge_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def merge_dict_lst(dict_lst):
    return {k: v for d in dict_lst for k, v in d.items()}


def create_filename(inp_lst):
    if '' in inp_lst:
        inp_lst.remove('')
    return '_'.join(inp_lst)



