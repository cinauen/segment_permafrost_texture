'''
Define folders as env variable
'''

import os

def main():

    import os

    # --- define main base folders (adjust path if required)
    # processing folder
    path_base = os.path.normpath(
        './example/')

    # ---- folder for temporary files (best use local foder, network folders
    # can cause issues when removing the folder)
    path_temp = os.path.normpath(
        './example/temp')


    os.environ['PROC_BASE_PATH'] = path_base
    os.environ['PROC_TEMP_PATH'] = path_temp

    return path_base, path_temp


if __name__ == "__main__":

    path_base, path_temp = main()

    # print command such that the it can be executed in the bash shell
    # and be available there (with eval $(python add_base_path.py))
    print(f'export PROC_BASE_PATH={os.environ["PROC_BASE_PATH"]}')
    print(f'export PROC_TEMP_PATH={os.environ["PROC_TEMP_PATH"]}')

