'''
==== Texture calculation on full image (proc step 02) =====

This script uses the library cupy-glcm (Julio Faracco)
(https://github.com/Eve-ning/glcm-cupy)
(must be run on GPUs)

Calculates GLCM texture features. To avoid memory overflow on thew GPUs,
the image is split into subtiles.

MAIN INPUTS (defined in site specific parameter file):
- Scaled imagery defined by scale type e.g. 'perc0-2_g0-3_8bit'
  -->  .../02_pre_proc/BLyaE_HEX1979_scale_perc0-2_g0-3_8bit.tif
- AOI within which texture should be claculated
  e.g. for full area .../01_input/BLyaE_processing_area_AOI_32654.geojson
- AOI sub-areas for calculating statistics (stats can later be used
  for standardisation during training)

PARAMETER FILES (paths to the parameter file must be provided as
function arguments):
- Site specific parameters:
    e.g. .../01_input/PROJ_PARAM_BLyaE_HEX1979_perc02_g03_8bit_v01.py
- Static parameters: seg_param/PARAM02_calc_texture.py
    For example texture paramters are defind in this file (e.g. directions
    and window size in PARAM['TEX_PARAM'])

OUTPUT
- Raster file with calculated textures (separate file per winow and
   direction set from PARAM['TEX_PARAM'])
- Metadata with list of created files: e.g. .../02_pre_proc/BLyaE_HEX1979_P02_???.txt


---- RUN script in command line:
usage: MAIN_calc_texture.py [-h] [--PARAM_FILE PARAM_FILE] [--GPU_DEVICE GPU_DEVICE] PATH_PROC PROJ_PARAM_FILE SCALE_TYPE

Calculate GLCM texture of GeoTIFFs

positional arguments:
  PATH_PROC                 prefix to processing folder and proc file
                            (required, e.g. "./docs/1_site_preproc/BLyaE_v1")
  PROJ_PARAM_FILE           file with project parameters
                            (required, e.g.: PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01)
  SCALE_TYPE                type of image scaling to be used (e.g. std4_8bit)

optional arguments:
  -h, --help                show this help message and exit
  --PARAM_FILE PARAM_FILE
                            name of task specific param file
                            (optional, default: PARAM02_calc_texture)
  --GPU_DEVICE GPU_DEVICE   GPU device number
                            (optional, default: 0)

--- for debugging use:
preproc/test/MAIN_img_preproc_test.py

'''

import os
import sys
import numpy as np
import datetime as dt
import argparse
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import gc
import logging

import memory_profiler

# -------------- PATHS & GLOBAL PARAM -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import utils.file_utils as file_utils
import utils.monitoring_utils as monitoring_utils
import utils.plotting_utils as plotting_utils
import utils.geo_utils as geo_utils
import param_settings.param_utils as param_utils


@monitoring_utils.conditional_profile
def main(inp):
    import texture_calc.texture_utils as texture_utils
    # -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # --- processing step identifier
    PROC_STEP = 2

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_all_params(inp, PROC_STEP)
    PARAM['LOG_FILE_SUFFIX'] = 'texture_calc'

    # --------------  save parameter values -----------------
    param_file_name = (
        f'A_{PARAM["FILE_PREFIX"]}_{PARAM["LOG_FILE_SUFFIX"]}_PARAM.txt')
    file_utils.write_param_file(
        PARAM['PATH_EXPORT'], param_file_name, {'all_param': PARAM})

    # --------------- setup logging ---------------
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{PARAM['FILE_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_EXPORT'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']}, Proc step {PROC_STEP}")

    # ----------------- GET OVERALL AOI to clip image --------------
    if PARAM['AOI_TEX_clip'] is not None:
        AOI_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
            os.path.join(PARAM['PATH_INP'], PARAM['AOI_TEX_clip']),
            PARAM['EPSG_TARGET'])
    else:
        AOI_coords, AOI_poly = None, None

    # -------------  initialize texture class --------------
    # for texture calculation images can be transferred to lower bits (per default use 4 bits)
    # this can provide better results since otherwisemight get too sparse matrix
    # padding is used to avoid edge effects
    texture = texture_utils.TextureCalc(
        PARAM['EPSG_TARGET'], PARAM['PATH_EXPORT'], PARAM['FILE_PREFIX'],
        padding_const=PARAM['PADDING_CONST'],
        bin_from=PARAM['BIN_FROM'], bin_to=PARAM['BIN_TO'],
        tex_suffix=PARAM['TEX_SUFFIX'])

    # ------------- read and preproc image file ----------
    logging.info(
        '\n---- read and preproc image: ' + PARAM["FILE_INP"] + ' '
        + dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    # with RESOLUTION_target=0 resolution is not changed since use
    # pre-processed image
    # reading os image is done with mask_nan due to cupy-glcm
    img_key = texture.preproc_img(
        PARAM['PATH_EXPORT'], PARAM['FILE_INP'],
        AOI_coords=AOI_coords, RESOLUTION_target=0,
        mask_nan=True)

    # ----- save preprocessed input image
    if PARAM['SAVE_PREPROC_IMG']:
        texture.image_save_several('proc_type', 'img')

    # ---- convert image to numpy array with dimension [y, x, bands] --
    # will be used as input for glcm_cupy
    texture.img_to_nparray(img_key)

    # ---- get stats of raw imagery for the predefined AOIs ---
    # PARAM['AOI_stats_calc'] is dict defind in PROJ_PARAM file
    texture.get_stats_set(
        ['raw'], aoi_dict=PARAM['AOI_stats_calc'],
        path_inp=PARAM['PATH_INP'],
        epsg_target=PARAM['EPSG_TARGET'])


    # ---- calculate texture for different window sizes and texture as
    # defined by PARAM['TEX_PARAM'] in PARAM02_calc_texture
    for i in PARAM['TEX_PARAM']:
        # Texture is calculated file by file. Parallelisation is not
        # done on file level due to large file sizes. But GLCM calculation
        # is parallized on subimage level
        logging.info(
            '\n---- Calc texture with param: ' + str(i) + '  '
            + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        texture.setup_window_param(*i)
        # calculate texture by splitting image into subimages of size 500 x 500
        # texture calcuation is done on sub-numpy arrays which are then
        # concatenated and saved as texture.texture_np dict
        texture.derive_texture_loop_GPU(500, parallel=False)

        logging.info(
            '\n---- Convert texture array to image and save tif: '
            + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        # create xarray from texture numpy array and save image
        texture.texture_to_img_save(
            AOI_poly=AOI_poly, save=True)

        logging.info(
            '\n---- Start calc stats: '
            + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        # calcuate teture statistics for full area as well as for sub
        # areas as specified in PARAM['AOI_stats_calc'] (these are later
        # used for standardising tiles if choose e.g. standardisation
        # based on training area stats)
        texture.get_stats_set(
            np.setdiff1d(list(texture.image.keys()), ['raw']),
            aoi_dict=PARAM['AOI_stats_calc'],
            path_inp=PARAM['PATH_INP'],
            epsg_target=PARAM['EPSG_TARGET'])

        # save and plot texture to pdf
        # C01 is for channel 1 (there could be several channels
        # for cross texture...)
        if PARAM['PLOT_TEX_DISTIBUTION']:
            for i_key in texture.image.keys():
                if i_key =='raw':
                    continue
                plotting_utils.overview_plot_texture(
                    texture.image[texture.img_inp_key].values[0, :, :],
                    texture.image[i_key].values,
                    texture.tex_measures,
                    PARAM['FILE_PREFIX'] + '_' + texture.prefix_add + '_C01',
                    PARAM['PATH_EXPORT'], min_perc=1, max_perc=99)
                # plots histogram of all texture features into one plot
                texture.add_img_hist_plot_hv(
                    i_key, np.nan, 256,
                    plot_name=i_key, range_tuple=(0, 1),
                    overlay_plot=True)
                texture.save_img_hist_plot_hv(
                    i_key, col_num=1,
                    fig_size=None, overlay_plot=True)

        # delete texture_np, and image image
        texture.update_save_metadata('tex', proc_nr_str='P02')
        texture.del_texture_and_img()

    # ====== calculate cross stats ==========
    # calculate e.g. standarddeviation for textures calculated using
    # different moving window directions
    logging.info(
            '\n---- Start cross texture calc: '
            + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    if PARAM['PARALLEL']:
        def inp_loop():
            for i_key, i_file_list in PARAM['img_cross_calc'].items():
                meta_data_file = texture.image_meta_df.query('name in @i_file_list')
                yield(meta_data_file, i_key)

        n_jobs = min(int(cpu_count()/10), len(PARAM['img_cross_calc']))
        w = Parallel(n_jobs=n_jobs, verbose=0)(delayed(
            texture_utils.read_calc_cross_image_stats)(*k, PARAM)
            for k in inp_loop())
        image_meta_out, stats_out = map(list, zip(*w))
        texture.image_meta_df = pd.concat(
            [texture.image_meta_df] + image_meta_out, axis=0).drop_duplicates(keep='last')
        [texture.stats.update(x) for x in stats_out]
    else:
        img_meta_out_lst = []
        for i_key, i_file_list in PARAM['img_cross_calc'].items():
            meta_data_file = texture.img_proc.query('name in @i_file_list')
            image_meta_out, stats_out = texture.read_calc_cross_image_stats(
                meta_data_file, i_key, PARAM)
            texture.stats.update(stats_out)
            img_meta_out_lst.append(image_meta_out)

        texture.image_meta_df = pd.concat(
            [texture.image_meta_df] + img_meta_out_lst, axis=0).drop_duplicates(keep='last')

    texture.del_texture_and_img()
    gc.collect()

    logging.info(
        '\n---- Finalised stats cross calc. Save metadata and stats DataFrame: '
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        + '\n\n')

    texture.update_save_metadata('tex', proc_nr_str='P02')

    # save stats
    df_stats = pd.concat(texture.stats, axis=0, names=['name'])
    df_stats.reset_index(inplace=True)
    df_stats[['name', 'aoi']] = df_stats['name'].str.split(":",expand=True)
    df_stats.set_index('name', inplace=True)
    df_stats.loc[df_stats['aoi'].values == None, 'aoi'] = PARAM['AOI_TEX_SUFFIX']
    aoi_lst = [PARAM['AOI_TEX_SUFFIX']] + list(PARAM['AOI_stats_calc'].keys())
    for i in aoi_lst:
        file_name = (
            f"{PARAM['PROJ_PREFIX']}_{i}_{PARAM['SCALE_TYPE']}_P02_tex_stats_file.txt")

        path_file = os.path.join(PARAM['PATH_EXPORT'], file_name)
        df_stats_save = df_stats.query('aoi == @i')
        try:
            df_stats_save.query('aoi == @i').to_csv(
                path_file, sep='\t', lineterminator='\n', header=True)
        except:
            df_stats_save.to_csv(path_file, sep='\t', header=True)


    # ---------- save time measure ----------
    time_file = f'A_{PARAM["FILE_PREFIX"]}_{PARAM["LOG_FILE_SUFFIX"]}'
    monitoring_utils.save_time_stats(prof, PARAM['PATH_EXPORT'], time_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate GLCM texture of GeoTIFFs')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder and proc file'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('file with project parameters'))
    parser.add_argument(
        'SCALE_TYPE', type=str,
        help='type of image scaling to be used (e.g. std4_8bit)')
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file',
        default='PARAM02_calc_texture')
    parser.add_argument(
        '--GPU_DEVICE', type=int,
        help='GPU device number', default=0)

    args = parser.parse_args()

    # For GPU device initialization use environmental variable. This
    # ensures that only the specified device is used.
    # This is required because importing cucim (if installed) within
    # glcm_cupy seems to initialize all GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vars(args)['GPU_DEVICE'])
    import cupy
    with cupy.cuda.Device(0):  # use zero here since only one device visible
        main(vars(args))
