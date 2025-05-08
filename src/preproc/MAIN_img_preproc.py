
'''
==== Image pre-processing (proc step 01) =====
Tests different scaling, quantizing (e.g. to 8bit) and histogram matching
(is a site specific processing step and run e.g. in example/1_site_preproc/BLyaE_v1)


----------
INPUT (defined in site specific parameter file)
- Full area AOI: e.g. .../01_input/BLyaE_processing_area_AOI_32654.geojson
- Raw, georeferenced input imagery: e.g. .../01_input/BLyaE_HEX1979_D3C1215-301069F003_georef_v1.tif

PARAMETER FILES (link and paths to the parameter file must be provided as
function arguments):
- Site specific parameters:
    e.g. .../01_input/PROJ_PARAM_BLyaE_HEX1979_perc02_g03_8bit_v01.py
- Static parameters: seg_param/PARAM01_img_preproc.py
    Scaling or hist match options are defined in this file
    as dictionaries (PARAM['SCALING_DICT'], PARAM['HIST_MATCH_DICT'])

OUTPUT:
- Scaled imagery: e.g. .../02_pre_proc/BLyaE_HEX1979_scale_perc0-2_g0-3_8bit.tif
- Histogram plots: e.g. .../02_pre_proc/BLyaE_HEX1979_hist_8bit.pdf
- Metadata with list of created files: e.g. .../02_pre_proc/BLyaE_HEX1979_P01_img_proc_file.txt


---- RUN script in command line:
usage: MAIN_img_preproc.py [-h] [--PARAM_FILE PARAM_FILE] [--PARALLEL_PROC PARALLEL_PROC] PATH_PROC PROJ_PARAM_FILE

Pre-process single images

positional arguments:
  PATH_PROC             prefix to processing folder
                        (required, e.g. "./docs/1_site_preproc/BLyaE_v1")
  PROJ_PARAM_FILE       name of project/site specific parameter file
                        (required, e.g.: PROJ_PARAM_BLyaE_HEX1979_scale_perc02_g03_8bit_v01)

options:
  -h, --help            show this help message and exit
  --PARAM_FILE PARAM_FILE
                        name of task specific param file
                        (optional, default: PARAM01_img_preproc)
  --PARALLEL_PROC PARALLEL_PROC
                        if want to use parallel processing
                        (optional, default: 1)

--- for debugging use:
preproc/test/MAIN_img_preproc_test.py

'''
import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import logging
import argparse
from joblib import Parallel, delayed, cpu_count
import matplotlib.pyplot as plt

import memory_profiler

# ----------------- import custom utils -----------------
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import utils.file_utils as file_utils
import utils.monitoring_utils as monitoring_utils
import utils.geo_utils as geo_utils
import utils.image_preproc as image_preproc
import param_settings.param_utils as param_utils


@monitoring_utils.conditional_profile
def main(inp):
    # -------------- setup time control ----------------
    prof = monitoring_utils.setup_time_control()

    # --- processing step identifier
    PROC_STEP = 1

    # scale type is not required in this processing step
    inp['SCALE_TYPE'] = 'NA'

    # -----initialize all paramters and save them for backup
    PARAM = param_utils.initialize_all_params(inp, PROC_STEP)
    # --------------  save parameter values -----------------
    param_file_name = (
        f"A_{PARAM['FILE_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_PARAM.txt")
    file_utils.write_param_file(
        PARAM['PATH_EXPORT'], param_file_name, {'all_param': PARAM})


    # ------------------ setup logging -----------------
    # redirect memory profiling output to console
    sys.stdout = memory_profiler.LogFile('log')  # @profile decorator needs to be running

    # create log file and initialize errors to be written to console
    log_file_name = (
        f"A_{PARAM['FILE_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}_error.log")
    monitoring_utils.init_logging(
        log_file=os.path.join(PARAM['PATH_EXPORT'], log_file_name),
        logging_step_str=f"{PARAM['LOG_FILE_SUFFIX']}, Proc step {PROC_STEP}")

    # ----------------- GET OVERALL AOI to clip image --------------
    # get AOI to clip image
    if PARAM['AOI_clip'] is not None:
        AOI_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
            os.path.join(PARAM['PATH_INP'], PARAM['AOI_clip']),
            PARAM['EPSG_TARGET'])
    else:
        AOI_coords, AOI_poly = None, None

    # --- for histogram matching setup reference file and AOI for
    # clipping the reference file if needed
    ref_img = {}
    if PARAM['REF_IMG_PATH'] is not None:
        if isinstance(PARAM['REF_IMG_PATH'], str):
            # if only one reference file provided as string
            ref_img['Rv1'] = geo_utils.read_to_xarray(
                PARAM['REF_IMG_PATH'], mask_nan=False, chunk='auto')
            if PARAM['AOI_REF_FILE_CLIP'] is not None:
                # clip reference file
                AOI_REF_FILE = os.path.join(
                    PARAM['PATH_INP'], PARAM['AOI_REF_FILE_CLIP'])
                AOI_REF_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
                        AOI_REF_FILE, PARAM['EPSG_TARGET'])
                ref_img['Rv1'] = geo_utils.clip_to_aoi(
                    ref_img['Rv1'], AOI_REF_coords, from_disk=True)
        else:
            for i_key, i_ref_path in  PARAM['REF_IMG_PATH'].items():
                # if several reference files provided as dict
                ref_img[i_key] = geo_utils.read_to_xarray(
                    i_ref_path, mask_nan=False, chunk='auto')

                if PARAM['AOI_REF_FILE_CLIP'][i_key] is not None:
                    # clip reference file
                    AOI_REF_FILE = os.path.join(
                        PARAM['PATH_INP'], PARAM['AOI_REF_FILE_CLIP'][i_key])
                    AOI_REF_coords, AOI_poly = geo_utils.read_transl_geojson_AOI_coords_single(
                        AOI_REF_FILE, PARAM['EPSG_TARGET'])
                    ref_img[i_key]  = geo_utils.clip_to_aoi(
                        ref_img[i_key], AOI_REF_coords, from_disk=True)
    else:
        ref_img = None


    # ---------------- IMAGE processing -------------------
    img_proc = image_preproc.ImagePreproc(
        PARAM['EPSG_TARGET'], PARAM['PATH_EXPORT'],
        PARAM['FILE_PREFIX'])

    # ----- preprocess image
    # (preprocessing is done on xarray image (self.image[key])
    # it doesn't make a difference if mask_nan is True or False)
    img_key = img_proc.preproc_img(
        PARAM['PATH_INP'], PARAM['FILE_INP'],
        AOI_coords=AOI_coords, mask_nan=False,
        RESOLUTION_target=PARAM['RESOLUTION_TARGET'])
    # save preprocessed file
    img_proc.image_save_several('name', img_key)
    img_proc.update_meta_df()

    # create numpy array from image. This is done since the input for
    # scikit image are numpy arrays with dimensions [y, x, bands]
    # (instead for xarray image it is [bands, y, x])
    # !!! For memory improvements could delete self.image and just
    # keep coordinates and attributes to later create new image
    img_proc.img_to_nparray(img_key)
    nodata_inp = img_proc.image[img_key].rio.nodata

    # create hist plot of intensity distrubtion of raw image
    img_proc.add_img_hist_plot_hv(
        'raw', nodata_inp, 256, plot_name='raw')
    img_proc.save_img_hist_plot_hv('raw')

    # ---------- scale image intensities with different parameters
    # parameters are defined in PARAM['SCALING_DICT']
    for i_name, i_scale in PARAM['SCALING_DICT'].items():
        if PARAM['PARALLEL_PROC'] == 1:
            # get number of parallel jobs
            n_jobs = min(int(np.ceil(cpu_count()/10)), len(i_scale))
            # scale images in parallel
            w = Parallel(
                n_jobs=n_jobs, verbose=1)(delayed(
                    image_preproc.convert_bit_img_parallel)(
                    img_proc.image[img_key], img_proc.img_np[img_key],
                    nodata_inp, i_dict, PARAM, img_key_inp=img_key)
                        for i_dict in i_scale)
            hist_plot, meta_lst = map(list, zip(*w))
            # assign the dict variables to the img_proc class
            img_proc.hist_plot = file_utils.merge_dict_lst(hist_plot)
            img_proc.image_meta = file_utils.merge_dict_lst(meta_lst)
        else:
            for i_dict in i_scale:
                img_proc.convert_bit_img(
                    nodata_inp, del_img=True, img_np_key_inp=img_key,
                    **i_dict)

        # save images and proc file
        img_proc.update_meta_df()
        img_proc.save_img_hist_plot_hv(i_name, overlay_plot=True)

    # ---------- do histogram matching with different parameters from
    # parameters are defined in PARAM['HIST_MATCH_DICT']
    if ref_img is not None:
        for i_ref_key, i_ref in ref_img.items():
            for i_name, i_hist in PARAM['HIST_MATCH_DICT'].items():
                img_lst = set([x['img_key_inp'] for x in i_hist])
                [img_proc.read_single_image(
                    PARAM['PATH_EXPORT'],
                    img_proc.image_meta_df.loc[x, 'file'],
                    img_key=x, meta_update=False) for x in img_lst]
                if PARAM['PARALLEL_PROC'] == 1:
                    # get number of parallel jobs
                    n_jobs = min(int(np.ceil(cpu_count()/10)), len(i_scale))
                    # scale images in parallel
                    w = Parallel(
                        n_jobs=n_jobs, verbose=1)(delayed(
                            image_preproc.hist_match_img_parallel)(
                            i_ref, nodata_inp, i_dict,
                            PARAM, file_placeh_out=i_ref_key,
                            image_inp=img_proc.image[i_dict['img_key_inp']]) for i_dict in i_hist)
                    hist_plot, meta_lst = map(list, zip(*w))
                    # assign the dict variables to the img_proc class
                    img_proc.hist_plot = file_utils.merge_dict_lst(hist_plot)
                    img_proc.image_meta = file_utils.merge_dict_lst(meta_lst)
                else:
                    n_plots = len(i_hist)
                    fig, axes_hist = plt.subplots(
                        nrows=n_plots, ncols=3, figsize=(9, n_plots*3))
                    axes_hist = np.atleast_2d(axes_hist)
                    for e_hist, i_dict in enumerate(i_hist):
                        img_proc.hist_match_img(
                            i_ref, nodata_inp, del_img=True,
                            file_placeh_out=i_ref_key,
                            ax_hist_match_plt=axes_hist[[e_hist], :],
                            **i_dict)
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(
                            PARAM['PATH_EXPORT'],
                            f"{PARAM['FILE_PREFIX']}_hist_match_{i_name}.pdf"),
                            format='pdf')

                # save images and proc file
                img_proc.update_meta_df()
                img_proc.save_img_hist_plot_hv(
                    i_name, overlay_plot=False, col_num=1)

    # --------- save metadata ----------
    # update metadata
    img_proc.update_save_metadata('img', proc_nr_str='P01')

    logging.info(
        '\n---- Finalised image preprocessing (scaling) '
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        + '\n\n')

    # ---------- save time measure ----------
    time_file = f"A_{PARAM['FILE_PREFIX']}_{PARAM['LOG_FILE_SUFFIX']}"
    monitoring_utils.save_time_stats(prof, PARAM['PATH_EXPORT'], time_file)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process single images')
    parser.add_argument(
        'PATH_PROC', type=str,
        help=('prefix to processing folder (required, e.g. ".docs/1_site_preproc/BLyaE_v1")'))
    parser.add_argument(
        'PROJ_PARAM_FILE', type=str,
        help=('name of project/site specific parameter file (required, e.g.: PROJ_PARAM_BLyaE_HEX1979_v01)'))
    parser.add_argument(
        '--PARAM_FILE', type=str,
        help='name of task specific param file (optional, default: PARAM01_img_preproc)',
        default='PARAM01_img_preproc')
    parser.add_argument(
        '--PARALLEL_PROC', type=int,
        help='if want to use parallel processing (optional, default: 0)',
        default=0)

    args = parser.parse_args()


    main(vars(args))
