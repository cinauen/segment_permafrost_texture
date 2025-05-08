"""
Functions for process monitoring:
 - memory profiling
 - time logging
 - error logging
"""

import os
import sys
import logging
import datetime as dt
import memory_profiler

import cProfile
import pstats


def get_GPU_memory_info(GPU_lst):
    '''
    Source
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    '''
    import pynvml
    for i in GPU_lst:
        print('=== nvidia memory check ==== '
              + dt.datetime.now().strftime('%Y-%m-%d_%H%M'))
        print('=== GPU num ' + str(i) + '===')
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')

    return


def setup_time_control():
    """
    Set up time control for profiling.

    Returns:
        cProfile.Profile: A Profiler object that can be used to measure
        execution time.
    """
    prof = cProfile.Profile()
    prof.disable()  # disable profiling if don't want to record time...
    prof.enable()  # profiling back on
    return prof


def save_time_stats(prof, PATH_OUT, FILE_PREFIX):
    """
    Save time statistics for the given profiler.

    Args:
        prof (cProfile.Profile): The profiler object that was initialized at the beginning of the script.
        PATH_OUT (str): The output directory where the statistics will be saved.
        FILE_PREFIX (str): The prefix for the output files.

    Returns:
        None
    """
    # save time measure
    path_stats = os.path.normpath(
        os.path.join(PATH_OUT, f'{FILE_PREFIX}_time_stats.stats'))
    path_stats_txt = os.path.normpath(
        os.path.join(PATH_OUT, f'{FILE_PREFIX}_time_stats.txt'))

    prof.disable()  # don't profile the generation of stats
    prof.dump_stats(path_stats)

    with open(path_stats_txt, 'wt') as output:
        stats = pstats.Stats(path_stats, stream=output)
        stats.sort_stats('cumulative', 'time')
        stats.print_stats()

    return


def conditional_profile(func):
    """
    Custom decorator to apply @profile from memory_profiler conditionally:
    - Only applies profiling when the script is run directly (`__main__`).
    - Skips profiling if debugging mode is detected.
    """
    if sys.gettrace() is not None:  # __name__ != "__main__" or
        # Return the original function without memory profiling
        print('profiler is NOT running')
        return func

    # Apply the memory_profiler's @profile decorator
    print('profiler is running')
    return memory_profiler.profile(func)


def init_logging_no_size_limit(log_file=None, append=True,
                 console_loglevel=logging.DEBUG,
                 log_level=logging.INFO,
                 debug=True):
    """Set up logging to file and console.

    If want write cusomt messages into logging file then use:
        logging.info('test')

    levels are:
        CRITICAL
        ERROR
        WARNING
        INFO
        DEBUG
        NOTSET
    """
    if log_file is not None:
        if append:
            filemode_val = 'a'
        else:
            filemode_val = 'w'
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
            filename=log_file,
            filemode=filemode_val)

    # following is required if run with nohup to log
    # console output errors to file
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    #handler = logging.FileHandler(log_file, mode='a')
    console.setLevel(console_loglevel)
    ## set a format which is simpler for console use
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    ## add the handler to the root logger
    logging.getLogger('').addHandler(console)
    #global LOG
    #LOG = logging.getLogger(__name__)#

    logging.info(
        '\n\n==== Proc start ====:'
        + dt.datetime.now().strftime('%Y-%m-%d_%H:%M'))

    return


def init_logging(
        log_file=None, append=True,
        console_loglevel=logging.DEBUG,
        log_level=logging.INFO,
        logging_step_str=''):
    """Set up logging to file and console.

    If want write cusomt messages into logging file then use:
        logging.info('test')

    levels are:
        CRITICAL
        ERROR
        WARNING
        INFO
        DEBUG
        NOTSET
    """
    if append:
        filemode_val = 'a'
    else:
        filemode_val = 'w'

    from logging.handlers import RotatingFileHandler
    # Create a logger
    logger = logging.getLogger('')
    logger.setLevel(log_level)
    handler = RotatingFileHandler(
        log_file,  # file name
        maxBytes=5*1024*1024, # Limit file size to 5 MB (5 * 1024 * 1024 bytes)
        backupCount=2,  # Keep 2 old log files (app.log.1, app.log.2, etc.))
        mode=filemode_val  # with rollover another mode than append doesn't really make sense
        )

    # Set a log format
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(console_loglevel)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)

    logger.addHandler(console)

    logging.info('\n\n ====  Proc start {} ====\n Start time: {}'.format(
        logging_step_str, dt.datetime.now().strftime('%Y-%m-%d_%H:%M')))

    return



