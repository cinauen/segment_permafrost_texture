"""
Functions to retrieve statistics from numpy arrays (e.g. for plotting
ranges) as well as functions to interpolate data to a regular grid
(used within image rotation to achieve regular grid at required
resolution).

"""

import numpy as np
from scipy.interpolate import griddata as sci_griddata



def check_nan_arrray_equal(array1, array2):
    return np.array_equal(array1, array2, equal_nan=True)


def find_limits(np_array, low_perc=10, high_perc=90,
                perc=True, equal=False, mid_val=0):
    """
    Find the minimum and maximum limits of a numpy array based on
    specified percentiles or bounds.
    This function is used e.g. to get plotting bounds

    Parameters
    ----------
    np_array : numpy.ndarray
        The input numpy array.
    low_perc : float, optional
        The lower percentile to use for calculating the minimum limit.
        Default is 10.
    high_perc : float, optional
        The upper percentile to use for calculating the maximum limit.
        Default is 90.
    perc : bool, optional
        If True, calculate the limits based on the specified percentiles.
        If False, use the full range of the array. Default is True.
    equal : bool, optional
        If True, make the limits symmetric around `mid_val`.
        If False, use the calculated limits. Default is False.
    mid_val : float, optional
        The middle value to use for symmetric limits.
        Only used if `equal` is True. Default is 0.

    Returns
    -------
    lim_min : float
        The calculated minimum limit.
    lim_max : float
        The calculated maximum limit.

    Notes
    -----
    - If `perc` is True, the limits are calculated as the percentiles
        of the array.
    - If `perc` is False, the limits are calculated as the minimum and
        maximum values of the array, and a small buffer is added to both
        sides.
    - If `equal` is True, the limits are adjusted to be symmetric
        around `mid_val`.
    """

    lim_min = 0
    lim_max = 100

    if perc:
        lim_min = np.nanpercentile(np_array, low_perc)
        lim_max = np.nanpercentile(np_array, high_perc)

    if not perc or lim_min == lim_max:
        v_min = np.nanmin(np_array)
        v_max = np.nanmax(np_array)
        diff = abs(v_max - v_min)

        add_scale = diff*0.05
        lim_min = v_min - add_scale
        lim_max = v_max + add_scale

    if equal:
        ampl = np.nanmax([abs(lim_max - mid_val), abs(lim_min - mid_val)])
        lim_min = mid_val - ampl
        lim_max = mid_val + ampl

    return lim_min, lim_max


def get_regular_grid_coords(
        x=None, y=None, dx=None, dy=None, nx=None, ny=None,
        grid_corners=None, cell_centers=True, corner=False):

    '''
    create x and y coordinates for regular grid

    Parameters
    ----------
    x : numpy array
        The x coordinates of the data points (e.g. from gdf.x.values).
    y : numpy array
        The y coordinates of the data points (e.g. from gdf.y.values).
    dx : float, optional
        The spacing between points in the x direction for the regular grid.
        If not provided, this will be calculated based on the grid corners
        and the cell number.
    dy : float, optional
        The spacing between points in the y direction for the regular grid.
        If not provided, this will be calculated based on the grid corners
        and the cell number.
    nx : int, REQIRED
        The number of cells in x direction for the regular grid.
    ny : int, REQIRED
        The number of cells in the y direction for the regular grid.
    grid_corners : list, optional
        List defiing corners of output grid with [minx, maxx, miny, maxy].
        If not provided (None), grid edges are calculated from input x and y
    cell_centers : bool, optional, default True
        If True, the regular grid will be defined such that the grid parameters
        (dx, dy, grid_corners) correspond to the cell centers.
        If False, they will correspond to the cell corners.
    corner : bool, optional, default False
        If the ouput gird coordinates should correspond to the corners
        or the cell centers. Default is False (cell centers)

    Note:
    - to get the corners need to set corner = True
        (in any case if cell_center is true or false)

    '''

    if grid_corners is None:
        x_start = np.nanmin(x)
        x_end = np.nanmax(x)
        y_start= np.nanmin(y)
        y_end =np.nanmax(y)
    else:
        x_start = grid_corners[0]
        x_end = grid_corners[1]
        y_start= grid_corners[2]
        y_end = grid_corners[3]

    if cell_centers:
        if dx is None:
            dx = (x_end-x_start)/(nx-1)
            #dx = (x_end-x_start)/(nx)
        if dy is None:
            dy = (y_end-y_start)/(ny-1)
            #dy = (y_end-y_start)/(ny)
    else:
        if dx is None:
            dx = (x_end-x_start)/(nx)
        if dy is None:
            dy = (y_end-y_start)/(ny)

    if cell_centers:
        if not corner:
            # interp_grid_x, interp_grid_y should correspond to cell centers
            interp_grid_x, interp_grid_y = np.mgrid[x_start : x_end+(dx*0.25) : dx,
                                                    y_start : y_end+(dy*0.25) : dy]
        else:
            # interp_grid_x, interp_grid_y should correspond to cell corners
            interp_grid_x, interp_grid_y = np.mgrid[x_start-dx/2 : x_end+(dx*0.75) : dx,
                                                    y_start-dy/2 : y_end+(dy*0.75) : dy]
    else:
        if not corner:
            # interp_grid_x, interp_grid_y should correspond to cell centers
            interp_grid_x, interp_grid_y = np.mgrid[x_start+dx/2 : x_end-(dx*0.25) : dx,
                                                    y_start+dx/2 : y_end-(dy*0.25) : dy]
        else:
            # interp_grid_x, interp_grid_y should correspond to cell corners
            interp_grid_x, interp_grid_y = np.mgrid[x_start : x_end+(dx*0.25) : dx,
                                                    y_start : y_end+(dy*0.25) : dy]

    return interp_grid_x.T, np.flipud(interp_grid_y.T), dx, dy


def interp2D_regular_grid_test(x, y, data, dx=None, dy=None, nx=None, ny=None,
                          method='linear', grid_corners=None, cell_centers=True,
                          rescale=True, interp_grid_x=None, interp_grid_y=None):

    """
    Interpolate regular or irregularly spaced data points to a
    regular grid

    Grid coordinates are derived by get_regular_grid_coords(),
    if they are not provided by interp_grid_x and interp_grid_z


    Parameters
    ----------
    x : numpy array
        x coordinates of the data points (e.g. from gdf.x.values).
    y : numpy array
        y coordinates of the data points (e.g. from gdf.y.values).
    data : numpy array
        Data values corresponding to the coordinates (x, y)
        (e.g. from gdf[col_value_name].values)
    dx : float, optional (only required if grid coords are not provided)
        Spacing between points in the x direction for the regular grid.
        If not provided, this will be calculated based on the grid corrners
        and the cell number.
    dy : float, optional (only required if grid coords are not provided)
        Spacing between points in the y direction for the regular grid.
        If not provided, this will be calculated based on the grid corrners
        and the cell number.
    nx : int, optional (only required if grid coords are not provided)
        Number of cells in the x direction for the regular grid.
        This is required, if no predefined x_coods are provided with
        interp_grid_x.
    ny : int, optional (only required if grid coords are not provided)
        Number of cells in the y direction for the regular grid.
        This is required, if no predefined y_coods are provided with
        interp_grid_y.
    method : str, optional
        Interpolation method to use ('linear', 'nearest', 'cubic').
        Default is 'linear'.
    grid_corners : list, optional (only required if grid coords are not provided)
        List defining corners of output grid with [minx, maxx, miny, maxy].
        If not provided (None) and grid not defined then the outer
        grid edges are calculated from input x and y
    cell_centers : bool, optional, (only required if grid coords are not provided)
        If True, the regular grid will be defined such that the grid parameters
        (dx, dy, grid_corners) correspond to the cell centers.
        If False, they will correspond to the cell corners.
    rescale : bool, optional
        If True, the data will be rescaled to the range of the new grid.
        Default is True.
    interp_grid_x : numpy array, optional
        If provided, these arrays will be used as the x coordinates of
        the regular grid. Otherwise, they will be calculated based on
        the input parameters with get_regular_grid_coords().
    interp_grid_y : numpy array, optional
        If provided, these arrays will be used as the y coordinates of
        the regular grid. Otherwise, they will be calculated based on
        the input parameters with  get_regular_grid_coords().

    Returns
    -------
    data_grid : array-like
        The interpolated data on the regular grid.
    interp_grid_x : array-like
        The x coordinates of the regular grid used for interpolation.
    interp_grid_y : array-like
        The y coordinates of the regular grid used for interpolation.
    dx : float
        The spacing between points in the x direction for the regular grid.
    dy : float
        The spacing between points in the y direction for the regular grid.
    """

    # get grid coordinates
    if interp_grid_x is None or interp_grid_y is None:
        interp_grid_x, interp_grid_y, dx, dy = get_regular_grid_coords(
                          x, y, dx, dy, nx, ny,
                          grid_corners, cell_centers,
                          rescale, interp_grid_x, interp_grid_y)

    xy_orig = np.array(list(zip(x, y)))

    # interpolate to specifies grid
    data_grid = sci_griddata(xy_orig, data, (interp_grid_x, interp_grid_y),
                             method=method, rescale=rescale)

    if interp_grid_x is None or interp_grid_y is None:
        return data_grid, interp_grid_x, interp_grid_y, dx, dy

    return data_grid


def calc_percentile_or_percentage(all_values, perc, fact=0.1):
    '''of perc[0] < 0 or perc[1] > 100 calculates percentage'''
    all_values = np.squeeze(np.array(all_values))
    #all_values = all_values[np.where(~np.isnan(all_values))[0]]
    all_max_val = np.nanmax(all_values)
    all_min_val = np.nanmin(all_values)
    diff = abs(all_max_val - all_min_val)

    if perc[0] < 0:
        min_val = all_min_val + perc[0]/100*diff
    else:
        min_val1 = np.nanpercentile(all_values, perc[0])

    if perc[1] > 100:
        max_val = all_max_val + (perc[1]-100)/100*diff
    else:
        max_val1 = np.nanpercentile(all_values, perc[1])

    if perc[0] >= 0:
        diff = abs(max_val1 - min_val1)
        min_val = min_val1 - fact*diff
    if perc[1] <= 100:
        diff = abs(max_val1 - min_val1)
        max_val = max_val1 + fact*diff

    return [min_val, max_val]


