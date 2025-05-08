'''
Functions to transform and rotate coordinates.
'''

import numpy as np
import cmath

import osgeo
from osgeo import osr
from osgeo import ogr



def coord_transformation(coord_array, in_syst=4326, out_syst=3857):
    '''
    Convert coordinate to new coordinate system
    in_syst: input coordinate system
    out_syst: output coordinate system
        use EPSG numbers:
            WGS 84: 4326
            WGS 84 / Pseudo-Mercator -- Spherical Mercator: 3857

    '''
    long, lat = coord_array[:, 0], coord_array[:, 1]
    # input SpatialReference
    inSpatialRef = define_spatial_ref()
    inSpatialRef.ImportFromEPSG(int(in_syst))

    # output SpatialReference
    outSpatialRef = define_spatial_ref()
    outSpatialRef.ImportFromEPSG(int(out_syst))

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    lat = np.atleast_2d(lat)
    long = np.atleast_2d(long)
    grid_shape = np.shape(lat)
    os_N = np.zeros(grid_shape)
    os_E = np.zeros(grid_shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):

            point = ogr.CreateGeometryFromWkt(
                'POINT (' + str(long[i, j]) + ' ' \
                + str(lat[i, j]) + ')')
            point.Transform(coordTrans)
            dump = point.ExportToWkt()[point.ExportToWkt().find('(')+1:point.ExportToWkt().find(')')].split(' ')
            os_E[i, j] = float(dump[0])
            os_N[i, j] = float(dump[1])

    return np.squeeze(os_E), np.squeeze(os_N)


def define_spatial_ref():
    '''
    Define spatial reference. This is required because x & y axis order
    was changed in GDAL version 3 (see:
    https://github.com/OSGeo/gdal/issues/1546)
    '''
    # input SpatialReference
    inSpatialRef = osr.SpatialReference() # Establish its coordinate encoding
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        inSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return inSpatialRef


def rotate_coord_list(inp, rot_rad, min_vals):
    '''rot rad is in in redians '''
    # rotatw
    #min_vals = np.nanmin(inp, axis=0)
    inp = inp - min_vals
    out = rotate(inp, rot_rad)

    out = out + min_vals
    return out


def rotate(inp, rot_rad, r1=0, r2=1, r3=2):#shift
    '''rotate

    !!! xy rot: r1: x, r2: y, r3: z --> 0, 1, 2
        xz rot: r1: x, r2: z, r3: y --> 0, 2, 1'''
    #rot_rad = degree_to_rad(rot_deg)

    coord_old = inp.copy()

    if ~np.isnan(rot_rad):
        outp = np.zeros((np.shape(inp)[0], np.shape(inp)[1]), dtype = float)*np.nan
        #calculate rotation
        surf = coord_old[:,r1].copy() + (coord_old[:,r2].copy())*1j

        surfr = surf*cmath.exp(-1j*(rot_rad))
        outp[:, r1] = surfr.real
        outp[:, r2] = surfr.imag
        outp[:, r3] = coord_old[:, r3]

        # self.coord_rot = np.transpose(np.vstack((x, y, coord_shift[:, 2].copy())))
    else:
        outp = coord_old.copy()

    return outp


def degree_to_rad(inp):
    out = inp*np.pi/180
    return out


def rad_to_degree(inp):
    out = inp*180/np.pi

    return out