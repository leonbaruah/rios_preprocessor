'''
# ------------------------------------------------------------------------------
# rios_preprocessor.py
# 
# Coded by Leon Baruah (University of Leeds / Landmark Information Group)
# based on Stacie Wolny's (Natural Capital Project) RIOS_Pre_Processing.py
#
# Performs the calculations necessary for producing input to the RIOS tool
#
# ------------------------------------------------------------------------------
'''
# Import system modules
from __future__ import print_function
# noinspection PyPep8
import os, time, re
import shutil
from os.path import basename
from os.path import splitext
from collections import OrderedDict
import numpy as np
import pandas
import geopandas as gpd
import fiona
import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon
import pygeoprocessing as pygeo
import pygeoprocessing.routing as pygrout
import itertools

import logging
logger = logging.getLogger(__name__)


################################################################################
def get_objectives_list():
    '''
    Returns list of tuples: (1) RIOS objectives, 
                            (2) their associated script flag, 
                            (3) parameter prefixes, and 
                            (4) short parameter prefixes
    '''
    return [tuple(['Erosion Control', 'do_erosion', 'erosion', 'er']),
            tuple(['Phosphorus Retention', 'do_nutrient_p', 'phosphorus', 'p']),
            tuple(['Nitrogen Retention', 'do_nutrient_n', 'nitrogen', 'n']),
            tuple(['Flood Mitigation', 'do_flood', 'flood', 'fl']),
            tuple(['Groundwater Recharge/Baseflow', 'do_gw_bf', 'gwater', 'gw'])
            ]


################################################################################
def get_input_data():
    '''
    Returns list of tuples: (1) dataset name with 
                            (2) corresponding initial of objective(s), 
                            (3) variable name in the main script, and 
                            (4) its expected type
    '''
    return [tuple(["Land use/land cover", "EPNFG",
                   "lulc_raster_uri", "raster"]),
            tuple(["RIOS biophysical coefficient table", "EPNFG",
                   "rios_coeff_table", "csv file"]),
            tuple(["DEM", "EPNFG",
                   "dem_raster_uri", "raster"]),
            tuple(["Rainfall erosivity", "EP",
                   "erosivity_raster_uri", "raster"]),
            tuple(["Erodibility", "EP",
                   "erodibility_raster_uri", "raster"]),
            tuple(["Soil depth", "EPNG",
                   "soil_depth_raster_uri", "raster"]),
            tuple(["Precipitation for wettest month", "F",
                   "precip_month_raster_uri", "raster"]),
            tuple(["Soil texture", "FG",
                   "soil_texture_raster_uri", "raster"]),
            tuple(["Annual average precipitation", "G",
                   "precip_annual_raster_uri", "raster"]),
            tuple(["Actual Evapotranspiration", "G",
                   "aet_raster_uri", "raster"])]


################################################################################
def get_intermediate_objective_suffix(suffix=""):
    '''
    Returns dictionary of dictionaries of suffixes to intermediate files.
    Dict contains:
        objective - initial(s) of corresponding objective(s)
        filesuffix - intermediate file/variable suffix
    
    Args (optional) :
        suffix - appends the user defined suffix to the one used by the 
                 intermediate file
    '''
    intsuffix = [tuple(["comb_weight_R", "EPN", "cwgt_r"]),
                 tuple(["comb_weight_Exp", "EPN", "cwgt_e"]),
                 tuple(["index_exp", "EPN", "ind_e"]),
                 tuple(["index_ret", "EPN", "ind_r"]),
                 tuple(["ups_flowacc", "EPNFG", "flowacc"]),
                 tuple(["dret_flowlen", "EPNFG", "flowlen"]),
                 tuple(["index_cover", "FG", "ind_c"]),
                 tuple(["index_rough", "FG", "ind_r"]),
                 tuple(["comb_weight_ret", "FG", "cwgt_r"]),
                 tuple(["comb_weight_source", "FG", "cwgt_s"]),
                 tuple(["rainfall_depth_index", "F", "rain_idx"]),
                 tuple(["precip_annual_index", "G", "prec_idx"]),
                 tuple(["aet_index", "G", "aet_idx"])]

    inter = dict()
    for sfx in intsuffix:
        filesuffix = '_'.join([sfx[2], suffix]).rstrip('_') + '.tif'
        inter[sfx[0]] = {'objective': sfx[1],
                         'filesuffix': filesuffix}
    return inter


################################################################################
def get_output_objective_suffix(suffix=""):
    '''
    Returns dictionary of dictionarys of suffixes to output files produced.
    Dict contains:
        objective - initial(s) of corresponding objective(s)
        filesuffix - output file suffix
    
    Args (optional) :
        suffix - appends the user defined suffix to one used by the output file

    '''
    outsuffix = [tuple(["dret_index", "EPNFG", "downslope_retention_index"]),
                 tuple(["upslope_source", "EPNFG", "upslope_source"]),
                 tuple(["riparian_index", "EPNFG", "riparian_index"]),
                 tuple(["slope_index", "FG", "slope_index"]), ]

    outer = dict()
    for sfx in outsuffix:
        filesuffix = '_'.join([sfx[2], suffix]).rstrip('_') + '.tif'
        outer[sfx[0]] = {'objective': sfx[1],
                         'filesuffix': filesuffix}
    return outer


################################################################################
def get_input_data_to_objective():
    '''
    Returns dictionary of all RIOS inputs in plain english and a string of the
    first letter(s) of each corresponding objective i.e., any combination of
    [E, P, N, F, G].
    '''
    full_data_in = get_input_data()
    return dict([tuple([dat[0], dat[1]]) for dat in full_data_in])


################################################################################
def get_input_data_param_dictionary():
    '''
    Returns dictionary of input names for use within script. Dict indexed by 
    plain english description of dataset. 
    Dict contains:
        found - boolean whether parameter has been input
        param - script name for dataset 
        type - file format e.g. csv, raster
    '''
    data_in = get_input_data()
    return OrderedDict([tuple([dat[0], {'found': False,
                                        'param': dat[2],
                                        'type': dat[3]}]) for dat in data_in])


################################################################################
def get_objective_todo(do_erosion=False, do_nutrient_p=False,
                       do_nutrient_n=False, do_flood=False,
                       do_gw_bf=False):
    '''
    Returns dictionary of all RIOS objectives to do, keyed by plain english 
    description.
    Dict contains:
        found - boolean whether objective is to be processed
        longprefix - identifiable contracted prefix for objective output files
        shortprefix - 1 or 2 letter code for objective intermediate files

    Args (optional) :
        do_erosion      - boolean for Erosion Control objective 
        do_nutrient_p   - boolean for Phosphorus objective
        do_nutrient_n   - boolean for Nitrogen objective
        do_flood        - boolean for Flood Control objective
        do_gw_bf        - boolean for Groundwater/Baseflow objective

    '''
    objective_in = get_objectives_list()
    objective_out = []
    for obj in objective_in:
        objective_out.append(tuple([obj[0], {'found': locals()[obj[1]],
                                             'longprefix': obj[2],
                                             'shrtprefix': obj[3]}]))
    return OrderedDict(objective_out)


################################################################################
def get_objective_dictionary(suffix='', do_erosion=False, do_nutrient_p=False,
                             do_nutrient_n=False, do_flood=False,
                             do_gw_bf=False):
    '''
    Returns dictionary, keyed by objective, with status of whether objective 
    used, list of data sets required for objective and various objective
    specific inputs and outputs.
    Dict contains:
        found -         boolean for objective (as defined by input)
        dataset -       list of plain english descriptions of input data
        intermediate -  dictionary of intermediate filenames indexed as 
                        get_intermediate_objective_suffix()
        output -        dictionary of output filenames, indexed as 
                        get_output_objective_suffix()
        longprefix -    identifiable prefix for objective output files
        shortprefix -   1 or 2 letter code for objective intermediate files

    Args (optional):
        do_erosion      - boolean for Erosion Control objective 
        do_nutrient_p   - boolean for Phosphorus objective
        do_nutrient_n   - boolean for Nitrogen objective
        do_flood        - boolean for Flood Control objective
        do_gw_bf        - boolean for Groundwater/Baseflow objective

    '''
    objective_dict = OrderedDict()
    datainput = get_input_data_to_objective()
    objectiveinput = get_objective_todo(do_erosion=do_erosion,
                                        do_nutrient_p=do_nutrient_p,
                                        do_nutrient_n=do_nutrient_n,
                                        do_flood=do_flood,
                                        do_gw_bf=do_gw_bf)
    objectiveinter = get_intermediate_objective_suffix(suffix=suffix)
    objectiveoutput = get_output_objective_suffix(suffix=suffix)
    for objective in objectiveinput.keys():

        data_in_this_objective = []
        intermed_in_this_objective = {}
        output_for_this_objective = {}
        # use initial letter of objective to determine which
        # data/intermediate/output files used.
        for data in datainput.keys():
            if objective[0] in datainput[data]:
                data_in_this_objective.append(data)
        for mediate in objectiveinter:
            if objective[0] in objectiveinter[mediate]['objective']:
                intermed_in_this_objective[mediate] = \
                                            '_'.join([objectiveinput[objective]['shrtprefix'], 
                                                      objectiveinter[mediate]['filesuffix']])
        for suffix in objectiveoutput:
            if objective[0] in objectiveoutput[suffix]['objective']:
                output_for_this_objective[suffix] = \
                                            '_'.join([objectiveinput[objective]['longprefix'], 
                                                      objectiveoutput[suffix]['filesuffix']])
        objective_dict[objective] = {'found': objectiveinput[objective]['found'],
                                     'dataset': data_in_this_objective,
                                     'intermediate': intermed_in_this_objective,
                                     'output': output_for_this_objective,
                                     'longprefix': objectiveinput[objective]['longprefix'],
                                     'shrtprefix': objectiveinput[objective]['shrtprefix']}
    return objective_dict


################################################################################
def optimize_threshold_flowacc(flow_acc_raster_uri,
                               river_reference_shape_uri_list,
                               workspace_path='.\\',
                               stream_length_multiplier=1.0,
                               aoi_shape_uri=None, suffix='', seedlen=1000,
                               streams_raster_uri=None, all_touched=True):
    '''
    Calculates the pixel threshold (length in pixel units) at which the flow
    accumulation raster best describes the river in river shapefiles supplied
    (optionally multiplied by some factor).
    Returns this pixel threshold value.

    Args :
        flow_acc_raster_uri             -   path to flow accumulation raster
        river_reference_shape_uri_list  -   list of shapefile(s) containing
                                            geometry of rivers

    Args (optional) :
        workspace_path              - path to where files describing rivers in
                                      local area will be placed
        seedlen                     - starting length (in pixel units) that
                                      flow accumulation must exceed
                                      to define river pixels
        stream_length_multiplier    - since shapefile of river may be polyline
                                      or polygon, multiplier extends/contracts
                                      stream by factos of flow accumulation
                                      threshold
        aoi_shape_uri               - path to shapefile of area of interest
        suffix                      - filename suffix
        seedlen                     - seed for flow accumulation threshold
        streams_raster_uri          - output file for streams line raster
                                      (1 = stream pixel)
        all_touched                 - river shape rasterization flag
    '''

    if hasattr(river_reference_shape_uri_list, 'format'):  # i.e. is a string
        river_reference_shape_uri_list = [river_reference_shape_uri_list]
    reference_name = basename(river_reference_shape_uri_list[0])
    river_local_shape_uri = workspace_path + ('_' + suffix).join(splitext(reference_name))
    river_local_raster_uri = splitext(river_local_shape_uri)[0] + '.tif'
    # also fetch metadata needed to initialize new raster
    with fiona.open(river_reference_shape_uri_list[0], 'r') as river_ref:
        rivercrs = river_ref.crs
        riverdriver = river_ref.driver
        riverschema = river_ref.schema
        # if a local cut-out of the detailed river
    if not os.path.exists(river_local_shape_uri):
        with rasterio.open(flow_acc_raster_uri, 'r') as flow_acc:
            flow_acc_meta = flow_acc.meta
        # make a bounding box
        xdem = [flow_acc_meta['affine'][2], flow_acc_meta['affine'][2] +
                flow_acc_meta['width'] * flow_acc_meta['affine'][0]]
        ydem = [flow_acc_meta['affine'][5], flow_acc_meta['affine'][5] +
                flow_acc_meta['height'] * flow_acc_meta['affine'][4]]
        flow_acc_bounds = Polygon([(xdem[0], ydem[0]), (xdem[1], ydem[0]),
                                   (xdem[1], ydem[1]), (xdem[0], ydem[1]),
                                   (xdem[0], ydem[0])])

        # clip river shape file to area limits
        logger.debug("\tClipping river shape geometries to raster boundaries")
        with fiona.open(river_local_shape_uri, 'w', crs=rivercrs, 
                        driver=riverdriver, schema=riverschema) as river_local:
            for river_reference_shape_uri in river_reference_shape_uri_list:
                with fiona.open(river_reference_shape_uri, 'r') as river_ref:
                    for river in river_ref:
                        if not flow_acc_bounds.intersects(shapely.geometry.shape(river['geometry'])):
                            continue
                        intersect_geom = flow_acc_bounds.intersection(shapely.geometry.shape(river['geometry']))
                        if intersect_geom.type == 'GeometryCollection':
                            for geom in intersect_geom:
                                clip_geom = shapely.geometry.mapping(shapely.geometry.shape(geom))
                                # filter out single points
                                if clip_geom['type'] == riverschema['geometry']:
                                    river_local.write({'properties': river['properties'],
                                                       'geometry': clip_geom})
                        else:
                            clip_geom = shapely.geometry.mapping(
                                shapely.geometry.shape(intersect_geom))
                        # filter out single points
                            if clip_geom['type'] == riverschema['geometry']:
                                river_local.write({'properties': river['properties'],
                                                   'geometry': clip_geom})

    logger.debug("\tRasterizing river shape geometries")
    # N.B. should reprocess each time: all_touched may change
    river_local_df = gpd.read_file(river_local_shape_uri)
    shapes = [tuple([geom, 1.]) for geom in river_local_df.geometry]
    with rasterio.open(flow_acc_raster_uri, 'r') as flow_acc:
        flow_acc_meta = flow_acc.meta
        flow_acc_shape = flow_acc.read(1).shape

    sitemask = features.rasterize(shapes, out_shape=flow_acc_shape,
                                  transform=flow_acc_meta['affine'],
                                  fill=0., all_touched=all_touched)

    with rasterio.open(river_local_raster_uri, 'w', **flow_acc_meta) as cliptif:
        cliptif.write_band(1, sitemask.astype(flow_acc_meta['dtype']))
    with rasterio.open(river_local_raster_uri, 'r') as river_raster:
        river_local_data = river_raster.read(1)
        river_local_meta = river_raster.meta
    # if an area of interst is defined, create boolean mask to select the area
    # we wish to optimize the river network
    # N.B. for this part we always select all_touched to be True
    if aoi_shape_uri is not None:
        logger.debug("\tLimiting analysis to Area of Interest")
        aoi_df = gpd.read_file(aoi_shape_uri)
        shapes = [tuple([geom, 1.]) for geom in aoi_df.geometry]
        aoi_mask = features.rasterize(shapes, out_shape=river_local_data.shape,
                                      transform=river_local_meta['affine'],
                                      fill=0., all_touched=True).astype(bool)
    else:
        aoi_mask = np.ones(flow_acc_shape).astype(bool)
    # count the number of pixels in area of interest
    n_riverpix = len(np.where(river_local_data[aoi_mask] == 1)[0]) * stream_length_multiplier
    with rasterio.open(flow_acc_raster_uri, 'r') as flow_raster:
        flow_acc_data = flow_raster.read(1)
        flow_acc_meta = flow_raster.meta
    n_flowpix = len(np.where(flow_acc_data[aoi_mask] > seedlen)[0])
    factor = 0.5
    nlooplimit = 10000
    n_verify = 10
    nloop = 0
    d_riverflow = abs(n_riverpix - n_flowpix)
    prevseeds = np.zeros(n_verify)
    logger.debug("Optimizing flow accumulation threshold")
    while d_riverflow > 0:
        prevseeds[nloop % n_verify] = seedlen
        if (np.max(prevseeds) == np.min(prevseeds)) & (np.max(prevseeds) != 0):
            break
        nloop += 1
        if (n_riverpix > n_flowpix):  # number of flow pixels needs to increase
            seedlen = (seedlen - (seedlen * factor))  # -> threshold decreases
        elif (n_riverpix < n_flowpix):  # no of flow pixels needs to decrease
            seedlen = (seedlen + (seedlen * factor))  # -> threshold increases
        # recalculate number of flow pixels with new seed length
        n_flowpix = len(np.where(flow_acc_data[aoi_mask] > seedlen)[0])
        # if new solution diverges, reduce the factor by which the seed changes
        if abs(n_riverpix - n_flowpix) > d_riverflow:
            factor *= 0.9
        # calculate the new difference between river & accumulated flow rasters
        d_riverflow = abs(n_riverpix - n_flowpix)
        if nloop > nlooplimit:  # break out of limit if stuck
            break
    logger.debug('%d loops iterated; optimal flow accumulation threshold = %f' %
                 (nloop,seedlen))

    if streams_raster_uri is not None:
        logger.debug("Saving streams raster as " + basename(streams_raster_uri))
        if os.path.sep not in streams_raster_uri:
            streams_raster_uri = workspace_path + streams_raster_uri
        stream_pixels = np.where(flow_acc_data > seedlen)
        stream_data = np.zeros(flow_acc_data.shape,
                               dtype=flow_acc_meta['dtype'])
        stream_data[stream_pixels] = 1.
        with rasterio.open(streams_raster_uri, 'w', **flow_acc_meta) as sraster:
            sraster.write_band(1, stream_data)
    
    return seedlen


###############################################################################
def define_channels(flow_dir_raster_uri, flow_dir_channels_raster_uri,
                    streams_raster_uri, nullvalue=None):
    '''
    Set flow direction raster to some arbitrary value or null it out where there
    are streams.
    
    Args :
        flow_dir_raster_uri          - raster of flow direction
                                       (created by pygeoprocessing)
        streams_raster_uri           - streams line raster (1 = stream pixel)
        flow_dir_channels_raster_uri - output raster of flow accumulation
                                       areas NOT considered to be streams.

    Args (optional):
        nullvalue   - value of null pixels in output raster 

    '''
    # read in raster inputs    
    with rasterio.open(flow_dir_raster_uri, "r") as flow_dir_raster:
        flow_dir_data = flow_dir_raster.read(1)
        flow_dir_meta = flow_dir_raster.meta
    # set up null value
    if nullvalue is None:
        nullvalue = flow_dir_meta["nodata"]
    # search for streams.
    with rasterio.open(streams_raster_uri, "r") as streams_raster:
        streams_data = streams_raster.read(1)
        streams = np.where(streams_data == 1)

    # copy flow direction image and null out stream values
    flowdir_channels_data = flow_dir_data.copy()
    flowdir_channels_data[streams] = nullvalue
    # write to file
    with rasterio.open(flow_dir_channels_raster_uri, "w", **flow_dir_meta) as out:
        out.write_band(1, flowdir_channels_data)


###############################################################################
def map_coefficients(lulc_raster_uri, lucode_field, rios_coeff_table):
    '''
    Map general LULC classes and coefficient table to user's landcover raster.
    Returns a pandas dataframe of land uses (and accompanying coefficients) that
    appear in the raster.

    Args: 
        lulc_raster_uri     - path to land use (LU) raster
        lucode_field        - name of LU field code in RIOS coefficient table
        rios_coeff_table    - path to csv containing biophysical coefficients
    '''
    # get LULC values in the lulc raster 
    with rasterio.open(lulc_raster_uri, 'r') as lulcraster:
      lulcrasterval = list(set(lulcraster.read(1).ravel()))
    # read in map coefficients table
    lulc_coeffs = pandas.read_csv(rios_coeff_table)
    # return only rows where lucode field values match those present in raster 
    return lulc_coeffs[lulc_coeffs[lucode_field].isin(lulcrasterval)]


###############################################################################
def normalize(in_raster_uri, out_raster_uri):
    '''
    Takes in raster file and outputs normalized raster.
    Will pull out null values from input and apply to output, if no null value
    in raster -> -9999

    Args:
        in_raster_uri   - path to raster to be normalized
        out_raster_uri  - path to output normalized raster.

    '''
    with rasterio.open(in_raster_uri, 'r') as rasterin:
        in_data = rasterin.read(1)
        in_meta = rasterin.meta
        if in_meta['nodata'] is None:
            in_meta.update(nodata=-9999)
        if len(in_meta['crs']) == 0:  # if no coordinate reference system -> BNG
            in_meta.update(crs={'init': u'epsg:27700'})
    out_data = normalize_array(in_data, nullvalue=in_meta['nodata'])
    with rasterio.open(out_raster_uri, 'w', **in_meta) as rasterout:
        rasterout.write_band(1, out_data)


###############################################################################
def normalize_array(raster_data, nullvalue=-9999.):
    '''
    Normalize raster, using the max value for that raster.
    Returns map (numpy array) of normalized data
    
    Args :
        raster_data - 2D numpy array of raster data.

    Args (optional) : 
        nullvalue - array value of missing data
    '''

    # get subset of array to normalize (exclude erroneous)
    val_to_norm = np.where(raster_data != nullvalue)
    null_data = np.where(raster_data == nullvalue)
    out_raster = raster_data.copy()
    out_raster[val_to_norm] = raster_data[val_to_norm] / np.max(raster_data)
    out_raster[null_data] = nullvalue
    return out_raster


###############################################################################
def derive_raster_from_lulc(lulc_raster_uri, lucode_field, lulc_coeff_df,
                            coeff_field, output_raster_uri):
    '''
    Replacing the Lookup() function from the spatial analyst, this function
    takes in the lulc raster file, pandas table of rios coefficients and the
    name of the rios coefficient for which a map should be made.
    Returns a raster file with  new coefficient values, but same null values.

    Args :
        lulc_raster_uri     - path to land use raster file
        lucode_field        - name of field containing land cover code
        lulc_coeff_df       - dataframe containing land use code and replacement
                              coefficient
        coeff_field         - name of field containing coefficient values to
                              replace land cover code
        output_raster_uri   - path to output raster with spatial land cover
                              pixels given coefficient values

    '''
    with rasterio.open(lulc_raster_uri, 'r') as lulcraster:
        lulcmeta = lulcraster.meta
        lulcdata = lulcraster.read(1)
        coeffraster = lulcraster.read(1).astype(float)  # so null values same
    for lucode in lulc_coeff_df[lucode_field].values:
        this_ludata = lulc_coeff_df[lulc_coeff_df[lucode_field] == lucode]
        replacement_value = this_ludata[coeff_field].values[0]
        coeffraster[np.where(lulcdata == lucode)] = replacement_value
    with rasterio.open(output_raster_uri, 'w', **lulcmeta) as outputraster:
        outputraster.write_band(1, coeffraster.astype(lulcmeta['dtype']))


################################################################################
def change_geotiff_nodata_value(geotiff, new_nodata=-9999):
    """
    Change the NODATA value in the geotiff, adjusting nodata values in
    metadata and the raster itself.
    """
    with rasterio.open(geotiff, 'r') as raster:
        raster_data = raster.read(1)
        raster_meta = raster.meta

    old_nodata = raster_meta['nodata']
    if old_nodata == new_nodata:
        return None

    nodata_loc = np.where(raster_data == old_nodata)
    raster_data[nodata_loc] = new_nodata
    raster_meta['nodata'] = new_nodata

    # archive the old version of the file
    old_geotiff = "_old_nodata".join(os.path.splitext(geotiff))
    shutil.copy(geotiff, old_geotiff)

    with rasterio.open(geotiff, 'w', **raster_meta) as raster:
        raster.write_band(1, raster_data)


###############################################################################
def weighted_flow_accumulation(flow_direction_uri, dem_uri, flux_output_uri,
                               source_weight_uri=None,
                               absorption_weight_uri=None, aoi_shape_uri=None):
    """
    A helper function to calculate flow accumulation, also returns intermediate
    rasters for future calculation. Borrowed from pygeoprocessing.

    Args :
        flow_direction_uri - a uri to a raster that has d-infinity flow
                             directions in it
        dem_uri            - path to gdal dataset representing a DEM, must be
                             aligned with flow_direction_uri
        flux_output_uri    - location to dump the raster representing flow
                             accumulation

    Args (optional) :
        source_weight_uri     - additive weight of pixel adding to flow
        absorption_weight_uri - subtractive weight of pixel, taking from flow
        aoi_shape_uri         - path to a datasource to mask out the dem
    """
    loss_uri = pygeo.temporary_filename(suffix='.tif')
    delete_uri = [loss_uri]
    if source_weight_uri is None:
        source_weight_uri = pygeo.temporary_filename(suffix='.tif')
        pygeo.make_constant_raster_from_base_uri(dem_uri, 1.0, source_weight_uri)
        delete_uri.append(source_weight_uri)

    if absorption_weight_uri is None:
        absorption_weight_uri = pygeo.temporary_filename(suffix='.tif')
        pygeo.make_constant_raster_from_base_uri(dem_uri, 0.0, absorption_weight_uri)
        delete_uri.append(absorption_weight_uri)

    pygrout.routing.route_flux(flow_direction_uri, dem_uri, source_weight_uri,
                               absorption_weight_uri, loss_uri, flux_output_uri,
                               'flux_only', aoi_uri=aoi_shape_uri)
    change_geotiff_nodata_value(flux_output_uri, new_nodata=-9999)

    for ds_uri in delete_uri:
        try:
            os.remove(ds_uri)
        except:
            logger.warning(("Couldn't delete %s: still open" % ds_uri))


###############################################################################
def average_raster(raster_uri_list=None, inverseraster_uri_list=None,
                   output_raster_uri='MEANRASTER.tif'):
    '''
    Outputs a map of the the mean raster pixel values. Missing pixels are
    excluded from calculation.
    Allows inverse of rasters (1-raster) to contribute to mean

    Args:
        raster_uri_list        - list of rasters to be added together
        inverseraster_uri_list - list of rasters where the values to be inversely added (1-raster)
        output_raster_uri      - name of the raster output file
    '''
    if ((raster_uri_list is None) and (inverseraster_uri_list is None)):
        raise Exception('No rasters supplied to average_raster routine')

    # turn Nonetype values into empty lists ######
    if raster_uri_list is None:
        raster_uri_list = []
    elif hasattr(raster_uri_list, 'format'):
        raster_uri_list = [raster_uri_list]
    if inverseraster_uri_list is None:
        inverseraster_uri_list = []
    elif hasattr(inverseraster_uri_list, 'format'):
        inverseraster_uri_list = [inverseraster_uri_list]
    ##############################################

    # fetch data and metadata (one of the lists should contain something)
    rasterexample = (raster_uri_list + inverseraster_uri_list)[0]
    with rasterio.open(rasterexample, 'r') as raster:
        rasterdata = raster.read(1)
        rastermeta = raster.meta

    # set up calculation rasters
    totaldata = np.zeros(rasterdata.shape)
    divisiondata = np.zeros(rasterdata.shape)

    # add up regular raster data
    for rasteruri in raster_uri_list:
        with rasterio.open(rasteruri, 'r') as raster:
            rasterdata = raster.read(1)
            rastermeta = raster.meta
        good = np.where(rasterdata != rastermeta['nodata'])
        totaldata[good] += rasterdata[good]
        divisiondata[good] += 1

    # add on (1-raster) data.
    for rasteruri in inverseraster_uri_list:
        with rasterio.open(rasteruri, 'r') as raster:
            rasterdata = raster.read(1)
            rastermeta = raster.meta
        good = np.where(rasterdata != rastermeta['nodata'])
        rasterdata = 1. - rasterdata
        totaldata[good] += rasterdata[good]
        divisiondata[good] += 1

    # now calculate mean
    bad = np.where(divisiondata == 0)
    good = np.where(divisiondata > 0)
    meandata = np.zeros(rasterdata.shape, dtype=rastermeta['dtype'])
    meandata[good] = totaldata[good] / divisiondata[good]
    meandata[bad] = rastermeta['nodata']
    # save mean raster
    with rasterio.open(output_raster_uri, 'w', **rastermeta) as outputraster:
        outputraster.write_band(1, meandata)


###############################################################################
def get_neighbouring_pixels(x=0, y=0, xlim=(-1, 1), ylim=(-1, 1), radius=1):
    '''
    Yields coordinates of pixels in square surrounding this one, exclude edges.
    By default, outputs relative pixel coordinate positions around centre (0,0).

    Args (default +/- 1 pixel around <0,0>):
        x       - x-coordinate of pixel
        y       - y-coordinate of pixel
        xlim    - (minimum, maximum) tuple of x-coordinate pixels
        ylim    - (minimum, maximum) tuple of y-coordinate pixels
        radius  - number of pixels above/below (x,y) that are returned
    '''
    intradius = int(np.ceil(radius))
    for dx, dy in (itertools.product((np.arange(-intradius, intradius+1)),
                                     (np.arange(-intradius, intradius+1)))):
        if ((dx == 0) and (dy == 0)):  # exclude the coordinate of pixel itself
            continue
        # if surrounding pixel is within array limits, return it :)
        if (min(xlim) <= (x+dx) <= max(xlim)) & (min(ylim) <= (y+dy) <= max(ylim)):
            yield tuple([x+dx, y+dy])


###############################################################################
def get_pixels_within_radius(x=0, y=0, xlim=(-1, 1), ylim=(-1, 1), radius=1.):
    '''
    Yields coordinates of pixels in circle surrounding this one, excluding edges.
    By default, spits out relative pixel coordinate positions around centre (0,0).
    Distances are measured from centre to pixel.

    Args (default radius of 1 around <0,0>):
        x       - x-coordinate of pixel
        y       - y-coordinate of pixel
        xlim    - (minimum, maximum) tuple of x-coordinate pixels
        ylim    - (minimum, maximum) tuple of y-coordinate pixels
        radius  - distance from (x,y) in pixel units to fetch neighbours.
    '''
    intradius = int(np.ceil(radius))
    for dx, dy in (itertools.product((np.arange(-intradius, intradius+1)),
                                     (np.arange(-intradius, intradius+1)))):
        if ((dx == 0) and (dy == 0)):  # exclude the coordinate of pixel itself
            continue
        if ((dx ** 2 + dy ** 2) > radius ** 2):  # pixel is within radius?
            continue
        # if surrounding pixel is within array limits, return it :)
        if (min(xlim) <= (x+dx) <= max(xlim)) & (min(ylim) <= (y+dy) <= max(ylim)):
            yield tuple([x+dx, y+dy])


###############################################################################
def pixel_neighbours_from_east(x=0, y=0, xlim=(-1, 1), ylim=(-1, 1),
                               clockwise=False):
    '''
    Returns coordinates neighbouring a single pixel in order from Easterly.
    Pixels defined relative to (0,0):
                                        (-1, 1) (0, 1) (1, 1)
                                        (-1, 0) (0, 0) (1, 0)
                                        (-1,-1) (0,-1) (1,-1)

    Args (default returns pixels anti-clockwise around <0,0>):
        x           - x-coordinate of pixel
        y           - y-coordinate of pixel
        xlim        - (minimum, maximum) tuple of x-coordinate pixels
        ylim        - (minimum, maximum) tuple of y-coordinate pixels
        clockwise   - if False: E -> NE -> N -> NW -> etc -> SE
                      if True:  E -> SE -> S -> SW -> etc -> NE
    '''
    radian_circle = np.arange(0., 2. * np.pi, np.pi / 4.)
    coord_circle = []
    for rad in radian_circle:
        coord_circle.append(tuple([np.round(np.sin(rad)), np.round(np.cos(rad))]))

    # if going clockwise from east, rearrange and reorder coordinate list
    if clockwise:
        coord_circle = [coord_circle[0]] + coord_circle[:0:-1]
    # if surrounding pixel is within array limits, return it :)
    for dx, dy in coord_circle:  # N.B. somehow this gets reversed
        if (min(xlim) <= (x+dx) <= max(xlim)) & (min(ylim) <= (y+dy) <= max(ylim)):
            yield tuple([x+dx, y+dy])


###############################################################################
def map_pixels_next_to_river(streams_raster_uri, radius=1.99):
    '''
    Read in a river raster and get coordinates of all pixels that neighbour the
    stream. Default radius covers the 8-directional neighbours in immediate
    vicinity of stream pixels.
    Returns list of (x,y) coordinate pairs.

    Args :
        streams_raster_uri  - path to raster file of stream pixels
                              (expects stream pixels = 1)

    Args (optional) :
        radius  - centre-to-centre pixel buffer to stream
                  N.B. the centres of diagonal neighbours are further than
                       adjacent ones
    '''
    # read in the stream raster
    with rasterio.open(streams_raster_uri) as stream_raster:
        stream_data = stream_raster.read(1)
    # set up raster to keep a record of stream neighbour pixels
    strdim = stream_data.shape
    nbor_stream_data = np.zeros(strdim).astype(stream_data.dtype)
    is_stream = np.where(stream_data == 1)
    logger.debug('\t\t\t%d stream pixels' % len(is_stream[0]))
    for strpix in zip(is_stream[0], is_stream[1]):
        nborpix = get_pixels_within_radius(x=strpix[0], y=strpix[1],
                                           xlim=(0, strdim[0] - 1),
                                           ylim=(0, strdim[1] - 1),
                                           radius=radius)
        for nbor in nborpix:
            if stream_data[nbor] == 1:
                continue
            # if the pixels immediately above, below, left and right are
            # river pixels, then this is not an edge
            # (use radius = 1 to omit diagonals)
            nbornborrad = get_pixels_within_radius(x=nbor[0], y=nbor[1],
                                                   xlim=(0, strdim[0] - 1),
                                                   ylim=(0, strdim[1] - 1),
                                                   radius=1.)
            nstream_nnbor = [stream_data[nnbor] for nnbor in nbornborrad]
            if len(nstream_nnbor) > np.sum(nstream_nnbor):
                nbor_stream_data[nbor] = 1.
    return nbor_stream_data


###############################################################################
def get_end_pixels_of_river_raster(streams_raster_uri):
    '''
    Find the ends (discontinuities) in a raster or rivers/streams. Stream
    terminators are simply the coordinates that only have one other pixel
    adjacent/diagonally-adjacent (or none for isolated stream pixel).
    Returns a list of (x,y) coordinate pairs.

    Args :
        streams_raster_uri  - path to raster file of stream pixels
                              (expects stream pixels = 1)
    '''
    # read in the river/stream data
    with rasterio.open(streams_raster_uri) as stream_raster:
        stream_data = stream_raster.read(1)
    # find the pixel set that are part of the stream
    strdim = stream_data.shape
    is_stream = np.where(stream_data == 1)
    # find the ends (candidate discontinuities) of all the rivers
    river_ends = []
    for strpix in zip(is_stream[0], is_stream[1]):
        nborpix = get_neighbouring_pixels(x=strpix[0], y=strpix[1],
                                          xlim=(0, strdim[0] - 1),
                                          ylim=(0, strdim[1] - 1),
                                          radius=1)
        nborrivercnt = np.sum([stream_data[nbor] for nbor in nborpix])
        if nborrivercnt <= 1:
            river_ends.append(strpix)
    logger.debug('\t\t\t%d river ends found' % len(river_ends))
    return river_ends


###############################################################################
def label_streams(streams_raster_uri):
    '''
    Takes in a stream raster and gives each contiguous segment an arbitrary
    identification number 1,2,3... n.
    Returns a map (numpy array) where the stream pixels of the input raster file
    are given a unique identification no.

    Args :
        streams_raster_uri  - path to raster file of stream pixels
                              (expects stream pixels = 1)
    '''
    # identify the riverends that need looking at
    end_to_check = get_end_pixels_of_river_raster(streams_raster_uri)
    # read in the stream raster
    with rasterio.open(streams_raster_uri) as stream_raster:
        stream_data = stream_raster.read(1)
    # make index in raster plane
    strdim = stream_data.shape
    stream_id = np.zeros(stream_data.shape).astype(int)
    stridx = 1
    for end in end_to_check:
        # check whether the end has already been assigned
        if stream_id[end] > 0:
            continue
        # assign a stream id
        stream_id[end] = stridx
        # find all neighbouring pixels and mark them as ID'd
        # start with putting all the neighbours of the end pixel into a buffer
        nborbuffer = list(get_neighbouring_pixels(x=end[0], y=end[1],
                                                  xlim=(0, strdim[0] - 1),
                                                  ylim=(0, strdim[1] - 1),
                                                  radius=1))
        while len(nborbuffer) > 0:
            nbor = nborbuffer.pop()  # pop the last coordinate off the buffer
            # assign the neighbour pixel an ID if it
            #   a) is a stream pixel and
            #   b) has not already been assigned an ID
            if (stream_data[nbor] != 0) and (stream_id[nbor] == 0):
                stream_id[nbor] = stridx
                # put neighbours of the neighbour onto the neighbour-buffer
                nbor_of_nbor = list(get_neighbouring_pixels(x=nbor[0], y=nbor[1],
                                                            xlim=(0, strdim[0] - 1),
                                                            ylim=(0, strdim[1] - 1),
                                                            radius=1))
                for nnbor in nbor_of_nbor:
                    if (stream_data[nnbor] != 0) and (stream_id[nnbor] == 0):
                        nborbuffer.append(nnbor)
        stridx += 1  # when the buffer is empty, increment the ID

    logger.debug('\t\t\t%d unique IDs given to streams' % stridx)
    return stream_id


###############################################################################
def label_river_banks(streams_raster_uri):
    '''
    Identifies river banks with each bank given a unique ID. Banks are defined
    as those pixels that lie next to a  river, but do not adjunct the river end.
    Returns a map (numpy array) of river bank IDs.

    Args :
        streams_raster_uri  - path to raster file of stream pixels
                             (expects stream pixels = 1)

    '''
    # Get all the pixels next to the river(s)

    logger.info('\t\tIdentifying map pixels next to river')
    stream_border = map_pixels_next_to_river(streams_raster_uri)
    # get all the river(s) ends
    logger.info('\t\tIdentifying river terminal pixels')
    stream_end = get_end_pixels_of_river_raster(streams_raster_uri)
    # label individual streams
    logger.info('\t\tAssigning IDs to streams')
    stream_id = label_streams(streams_raster_uri)
    strdim = stream_id.shape
    # set up output and intermediary maps
    bank_map = np.zeros(strdim).astype(stream_id.dtype)
    river_bank = []
    bank_idx = 1
    # cwise = False
    max_search_attempts = 10
    for strid in list(set(stream_id[np.where(stream_id != 0)])):

        logger.debug("Starting stream %d analysis" % strid)
        # identify this stream
        this_stream = np.where(stream_id == strid)
        this_stream_coord = zip(this_stream[0], this_stream[1])
        if len(this_stream_coord) < 3:
            continue
        # figure out which junctions/ends are part of this stream
        this_stream_end = [end for end in stream_end
                           if end in this_stream_coord]
        # a river segment have two banks, so follow ends twice
        for end in this_stream_end + this_stream_end:
            # initialize this bank pixel record
            this_bank = []
            this_queue = []
            # prime neighbour pixels, but don't add to buffer yet
            end_queue = [end]
            nend = end_queue.pop()
            stream_travelled = [nend]
            nbor_of_end = list(pixel_neighbours_from_east(x=nend[0], y=nend[1],
                                                          xlim=(0, strdim[0] - 1),
                                                          ylim=(0, strdim[1] - 1)))
            found_border = False
            search_attempts = 0

            # while no border pixels amongst neighbours =============
            # if the river pixel happens to be at the edge of the
            # raster, this loop will travel down river until it
            # finds a border or reaches another end of the river
            while not found_border:
                search_attempts += 1
                if search_attempts >= max_search_attempts:
                    break
                # check for border pixels
                for nbor in nbor_of_end:
                    if (stream_border[nbor] == 0) or (bank_map[nbor] != 0):
                        continue
                    # if border pixel is ONLY next to the terminus, ignore
                    nbor_nbor_of_end = list(pixel_neighbours_from_east(x=nbor[0], y=nbor[1],
                                                                       xlim=(0, strdim[0] - 1),
                                                                       ylim=(0, strdim[1] - 1)))
                    nbor_is_stream = [pix for pix in nbor_nbor_of_end
                                      if pix in this_stream_coord]
                    if len(nbor_is_stream) > 1:
                        found_border = True
                # if no unassigned border pixels found
                if not found_border:
                    # should be another river pixel next to it;
                    # define it as the new end
                    nend_candidates = [pix for pix in nbor_of_end
                                       if ((pix not in stream_travelled) and
                                           (pix in this_stream_coord))]
                    # add every potential pixel candidate (because river width)
                    for nend_cand in nend_candidates:
                        end_queue.append(nend_cand)
                    if len(end_queue) == 0:
                        break
                    nend = end_queue.pop()
                    stream_travelled.append(nend)
                    nbor_of_end = list(pixel_neighbours_from_east(x=nend[0], y=nend[1],
                                                                  xlim=(0, strdim[0] - 1),
                                                                  ylim=(0, strdim[1] - 1)))

                # if the stream traversed has reached an end pixel that is NOT
                # the starting end pixel
                if ((nend in this_stream_end) and (nend != end)):
                    break

            # unsuccessful search ===================================
            if not found_border:
                continue

            # queue border's first pixel ============================
            # find border pixels in that set that neighbour more than
            # one river pixel
            for nbor in nbor_of_end:
                # check this is an unassigned border pixel
                if (stream_border[nbor] == 0) or (bank_map[nbor] != 0):
                    continue
                # identify stream pixels around it
                nbor_nbor_of_end = list(pixel_neighbours_from_east(x=nbor[0], y=nbor[1],
                                                                   xlim=(0, strdim[0] - 1),
                                                                   ylim=(0, strdim[1] - 1)))
                nbor_is_stream = [pix for pix in nbor_nbor_of_end if pix in this_stream_coord]
                nbor_is_not_stream = [pix for pix in nbor_nbor_of_end if pix not in nbor_is_stream]
                # border pixel should be next to more than one river pixel
                if (len(nbor_is_stream) <= 1):
                    continue
                # if bank is uninitialised, push border pixel onto queue
                # then loop again
                if len(this_bank) == 0:
                    this_queue.append(nbor)
                    this_bank.append(nbor)
                    continue
                # if bank IS initialised, then this neighbour pixel must also
                #       (1) be a neighbour of any of the current bank pixels
                #       (2) share non-river pixels
                nbor_in_bank = [pix for pix in nbor_nbor_of_end if pix in this_bank]
                if len(nbor_in_bank) <= 0:
                    continue
                # see if these pixels share any NON-river pixels
                for bnbor in nbor_in_bank:
                    nbor_nbor_of_bank = list(pixel_neighbours_from_east(x=bnbor[0], y=bnbor[1],
                                                                        xlim=(0, strdim[0] - 1),
                                                                        ylim=(0, strdim[1] - 1)))
                    bank_nbor_is_not_stream = [pix for pix in nbor_nbor_of_bank
                                               if pix not in this_stream_coord]
                    if any(pix in bank_nbor_is_not_stream for pix in nbor_is_not_stream):
                        this_queue.append(nbor)
                        this_bank.append(nbor)

            # assign ID to bank pixels ==============================
            for bank in this_bank:
                bank_map[bank] = bank_idx

            # queue and assign adjacent border pixels ===============
            got_to_end = False
            # while queue isn't empty
            while len(this_queue) > 0:
                # pinch off border pixel
                borderpix = this_queue.pop()
                # record a copy of the queue before adding to it,
                # in case this is end pixel
                prev_queue = this_queue[:]
                # find all neighbours that are [ignoring assigned border pixels]
                nbor_border = []  # (a) unassigned border pixels
                nbor_river = []  # (b) in the river
                nbor_neither = []  # (c) not in the river

                for pix in pixel_neighbours_from_east(x=borderpix[0],
                                                      y=borderpix[1],
                                                      xlim=(0, strdim[0] - 1),
                                                      ylim=(0, strdim[1] - 1)):
                    if pix in this_stream_coord:
                        nbor_river.append(pix)
                    elif (stream_border[pix] != 0) and (bank_map[pix] == 0):
                        nbor_border.append(pix)
                    elif (stream_border[pix] == 0) and (pix not in this_stream_coord):
                        nbor_neither.append(pix)

                for nbor in nbor_border:
                    # find (d) pixels in the river and (e) not in the river
                    if (bank_map[nbor] != 0) or (nbor in this_bank):  # don't re-record
                        continue
                    nbor_nbor_river = []
                    nbor_nbor_neither = []
                    nbor_nbor_border = []
                    for pix in list(pixel_neighbours_from_east(x=nbor[0], y=nbor[1],
                                                               xlim=(0, strdim[0] - 1),
                                                               ylim=(0, strdim[1] - 1))):
                        if pix in this_stream_coord:
                            # is a stream
                            nbor_nbor_river.append(pix)
                        elif (stream_border[pix] != 0) and (bank_map[pix] == 0):
                            # unassigned bank
                            nbor_nbor_border.append(pix)
                        elif (stream_border[pix] == 0) and (pix not in this_stream_coord):
                            # not river or bank
                            nbor_nbor_neither.append(pix)
                    # see if there is at least one shared river pixel between
                    # (b) from the border pixel and (d) its neighbour
                    shared_river = [spix for spix in nbor_nbor_river if spix in nbor_river]
                    # see if there is at least one shared "neither" pixel
                    # between (b) from the border pixel and (d) its neighbour
                    shared_neither = [spix for spix in nbor_nbor_neither if spix in nbor_neither]
                    # and again with
                    shared_border = [spix for spix in nbor_nbor_border if spix in nbor_border]
                    for shrd in shared_river:
                        if shrd not in stream_travelled:
                            stream_travelled.append(shrd)
                            # neighbouring pixel is only connected to the
                            # end pixel of a river; do not add to bank or queue
                    if (len(shared_river) == 1) and \
                            (shared_river[0] in this_stream_end) and \
                            (shared_river[0] != end):
                        got_to_end = True
                        continue
                    # skip if pixel is the end we started at
                    elif (len(shared_river) == 1) and (shared_river[0] == end):
                        continue
                    # else if river & neither-or-border pixels shared between
                    # border from queue and border-that-borders
                    elif (len(shared_river) > 0) and \
                            ((len(shared_neither) > 0) or (len(shared_border) > 0)):
                        # add border-that-borders to queue
                        if nbor not in this_bank:
                            this_queue.append(nbor)
                            this_bank.append(nbor)
                if got_to_end:
                    this_queue = prev_queue[:]  # in case we need to start
                    # buffering from the other end

            # assign ID to remainder of bank ========================
            for bank in this_bank:
                bank_map[bank] = bank_idx

            # =======================================================
            river_bank.append(this_bank)
            bank_idx += 1  # increment border pixel ID
    return bank_map


###############################################################################
def label_river_bank_buffers(streams_raster_uri, map_buffer=45.):
    '''
    Calculates two maps (numpy arrays) from the stream raster, a uniquely ID'd
    set of river banks (buffered out to the distance requested), and a second
    map where pixels that are equidistant  from two banks are given the ID of
    their alternately eligible bank.
    Returns both river bank and alternative river bank maps.

    Args :
        streams_raster_uri  - path to raster file of stream pixels
                             (expects stream pixels = 1)

    Args (optional) :
        map_buffer - buffer length in metres (raster unit not n_pixels) of
                     river bank. Default = 45m
    '''

    # read in the stream raster
    with rasterio.open(streams_raster_uri, 'r') as stream_raster:
        stream_data = stream_raster.read(1)
        stream_meta = stream_raster.meta
    # do buffering in native units of raster; hence need to calculate what the
    # size of the buffer is in pixels
    mapunits_per_pixel = abs(stream_meta['affine'][0])
    pixel_buffer = map_buffer / mapunits_per_pixel
    buffer_postfix = '_buffered' + str(int(map_buffer)) + 'm'
    # set up raster to keep a record of stream neighbour pixels
    logger.info('\t\tBuffering stream by %d pixels (%d metres)' %
                (pixel_buffer, map_buffer))
    buffered_stream_uri = buffer_postfix.join(splitext(streams_raster_uri))

    if not os.path.exists(buffered_stream_uri):
        buffered_stream = map_pixels_next_to_river(streams_raster_uri,
                                                   radius=pixel_buffer)
        with rasterio.open(buffered_stream_uri, 'w', **stream_meta) as buffered_stream_raster:
            buffered_stream_raster.write_band(1, buffered_stream.astype(stream_meta['dtype']))
    else:
        with rasterio.open(buffered_stream_uri, 'r') as buffered_stream_raster:
            buffered_stream = buffered_stream_raster.read(1)
    # will also add the coordinates of the streams
    buffered_stream = buffered_stream + stream_data
    # first figure out which bank each buffer pixel is closest to
    logger.info('\t\tIdentifying banks')
    bankmap_raster_uri = '_bankmap'.join(splitext(streams_raster_uri))
    if not os.path.exists(bankmap_raster_uri):
        bank_map = label_river_banks(streams_raster_uri)
        with rasterio.open(bankmap_raster_uri, 'w', **stream_meta) as bankmap_raster:
            bankmap_raster.write_band(1, bank_map.astype(stream_meta['dtype']))
    else:
        with rasterio.open(bankmap_raster_uri, 'r') as bankmap_raster:
            bank_map = bankmap_raster.read(1)
    is_bank = np.where(bank_map != 0)
    bank_coords = zip(is_bank[0], is_bank[1])
    len_bank_coords = len(bank_coords)
    bnkdim = bank_map.shape
    logger.info('\t\tAssigning buffers to banks')
    # calculate squared distance to save us a square root operation
    # initialize distance to buffer as number _above_ the pixel_buffer size (will clip later)
    # NOTE: THIS IS SCALED TO PIXEL BUFFER PLUS 20%
    sq_dist_to_bank = np.ones(bnkdim).astype(bank_map.dtype) * ((pixel_buffer*1.2)**2)
    buff_bank = np.zeros(bnkdim).astype(bank_map.dtype)
    # shared_bank = secondary map: pixel is equidistant from two banks
    shared_bank = np.zeros(bnkdim).astype(bank_map.dtype)
    loopcount = 0
    altcount = 0
    # loop once to assign 'best' bank pixel
    for bnkpx in bank_coords:
        # fetch nearby buffer pixels that have not been assigned to this bank
        loopcount += 1
        if ((loopcount % 500) == 0):
            logger.debug('\t\t %d out of %d bank pixels assigned' % (loopcount, len_bank_coords))
        nearest_buffer_pixels = []
        # Clipping the actual bank distance here.
        for pix in get_pixels_within_radius(x=bnkpx[0], y=bnkpx[1],
                                            xlim=(0, bnkdim[0] - 1),
                                            ylim=(0, bnkdim[1] - 1),
                                            radius=pixel_buffer):
            if (buffered_stream[pix] != 0):
                nearest_buffer_pixels.append(pix)
        for nbpx in nearest_buffer_pixels:
            sq_buf_bnk_dist = (bnkpx[0] - nbpx[0])**2 + (bnkpx[1] - nbpx[1])**2
            if sq_buf_bnk_dist < sq_dist_to_bank[nbpx]:
                sq_dist_to_bank[nbpx] = sq_buf_bnk_dist
                buff_bank[nbpx] = bank_map[bnkpx]
    logger.info('\t\tSearching for ties')
    # now loop again to find pixels that belong to more than one bank
    # (i.e. they are equidistant from another pixel)
    # N.B. can't do this in loop above as the min distances are required
    # (and they may not have minimized in 1st loop)
    for bnkpx in bank_coords:
        # fetch nearest buffer pixels that have _not_ been assigned to this bank
        loopcount += 1
        if ((loopcount % 500) == 0):
            logger.debug('\t\t %d / %d (%d total) bank pixels alternately assigned' %
                         (altcount, loopcount, len_bank_coords))
        nearest_buffer_pixels = []
        for pix in get_pixels_within_radius(x=bnkpx[0], y=bnkpx[1],
                                            xlim=(0, bnkdim[0] - 1),
                                            ylim=(0, bnkdim[1] - 1),
                                            radius=pixel_buffer):
            if (buffered_stream[pix] != 0) and (buff_bank[pix] != bank_map[bnkpx]):
                nearest_buffer_pixels.append(pix)
        # check misassigned pixels are equidistant to river from other pixel
        for nbpx in nearest_buffer_pixels:
            sq_buf_bnk_dist = (bnkpx[0] - nbpx[0])**2 + (bnkpx[1] - nbpx[1])**2
            if (sq_buf_bnk_dist == sq_dist_to_bank[nbpx]):
                if (buff_bank[nbpx] != bank_map[bnkpx]):
                    if (buff_bank[nbpx] != 0):
                        altcount += 1
                        shared_bank[nbpx] = bank_map[bnkpx]
    return buff_bank, shared_bank


###############################################################################
def calculate_riparian_index(streams_raster_uri, retention_index_uri,
                             output_riparian_index_uri, map_buffer=45.):
    '''
    Fulfils purpose of the riparian_index routine from RIOS_Pre_Processing.py
    Calculates maximum riparian retention index (average of 3x3 pixel matrix
    around river bank pixel) in river bank and saves to file.

    Args :
        streams_raster_uri  - path to raster file of stream pixels
                              (expects stream pixels = 1)
        retention_index_uri - path to raster file of indexed pixel retention
                              (of E/P/N/F/G data)
        output_riparian_index_uri - output path to raster file of riparian index

    Args (optional) :
        map_buffer - buffer length in metres (raster unit not n_pixels) of
                     river bank. Default = 45m
    '''
    # get and identify the buffered banks
    str_buffer = str(int(map_buffer)) + 'm'
    buffered_bank_raster_uri = (str_buffer+'bufferbank').join(splitext(streams_raster_uri))
    shared_bank_raster_uri = (str_buffer+'sharedbank').join(splitext(streams_raster_uri))

    if os.path.exists(buffered_bank_raster_uri) and os.path.exists(shared_bank_raster_uri):
        with rasterio.open(buffered_bank_raster_uri, 'r') as buffered_bank_raster:
            buffered_bank = buffered_bank_raster.read(1)
        with rasterio.open(shared_bank_raster_uri, 'r') as shared_bank_raster:
            shared_bank = shared_bank_raster.read(1)
    else:
        with rasterio.open(streams_raster_uri, 'r') as streams_raster:
            streams_meta = streams_raster.meta
        buffered_bank, shared_bank = label_river_bank_buffers(streams_raster_uri,
                                                              map_buffer=map_buffer)
        with rasterio.open(buffered_bank_raster_uri, 'w', **streams_meta) as buffered_bank_raster:
            buffered_bank_raster.write_band(1, buffered_bank.astype(streams_meta['dtype']))
        with rasterio.open(shared_bank_raster_uri, 'w', **streams_meta) as shared_bank_raster:
            shared_bank_raster.write_band(1, shared_bank.astype(streams_meta['dtype']))

    unique_buffer_id = [buffid for buffid in set(buffered_bank.ravel().tolist()
                                                 + shared_bank.ravel().tolist()) if buffid != 0]
    bufdim = buffered_bank.shape

    with rasterio.open(retention_index_uri, 'r') as retention_raster:
        retention_data = retention_raster.read(1)
        retention_meta = retention_raster.meta
    riparian_index_data = np.zeros(retention_data.shape).astype(retention_data.dtype)

    for buffer_id in unique_buffer_id:
        # make a temporary map of only the pixels with this bank ID
        this_buffer = np.zeros(bufdim).astype(buffered_bank.dtype)
        this_buffer[np.where(buffered_bank == buffer_id)] = 1
        this_buffer[np.where(shared_bank == buffer_id)] = 1
        # for all pixels in this buffer, figure out 3x3 matrix mean
        # around each pixel
        this_buffer_good = np.where(this_buffer > 0)
        for bufpix in zip(this_buffer_good[0], this_buffer_good[1]):
            # identify pixels in 3x3 matrix that are also part of this buffer
            fullpixmatrix = [bufpix] + list(get_neighbouring_pixels(x=bufpix[0], y=bufpix[1],
                                                                     xlim=(0, bufdim[0] - 1),
                                                                     ylim=(0, bufdim[1] - 1),
                                                                     radius=1))
            # separating out bufer pixels for mean calculation
            pixmatrix = [pixmat for pixmat in fullpixmatrix if (this_buffer[pixmat] == 1)]
            # calculate mean value of pixel
            sum_retention = np.sum([retention_data[pix] for pix in pixmatrix]).astype(float)
            mean_retention = sum_retention / len(pixmatrix)
            # riparian index is the maximum value of the mean calculation
            riparian_index_data[bufpix] = np.max([riparian_index_data[bufpix], mean_retention])
    with rasterio.open(output_riparian_index_uri, 'w', **retention_meta) as riparian_index:
        riparian_index.write_band(1, riparian_index_data)


###############################################################################
def raster_value_to_index(input_raster_uri, output_raster_uri, value_bounds,
                          replacement_index):
    '''
    Takes raster of continuous data and replaces values within/outside various
    bounding values with indices. Note that the number of indices should be one
    more than the number of boundaries.

    Args :
        input_raster_uri  - path to raster file of continuous data
        output_raster_uri  - path to file where indexed raster will be output
        value_bounds -  array of values where each pair of consecutive values
                        form the boundary values to index
        output_riparian_index_uri - output path to raster file of riparian index

    '''

    if hasattr(replacement_index, 'sort'):
        replacement_index = np.array(replacement_index)
    if len(replacement_index) <= 1:
        raise (ValueError, 'Missing replacement indices')
    if hasattr(value_bounds, 'format'):
        value_bounds = [value_bounds]
    if len(value_bounds) != len(replacement_index) - 1:
        raise (ValueError, 'There are an inconsistent number (n) of values '
                           + 'and replacement indices (n+1)')

    # read input raster
    with rasterio.open(input_raster_uri, 'r') as input_raster:
        input_data = input_raster.read(1)
        input_meta = input_raster.meta

    # make a new raster from input_data
    output_data = input_data.copy()
    output_data[:] = replacement_index[0]

    # set output raster values to the indices in the appropriate ranges
    for bidx in range(len(value_bounds)):
        if value_bounds[bidx] != value_bounds[-1]:
            this_range = np.where((input_data > value_bounds[bidx]) &
                                  (input_data <= value_bounds[bidx + 1]))
        else:
            this_range = np.where(input_data > value_bounds[bidx])
        if len(this_range[0]) > 0:
            output_data[this_range] = replacement_index[bidx + 1]

    # propogate errors
    bad_pixel = np.where(input_data == input_meta['nodata'])
    output_data[bad_pixel] = input_meta['nodata']

    with rasterio.open(output_raster_uri, 'w', **input_meta) as output_raster:
        output_raster.write_band(1, output_data.astype(input_meta['dtype']))


###############################################################################
###############################################################################
def main(working_path, output_path, hydro_path, rios_coeff_table,
         lulc_raster_uri, dem_raster_uri,
         erosivity_raster_uri=None, erodibility_raster_uri=None,
         soil_depth_raster_uri=None, precip_month_raster_uri=None,
         soil_texture_raster_uri=None, precip_annual_raster_uri=None,
         aet_raster_uri=None,
         river_buffer_dist=45., suffix="",
         aoi_shape_uri=None, river_reference_shape_uri_list=None,
         streams_raster_uri=None,
         do_erosion=False, do_nutrient_p=False, do_nutrient_n=False,
         do_flood=False, do_gw_bf=False,
         verbose=True, clean_intermediate_files=False):
    '''
    The main process that replaces the ArcGIS RIOS_Pre_Processing script.
    It calculates the inputs for the RIOS IPA program such as
    downslope retention index, upslope source, riparian index,
    and slope index appropriately for Erosion Control, Phosphorus Retention,
    Nitrogen Retention, Flood Mitigation, and Groundwater Retention/Baseflow.

    Args :
        working_path     - path to directory for preprocessor intermediate files
        output_path      - path to directory for preprocessor outputs
        hydro_path       - path to directory for flow direction/path/
                           accumulation rasters
        rios_coeff_table - path to csv table containing biophysical coefficients
        lulc_raster_uri  - path to raster of land use/land cover
        dem_raster_uri   - path to raster of digital elevation

    Args [grouped by category] (optional):
        [Rasters to input data]
        erosivity_raster_uri     - path to raster of rainfall erosivity
        erodibility_raster_uri   - path to raster of soil erodibility
        soil_depth_raster_uri    - path to raster of soil depth
        precip_month_raster_uri  - path to raster of peak monthly precipitation
        soil_texture_raster_uri  - path to raster of soil texture
        precip_annual_raster_uri - path to raster of annual precipitaion
        aet_raster_uri           - path to raster of actual evapotranspiration

        [Calculation-specific inputs/data]
        river_buffer_dist - extent of riparian buffer (in raster
                                          map units e.g. metres) Default = 45(m)
        suffix            - string to identify output files
        aoi_shape_uri     - path to shapefile of area of interest

        [River data sources]
        river_reference_shape_uri_list - path to list of shapefiles describing
                                         rivers in the area
        streams_raster_uri             - path to raster describing DEM-
                                         compatible stream (stream pixels = 1)

        [Flags to trigger data preparation for RIOS objectives]
        do_erosion    - runs erosion control objective
        do_nutrient_p - runs phosphorus fixing objective
        do_nutrient_n - runs nitrogen fixing objective
        do_flood      - runs flood control objective
        do_gw_bf      - runs groundwater retention/baseflow objective
        
        [Misc]
        verbose - set logging status of function to DEBUG
    '''
    # get basic setups for objectives and datasets
    objective = get_objective_dictionary(suffix=suffix,
                                         do_erosion=do_erosion,
                                         do_nutrient_p=do_nutrient_p,
                                         do_nutrient_n=do_nutrient_n,
                                         do_flood=do_flood,
                                         do_gw_bf=do_gw_bf)
    input_data = get_input_data_param_dictionary()

    # make a bunch of lists to keep logs
    parameter_log = []  # replacement for parameters file to be written

    # With parameters log; later write input parameter values to an output file
    parameter_log.append("Date and Time: " + time.strftime("%Y-%m-%d %H:%M"))
    logger.info("Validating arguments...")

    # Log whether we calculate inputs for the objectives
    for obj in objective:
        parameter_log.append("Calculate for %s: %s" %
                             (obj, str(objective[obj]['found']).lower()))
    # Directory where output files will be written
    working_path = os.path.normpath(working_path) + os.sep
    parameter_log.append("Workspace: " + working_path)
    output_path = os.path.normpath(output_path) + os.sep
    parameter_log.append("Output path: " + output_path)
    # Describe what the data is
    for indata in input_data:
        this_param = locals()[input_data[indata]['param']]
        parameter_log.append(("%s: %s" % (indata, this_param)))
        if ("".join(this_param.split()) != "") and ("".join(this_param.split()) != "#"):
            input_data[indata]['found'] = True

    # suffix to add to end of output filenames, as <filename>_<suffix>
    parameter_log.append("Suffix: " + suffix)
    logger.info(parameter_log[-1])

    if ("".join(suffix.split()) == "") or (suffix == "#"):
        suffix = ""
    # note: add the underscore when needed, not before

    # Make sure that required inputs are provided for each objective chosen
    input_raster_list = []
    missing_data = False
    for obj in objective:  # So for each objective...
        if objective[obj]['found']:  # ... that we have found
            logger.info(obj + " selected, checking sources: ")
            for dataset in objective[obj]['dataset']:  # check  each dataset
                if input_data[dataset]['found']:  # ... has also been found
                    logger.debug("\t" + dataset)
                    # .. and if it's a raster
                    if locals()[input_data[dataset]['param']].endswith('tif'):
                        # ... save it to our list
                        input_raster_list.append(locals()[input_data[dataset]['param']])
                else:
                    logger.error("Missing Data: %s %s required for %s" %
                                 (dataset, input_data[dataset]['type'], obj))
                    missing_data = True  # if not found: log + flag problem

    # Handle exceptions.
    if missing_data:
        raise IOError("Please identify all required data inputs.")
    del missing_data  # housekeeping

    # Check and create intermediate/output folders
    for folder in [output_path, working_path]:
        if not os.path.exists(folder):
            os.mkdir(folder)

###############################################################################
    # Output files

    # Intermediate files that are not objective specific
    flow_dir_channels_raster_uri = working_path + "flowdir_channels_" + suffix + ".tif"
    slope_index_uri = working_path + "slope_idx_" + suffix + ".tif"
    erosivity_index_uri = working_path + "eros_idx_" + suffix + ".tif"
    erodibility_index_uri = working_path + "erod_idx_" + suffix + ".tif"
    soil_depth_norm_raster_uri = working_path + "sdepth_norm_" + suffix + ".tif"
    soil_depth_index_uri = working_path + "sdepth_idx_" + suffix + ".tif"

    # Field names in RIOS coefficient table
    lucode_field = "lucode"
    sed_ret_rios_field = "sed_ret"
    sed_exp_rios_field = "sed_exp"
    n_exp_rios_field = "N_exp"
    n_ret_rios_field = "N_ret"
    p_exp_rios_field = "P_exp"
    p_ret_rios_field = "P_ret"
    roughness_rios_field = "rough_rank"
    cover_rios_field = "cover_rank"
    rios_fields = [sed_ret_rios_field, sed_exp_rios_field, n_exp_rios_field,
                   n_ret_rios_field, p_exp_rios_field, p_ret_rios_field,
                   roughness_rios_field, cover_rios_field]
    # Keep track of whether frequently-used layers have been created in this run
    # Want to override previous runs, but re-use current versions
    made_lulc_coeffs = False
    made_flowdir_channels = False
    made_slope_index = False
    made_flood_slope_index = False
    made_gwater_slope_index = False
    made_soil_depth_index = False
    made_erosivity_index = False
    made_erodibility_index = False
    made_flgw_index_cover = False
    made_flgw_index_rough = False

###############################################################################
    # Start using geoprocessor for stuff

    # Check input raster projections - they should all be the same
    logger.debug("Checking input raster projections...")

    # Let's just enforce the DEM projection
    with rasterio.open(dem_raster_uri, 'r') as DEMdata:
        DEMprojectionwkt = DEMdata.crs_wkt

    demprojectionname = re.findall('\".*?\"', DEMprojectionwkt)[0]
    logger.info("DEM projected as %s" % demprojectionname)
    errorprojection = False

    for inraster in input_raster_list:
        with rasterio.open(inraster, 'r') as rasterdata:
            rasterprojectionwkt = rasterdata.crs_wkt
            if DEMprojectionwkt != rasterprojectionwkt:
                logger.error(inraster + " does not appear to be projected as "
                             + demprojectionname)
                logger.error(basename(inraster) + "projected as "
                             + rasterprojectionwkt)
                errorprojection = True

    if errorprojection:
        raise AssertionError("Input rasters must be in the same projection")
    del errorprojection

    ###########################################################################
    # Preprocess DEM derivatives for hydrological routing
    logger.info("Creating hydrology layers...")

    # Create flow direction raster
    flow_dir_raster_uri = hydro_path + basename("_flow_dir".join(splitext(dem_raster_uri)))
    if not os.path.exists(flow_dir_raster_uri):
        logger.debug("Calculating Flow Direction raster")
        pygrout.routing.flow_direction_d_inf(dem_raster_uri, flow_dir_raster_uri)

    # Create slope raster
    slope_raster_uri = hydro_path + basename("_slope".join(splitext(dem_raster_uri)))
    if not os.path.exists(slope_raster_uri):
        logger.debug("Calculating Slope raster")
        pygeo.calculate_slope(dem_raster_uri, slope_raster_uri, aoi_uri=None)

    # Create flow accumulation raster
    flow_acc_raster_uri = hydro_path + basename("_flow_acc".join(splitext(dem_raster_uri)))
    if not os.path.exists(flow_acc_raster_uri):
        logger.debug("Calculating Flow Accumulation raster")
        pygrout.routing.flow_accumulation(flow_dir_raster_uri, dem_raster_uri,
                                          flow_acc_raster_uri)
        change_geotiff_nodata_value(flow_acc_raster_uri, new_nodata=-9999)

    if (streams_raster_uri is None):
        streams_raster_uri = output_path + "streams_" + suffix + ".tif"

    if not os.path.exists(streams_raster_uri):
        logger.warning("No streams raster supplied")
        if river_reference_shape_uri_list is None:
            logger.error("Missing streams raster (%s) and no shapefiles supplied " +
                         basename(streams_raster_uri) )
            raise IOError("No stream raster supplied nor can be created")
        elif len(river_reference_shape_uri_list) == 0:
            logger.error("Missing streams raster (%s) and no shapefiles supplied " +
                         basename(streams_raster_uri))
            raise IOError("No stream raster supplied nor can be created")
        else:
            # real purpose of this function is to create the stream raster;
            # 'threshold flow accumulation' value is deprecated
            logger.info("Creating streams raster from reference shapefiles")
            thflac = optimize_threshold_flowacc(flow_acc_raster_uri,
                                                river_reference_shape_uri_list,
                                                workspace_path=working_path,
                                                suffix=suffix, seedlen=1000,
                                                aoi_shape_uri=aoi_shape_uri,
                                                streams_raster_uri=streams_raster_uri)

    # Set flow direction raster to null where there are streams
    if not made_flowdir_channels:
        logger.info("Defining flow direction channels...")
        if not os.path.exists(flow_dir_channels_raster_uri):
            define_channels(flow_dir_raster_uri,
                            flow_dir_channels_raster_uri,
                            streams_raster_uri)
        made_flowdir_channels = True

    ################################################################################
    # Make sure that at least one objective is chosen
    if True not in [objective[obj]['found'] for obj in objective.keys()]:
        logger.error("Error: No objectives were selected. "
                     + "Please choose objectives to be pre-processed.")
        raise IOError("No objectives were selected.")

    ################################################################################
    ### Preprocess remaining variables for RIOS calculation
    ## LULC Index-R
    if not made_lulc_coeffs:  # N.B. this is an internal pandas table, not a file
        logger.info("\tMapping coefficients to landcover...")
        lulc_coeff_df = map_coefficients(lulc_raster_uri, lucode_field, rios_coeff_table)
        made_lulc_coeffs = True

    # Soil depth index
    if not made_soil_depth_index:
        if not os.path.exists(soil_depth_index_uri):
            normalize(soil_depth_raster_uri, soil_depth_index_uri)
        made_soil_depth_index = True
    if not made_slope_index:
        if not os.path.exists(slope_index_uri):
            normalize(slope_raster_uri, slope_index_uri)
        made_slope_index = True

    ################################################################################
    ### Process Erosion Control objective
    if objective['Erosion Control']['found']:

        this_obj = "Erosion Control"
        intermediate_files = objective[this_obj]["intermediate"]
        output_files = objective[this_obj]["output"]
        erosion_index_exp = working_path + intermediate_files['index_exp']
        erosion_index_ret = working_path + intermediate_files['index_ret']
        erosion_comb_weight_R = working_path  + intermediate_files['comb_weight_R']
        erosion_dret_flowlen = working_path + intermediate_files['dret_flowlen']
        erosion_comb_weight_Exp = working_path + intermediate_files['comb_weight_Exp']
        erosion_dret_index = output_path + output_files['dret_index']
        erosion_upslope_source = output_path + output_files['upslope_source']
        erosion_riparian_index = output_path + output_files['riparian_index']

        logger.info("Processing Erosion Control objective...")
        try:
            # Make Export and Retention Index rasters
            if not os.path.exists(erosion_index_exp):
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, sed_exp_rios_field,
                                        erosion_index_exp)

            if not os.path.exists(erosion_index_ret):
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, sed_ret_rios_field,
                                        erosion_index_ret)

            logger.debug("\tCreating downslope retention index...")
            ## Combined weight retention calculated to extend flow length by
            #  slope and erosion factor

            if not os.path.exists(erosion_comb_weight_R):
                average_raster(raster_uri_list=[erosion_index_ret],
                               inverseraster_uri_list=[slope_index_uri],
                               output_raster_uri=erosion_comb_weight_R)

            ## Downslope retention index
            if not os.path.exists(erosion_dret_flowlen):
                pygrout.routing.distance_to_stream(flow_dir_raster_uri,
                                                   streams_raster_uri,
                                                   erosion_dret_flowlen,
                                                   factor_uri=erosion_comb_weight_R)

            if not os.path.exists(erosion_dret_index):
                normalize(erosion_dret_flowlen, erosion_dret_index)
            ####
            logger.info("\tCreated Erosion downslope retention index: "
                          + basename(erosion_dret_index))
            logger.debug("\tCreating upslope source...")
            # Erosivity index
            if not made_erosivity_index:
                if not os.path.exists(erosivity_index_uri):
                    normalize(erosivity_raster_uri, erosivity_index_uri)
                made_erosivity_index = True

            # Erodibility index
            if not made_erodibility_index:
                if not os.path.exists(erodibility_index_uri):
                    normalize(erodibility_raster_uri, erodibility_index_uri)
                made_erodibility_index = True

            # Combined weight export
            if not os.path.exists(erosion_comb_weight_Exp):
                average_raster(raster_uri_list=[slope_index_uri,
                                                erosivity_index_uri,
                                                erodibility_index_uri,
                                                soil_depth_index_uri,
                                                erosion_index_exp],
                               inverseraster_uri_list=[erosion_index_ret],
                               output_raster_uri=erosion_comb_weight_Exp)
            ## Upslope source
            ## Not an index because we're not normalizing in this script
            if not os.path.exists(erosion_upslope_source):
                weighted_flow_accumulation(flow_dir_raster_uri,
                                           dem_raster_uri, erosion_upslope_source,
                                           source_weight_uri=erosion_comb_weight_Exp)
            logger.info("\tCreated Erosion upslope source: "
                        + basename(erosion_upslope_source))

            ## Riparian continuity
            logger.debug("\tCreating riparian index...")

            if not os.path.exists(erosion_riparian_index):
                calculate_riparian_index(streams_raster_uri=streams_raster_uri,
                                         retention_index_uri=erosion_index_ret,
                                         output_riparian_index_uri=erosion_riparian_index,
                                         map_buffer=river_buffer_dist)
            logger.info("\tCreated Erosion riparian continuity index: "
                        + basename(erosion_riparian_index))

        except:
            logger.error("Error processing Erosion Control objective")
            raise RuntimeError("Error processing Erosion Control objective")

        ################################################################################
    ### Process Phosphorus Retention objective
    if objective['Phosphorus Retention']['found']:
        this_obj = "Phosphorus Retention"
        intermediate_files = objective[this_obj]["intermediate"]
        output_files = objective[this_obj]["output"]
        phosphorus_index_exp = working_path  + intermediate_files['index_exp']
        phosphorus_index_ret = working_path + intermediate_files['index_ret']
        phosphorus_comb_weight_R = working_path + intermediate_files['comb_weight_R']
        phosphorus_dret_flowlen = working_path + intermediate_files['dret_flowlen']
        phosphorus_dret_index = output_path + output_files['dret_index']
        phosphorus_comb_weight_Exp = working_path + intermediate_files['comb_weight_Exp']
        phosphorus_upslope_source = output_path + output_files['upslope_source']
        phosphorus_riparian_index =  output_path + output_files['riparian_index']

        try:
            logger.info("Processing Phosphorus Retention objective...")

            # Make Export and Retention Index rasters
            if not os.path.exists(phosphorus_index_exp):
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, p_exp_rios_field,
                                        phosphorus_index_exp)
            if not os.path.exists(phosphorus_index_ret):
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, p_ret_rios_field,
                                        phosphorus_index_ret)

            logger.debug("\tCreating downslope retention index...")

            ## Combined weight retention calculated to extend flow length by slope
            #  and erosion factor
            if not os.path.exists(phosphorus_comb_weight_R):
                average_raster(raster_uri_list=[phosphorus_index_ret],
                               inverseraster_uri_list=[slope_index_uri],
                               output_raster_uri=phosphorus_comb_weight_R)

            ## Downslope retention index
            if not os.path.exists(phosphorus_dret_flowlen):
                pygrout.routing.distance_to_stream(flow_dir_raster_uri,
                                                   streams_raster_uri,
                                                   phosphorus_dret_flowlen,
                                                   factor_uri=phosphorus_comb_weight_R)
            if not os.path.exists(phosphorus_dret_index):
                normalize(phosphorus_dret_flowlen, phosphorus_dret_index)
            logger.info("\t... created Phosphorus downslope retention index: "
                          + basename(phosphorus_dret_index))

            logger.debug("\tCreating upslope source...")
            # Erosivity index
            if not made_erosivity_index:
                if not os.path.exists(erosivity_index_uri):
                    normalize(erosivity_raster_uri, erosivity_index_uri)
                made_erosivity_index = True
            # Erodibility index
            if not made_erodibility_index:
                if not os.path.exists(erodibility_index_uri):
                    normalize(erodibility_raster_uri, erodibility_index_uri)
                made_erodibility_index = True

            # Combined weight export
            if not os.path.exists(phosphorus_comb_weight_Exp):
                average_raster(
                    raster_uri_list=[slope_index_uri,
                                     erosivity_index_uri,
                                     erodibility_index_uri,
                                     soil_depth_index_uri,
                                     phosphorus_index_exp],
                    inverseraster_uri_list=[phosphorus_index_ret],
                    output_raster_uri=phosphorus_comb_weight_Exp)

            ## Upslope source
            if not os.path.exists(phosphorus_upslope_source):
                weighted_flow_accumulation(flow_dir_raster_uri,
                                           dem_raster_uri,
                                           phosphorus_upslope_source,
                                           source_weight_uri=phosphorus_comb_weight_Exp)
            logger.info("\t... created Phosphorus upslope source: "
                        + basename(phosphorus_upslope_source))

            ## Riparian continuity
            logger.debug("\tCreating riparian index...")
            if not os.path.exists(phosphorus_riparian_index):
                calculate_riparian_index(streams_raster_uri=streams_raster_uri,
                                         retention_index_uri=phosphorus_index_ret,
                                         output_riparian_index_uri=phosphorus_riparian_index,
                                         map_buffer=river_buffer_dist)
            logger.info("\t... created Phosphorus riparian continuity index: "
                        + basename(phosphorus_riparian_index))

        except:
            logger.error("Error processing Phosphorus Retention objective")
            raise RuntimeError("Error processing Phosphorus Retention objective")

################################################################################
    ### Process Nitrogen Retention objective

    if objective['Nitrogen Retention']['found']:
        this_obj = "Nitrogen Retention"
        intermediate_files = objective[this_obj]["intermediate"]
        output_files = objective[this_obj]["output"]
        nitrogen_index_exp = working_path + intermediate_files['index_exp']
        nitrogen_index_ret = working_path + intermediate_files['index_ret']
        nitrogen_comb_weight_R = working_path + intermediate_files['comb_weight_R']
        nitrogen_dret_flowlen = working_path + intermediate_files['dret_flowlen']
        nitrogen_dret_index = output_path + output_files['dret_index']
        nitrogen_comb_weight_Exp = working_path + intermediate_files['comb_weight_Exp']
        nitrogen_upslope_source = output_path + output_files['upslope_source']
        nitrogen_riparian_index = output_path + output_files['riparian_index']

        try:
            logger.info("Processing Nitrogen Retention objective...")
            # Make Export and Retention Index rasters
            if not os.path.exists(nitrogen_index_exp):
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, n_exp_rios_field,
                                        nitrogen_index_exp)
            if not os.path.exists(nitrogen_index_ret):
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, n_ret_rios_field,
                                        nitrogen_index_ret)

            logger.debug("\tCreating downslope retention index..." + time.asctime())

            ## Combined weight retention
            if not made_slope_index:
                if not os.path.exists(slope_index_uri):
                    normalize(slope_raster_uri, slope_index_uri)
                made_slope_index = True

            if not os.path.exists(nitrogen_comb_weight_R):
                average_raster(raster_uri_list=[nitrogen_index_ret],
                               inverseraster_uri_list=[slope_index_uri],
                               output_raster_uri=nitrogen_comb_weight_R)

            ## Downslope retention index

            if not os.path.exists(nitrogen_dret_flowlen):
                pygrout.routing.distance_to_stream(flow_dir_raster_uri,
                                                   streams_raster_uri,
                                                   nitrogen_dret_flowlen,
                                                   factor_uri=nitrogen_comb_weight_R)

            if not os.path.exists(nitrogen_dret_index):
                normalize(nitrogen_dret_flowlen, nitrogen_dret_index)
            logger.info("\t... created Nitrogen downslope retention index: "
                        + basename(nitrogen_dret_index))
            logger.debug("\tCreating upslope source...")
            # Combined weight export
            if not os.path.exists(nitrogen_comb_weight_Exp):
                average_raster(raster_uri_list=[slope_index_uri,
                                                soil_depth_index_uri,
                                                nitrogen_index_exp],
                               inverseraster_uri_list=[nitrogen_index_ret],
                               output_raster_uri=nitrogen_comb_weight_Exp)

            ## Upslope source
            if not os.path.exists(nitrogen_upslope_source):
                weighted_flow_accumulation(flow_dir_raster_uri,
                                           dem_raster_uri,
                                           nitrogen_upslope_source,
                                           source_weight_uri=nitrogen_comb_weight_Exp)
            logger.info("\t... created Nitrogen upslope source: "
                               + basename(nitrogen_upslope_source))

            ## Riparian continuity
            logger.debug("\tCreating riparian index...")
            if not os.path.exists(nitrogen_riparian_index):
                calculate_riparian_index(streams_raster_uri=streams_raster_uri,
                                         retention_index_uri=nitrogen_index_ret,
                                         output_riparian_index_uri=nitrogen_riparian_index,
                                         map_buffer=river_buffer_dist)
            logger.info("\t... created Nitrogen riparian continuity index: "
                               + basename(nitrogen_riparian_index))

        except:
            logger.error("Error processing Nitrogen Retention objective")
            raise RuntimeError("Error processing Nitrogen Retention objective")

    ###########################################################################
    ### Process Flood Mitigation objective
    if objective['Flood Mitigation']['found']:
        this_obj = "Flood Mitigation"
        intermediate_files = objective[this_obj]["intermediate"]
        output_files = objective[this_obj]["output"]
        flood_index_cover = working_path + intermediate_files['index_cover']
        flood_index_rough = working_path + intermediate_files['index_rough']
        flood_riparian_index = output_path + output_files['riparian_index']
        flood_slope_index = output_path +output_files['slope_index']
        flood_comb_weight_R = working_path + intermediate_files['comb_weight_ret']
        flood_dret_flowlen = working_path + intermediate_files['dret_flowlen']
        flood_dret_index = output_path + output_files['dret_index']
        flood_rainfall_depth_index = working_path + intermediate_files['rainfall_depth_index']
        flood_comb_weight_source = working_path + intermediate_files['comb_weight_source']
        flood_upslope_source = output_path + output_files['upslope_source']

        logger.info("Processing Flood Mitigation objective...")

        # Make Cover and Roughness Index rasters
        if not os.path.exists(flood_index_cover):
            derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                    lulc_coeff_df, cover_rios_field,
                                    flood_index_cover)

        if not os.path.exists(flood_index_rough):
            derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                    lulc_coeff_df, roughness_rios_field,
                                    flood_index_rough)

        # Riparian continuity
        logger.debug("\tCreating riparian index...")

        if not os.path.exists(flood_riparian_index):
            calculate_riparian_index(streams_raster_uri=streams_raster_uri,
                                     retention_index_uri=flood_index_rough,
                                     output_riparian_index_uri=flood_riparian_index,
                                     map_buffer=river_buffer_dist)

        logger.info("\t... created Flood Mitigation riparian continuity index: "
                    + basename(flood_riparian_index))

        # Slope index - binned, not normalized
        logger.debug("\tCreating slope index..." + time.asctime())

        value_bounds = [5.0, 10.0]
        replacement_index = [0.33, 0.66, 1.0]
        if not os.path.exists(flood_slope_index):
            raster_value_to_index(slope_raster_uri, flood_slope_index,
                                  value_bounds, replacement_index)

        logger.info("\t... created Flood slope index: "
                    + basename(flood_slope_index))

        # Combined weight R
        logger.debug("\tCreating downslope retention index...")

        if not os.path.exists(flood_comb_weight_R):
            average_raster(raster_uri_list=[flood_index_rough],
                           inverseraster_uri_list=[flood_slope_index],
                           output_raster_uri=flood_comb_weight_R)

        ## Downslope retention index
        if not os.path.exists(flood_dret_flowlen):
            pygrout.routing.distance_to_stream(flow_dir_raster_uri,
                                               streams_raster_uri,
                                               flood_dret_flowlen,
                                               factor_uri=flood_comb_weight_R)
        if not os.path.exists(flood_dret_index):
            normalize(flood_dret_flowlen, flood_dret_index)

        logger.info("\t... created Flood Mitigation downslope retention index: "
                     + basename(flood_dret_index))

        logger.debug("\tCreating upslope source...")
        # Rainfall depth index
        if not os.path.exists(flood_rainfall_depth_index):
            normalize(precip_month_raster_uri, flood_rainfall_depth_index)

        # Combined weight source
        if not os.path.exists(flood_comb_weight_source):
            average_raster(raster_uri_list=[flood_rainfall_depth_index,
                                            soil_texture_raster_uri,
                                            flood_slope_index],
                           inverseraster_uri_list=[flood_index_cover,
                                                   flood_index_rough],
                           output_raster_uri=flood_comb_weight_source)

        ## Upslope source
        if not os.path.exists(flood_upslope_source):
            weighted_flow_accumulation(flow_dir_raster_uri,
                                       dem_raster_uri,
                                       flood_upslope_source,
                                       source_weight_uri=flood_comb_weight_source)
        logger.info("\t... created Flood Mitigation upslope source: "
                           + basename(flood_upslope_source))

    ################################################################################
    ### Process Groundwater Recharge/Baseflow objective
    if objective['Groundwater Recharge/Baseflow']['found']:
        this_obj = "Groundwater Recharge/Baseflow"
        intermediate_files = objective[this_obj]["intermediate"]
        output_files = objective[this_obj]["output"]
        gwater_index_cover = working_path + intermediate_files['index_cover']
        gwater_index_rough = working_path + intermediate_files['index_rough']
        gwater_slope_index = output_path + output_files['slope_index']
        gwater_comb_weight_R = working_path + intermediate_files['comb_weight_ret']
        gwater_dret_flowlen = working_path + intermediate_files['dret_flowlen']
        gwater_dret_index = output_path + output_files['dret_index']
        gwater_precip_annual_index = working_path + intermediate_files['precip_annual_index']
        gwater_aet_index = working_path + intermediate_files['aet_index']
        gwater_comb_weight_source = working_path + intermediate_files['comb_weight_source']
        gwater_upslope_source = output_path + output_files['upslope_source']

        logger.info("Processing Groundwater Recharge/Baseflow objective...")

        ## LULC Index-R
        # Make Cover and Roughness Index rasters
        if not os.path.exists(gwater_index_cover):
            # if flood mitigation done, reuse data
            if objective['Flood Mitigation']['found']:
                gwater_index_cover = flood_index_cover
            else:
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, cover_rios_field,
                                        gwater_index_cover)

        if not os.path.exists(gwater_index_rough):
            # if flood mitigation done, reuse data
            if objective['Flood Mitigation']['found']:
                gwater_index_rough = flood_index_rough
            else:
                derive_raster_from_lulc(lulc_raster_uri, lucode_field,
                                        lulc_coeff_df, roughness_rios_field,
                                        gwater_index_rough)

        ## Slope index - binned, not normalized,
        # use Flood's slope index if it was created
        logger.debug("\tCreating slope index...")

        if not os.path.exists(gwater_slope_index):
            if objective['Flood Mitigation']['found']:
                gwater_slope_index = flood_slope_index
            else:
                value_bounds = [5.0, 10.0]
                replacement_index = [0.33, 0.66, 1.0]
                raster_value_to_index(slope_raster_uri, gwater_slope_index,
                                      value_bounds, replacement_index)

        logger.info("\t... created Groundwater/Baseflow slope index: "
                    + basename(gwater_slope_index))

        ## Downslope retention Index
        # Since these are same as Flood mitigation, reuse those rasters if they've been done
        logger.debug("\tCreating downslope retention index...")
        # Combined weight R
        if not os.path.exists(gwater_comb_weight_R):
            if objective['Flood Mitigation']['found']:
                gwater_comb_weight_R = flood_comb_weight_R
            else:
                average_raster(raster_uri_list=[gwater_index_rough],
                               inverseraster_uri_list=[gwater_slope_index],
                               output_raster_uri=gwater_comb_weight_R)

        # Downslope retention index
        if not os.path.exists(gwater_dret_flowlen):
            if objective['Flood Mitigation']['found']:
                gwater_dret_flowlen = flood_dret_flowlen
                gwater_dret_index = flood_dret_index
            else:
                pygrout.routing.distance_to_stream(flow_dir_raster_uri,
                                                   streams_raster_uri,
                                                   gwater_dret_flowlen,
                                                   factor_uri=gwater_comb_weight_R)
                normalize(gwater_dret_flowlen, gwater_dret_index)
        logger.info("\t... created Groundwater/Baseflow downslope retention index: "
                    + basename(gwater_dret_index))

        logger.debug("\tCreating upslope source...")

        # Annual average precipitation index
        if not os.path.exists(gwater_precip_annual_index):
            normalize(precip_annual_raster_uri, gwater_precip_annual_index)

        # Actual Evapotranspiration (AET) index
        if not os.path.exists(gwater_aet_index):
            normalize(aet_raster_uri, gwater_aet_index)

        # Combined weight source
        if not os.path.exists(gwater_comb_weight_source):
            average_raster(raster_uri_list=[gwater_precip_annual_index,
                                            soil_texture_raster_uri,
                                            gwater_slope_index,
                                            soil_depth_index_uri],
                           inverseraster_uri_list=[gwater_aet_index,
                                                   gwater_index_cover,
                                                   gwater_index_rough],
                           output_raster_uri=gwater_comb_weight_source)

        ## Upslope source
        if not os.path.exists(gwater_upslope_source):
            weighted_flow_accumulation(flow_dir_raster_uri,
                                       dem_raster_uri,
                                       gwater_upslope_source,
                                       source_weight_uri=gwater_comb_weight_source)

        logger.info("\t... created Groundwater/Baseflow upslope source: "
                           + basename(gwater_upslope_source))

    ############################################################################
    ### Write input parameters to an output file for user reference
    try:
        parameterfile_uri = output_path \
                            + "RIOS_Pre_Processing_" \
                            + time.strftime("%Y-%m-%d-%H-%M") \
                            + suffix + ".txt"
        with open(parameterfile_uri, "w") as parafile:
            parafile.writelines("RIOS PRE-PROCESSING PARAMETERS\n")
            parafile.writelines("______________________________\n\n")
            for para in parameter_log:
                parafile.writelines(para + "\n")
    except:
        logger.error("Error creating parameter file")

    logger.warning("!!!!! NOT CLEANING UP TEMPORARY FILES !!!!!")
