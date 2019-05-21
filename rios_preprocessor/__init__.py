__version__ = '0.2.2'
import types as _types
import rios_preprocessor
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
from rios_preprocessor import (average_raster,
                               burn_stream_into_flowdir_channels,
                               calculate_downslope_retention,
                               calculate_downslope_retention_index,
                               calculate_riparian_index,
                               calculate_slope_index,
                               calculate_upslope_source,
                               change_geotiff_nodata_value,
                               check_streams_raster_sourcedata,
                               create_hydro_layers,
                               define_channels, 
                               derive_raster_from_lulc,
                               get_end_pixels_of_river_raster,
                               get_input_data, 
                               get_input_data_param_dictionary, 
                               get_input_data_to_objective,
                               get_intermediate_file,
                               get_intermediate_objective_suffix,
                               get_neighbouring_pixels,
                               get_normalisation_factor_from_file,
                               get_objective_df,
                               get_objective_dictionary, 
                               get_objective_todo, 
                               get_objectives_list,
                               get_objective_df,
                               get_output_objective_suffix, 
                               get_pixels_within_radius, 
                               get_rios_coefficient_fieldnames,
                               hydro_naming_convention,
                               is_projection_consistent,
                               label_river_bank_buffers, 
                               label_river_banks, 
                               label_streams, 
                               main, 
                               map_coefficients, 
                               map_pixels_next_to_river, 
                               normalize, 
                               normalize_array, 
                               optimize_threshold_flowacc, 
                               pixel_neighbours_from_east,
                               process_erosion_control,
                               process_flood_mitigation,
                               process_groundwater_recharge,
                               process_nitrogen_retention,
                               process_phosphorus_retention,
                               raster_value_to_index,
                               weighted_flow_accumulation)

__all__ = []
for _attrname in dir(rios_preprocessor):
    if type(getattr(rios_preprocessor, _attrname)) is _types.FunctionType:
        __all__.append(_attrname)

import logging
logging.basicConfig()

del _attrname
del _types