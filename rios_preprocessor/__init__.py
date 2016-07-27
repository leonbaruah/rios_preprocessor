__version__ = '0.1.7'
import types as _types
import rios_preprocessor
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
from rios_preprocessor import (average_raster, 
                               calculate_riparian_index,
                               change_geotiff_nodata_value, 
                               define_channels, 
                               derive_raster_from_lulc, 
                               get_end_pixels_of_river_raster,
                               get_input_data, 
                               get_input_data_param_dictionary, 
                               get_input_data_to_objective, 
                               get_intermediate_objective_suffix,
                               get_neighbouring_pixels, 
                               get_objective_dictionary, 
                               get_objective_todo, 
                               get_objectives_list, 
                               get_output_objective_suffix, 
                               get_pixels_within_radius,
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
