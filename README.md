Homepages
---------

Visit the  `Viridian Logic` homepage here: [viridianlogic.com](https://viridianlogic.com).
Visit the home of `RIOS` on the web: [naturalcapitalproject.org](https://www.naturalcapitalproject.org/software/#rios)

Discuss
-------

Talk to us at Viridian Logic through [our website](http://viridianlogic.com/#contactus)
Looking for a place to ask questions about RIOS? Check out the <a href="http://forums.naturalcapitalproject.org/index.php?p=/categories/rios">Natural Capital Project forums</a>!
 
Documentation
-------------

`rios_preprocessor` is meant as a platform-independent replacement for the ArcGIS python code that currently
produces the inputs for the main RIOS program.

`rios_preprocessor` produces rasters of:

* Downslope retention index
* Upslope source
* Riparian index
* Slope index

that are used as input for the following RIOS objectives:

* Erosion Control
* Nitrogen Retention
* Phosphorus
* Flood Mitigation
* Groundwater Recharge/Baseflow

Along the way, there are some additional tricks `rios_preprocessor` uses that may be useful in other contexts:

* rasterizes an input shapefile river network and matches it to a (raster) DEM/DTM
* identifies discontinuous river banks along a rasterized river
* identifies the end points of rasterized rivers

## Dependencies

* numpy
* pandas
* geopandas
* fiona
* rasterio > 0.36.0
* shapely
* pygeoprocessing ~= 0.3.3

### Installation

`pip install rios_preprocessor`

## Contact Details

Questions about `RIOS` are picked up at the <a href="http://forums.naturalcapitalproject.org/index.php?p=/categories/rios">Natural Capital Project forums</a>.

If you have any questions or comments about `rios_preprocessor` specifically, you can contact me through the [Viridian Logic website](http://viridianlogic.com/#contactus).

# Changelog

## Version 0.2.0

* Refactored and modularised code for increased usability
* Raster projection checks added
* Set deprecation warning for get_objectives_list()
* Minor bug fixes

## Version 0.1.9

* Fixed upslope source and riparian index getting affected by integer bug

## Version 0.1.8

* Deprecated GDAL style transforms in line with rasterio > 0.36
* Set upper limit on pygeoprocessing to retain flow algorithms
* Minor bug fixes

## Version 0.1.7

* Minor bug fixes

## Version 0.1.6

* Implemented logging
* PEP8 fixes

## Version 0.1.5

* Python 3 style print statement fixing
* Cleaned up documentation

