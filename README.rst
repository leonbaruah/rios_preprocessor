Homepages
---------

Visit the ``Landmark Information Group`` homepage here:
`landmark.co.uk <http://www.landmark.co.uk/>`__.
``Landmark Ecoservices`` page coming soon! Visit the home of ``RIOS`` on
the web:
`naturalcapitalproject.org <http://www.naturalcapitalproject.org/software/#rios>`__

Discuss
-------

Looking for a place to ask questions about RIOS? Check out the Natural
Capital Project forums!

Documentation
-------------

``rios_preprocessor`` is meant as a replacement for the ArcGIS python
code that currently produces the inputs for the main RIOS program.

``rios_preprocessor`` produces rasters of:

-  Downslope retention index
-  Upslope source
-  Riparian index
-  Slope index

that are used as input for the following RIOS objectives:

-  Erosion Control
-  Nitrogen Retention
-  Phosphorus
-  Flood Mitigation
-  Groundwater Recharge/Baseflow

Along the way, there are some additional tricks ``rios_preprocessor``
uses that may be useful in other contexts:

-  rasterizes an input shapefile river network and matches it to a
   (raster) DEM/DTM
-  identifies discontinuous river banks along a rasterized river
-  identifies the end points of rasterized rivers

Dependencies
------------

-  numpy
-  pandas
-  geopandas
-  fiona
-  rasterio
-  shapely
-  pygeoprocessing

Installation
~~~~~~~~~~~~

``pip install rios_preprocessor``

Contact Details
---------------

Questions about ``RIOS`` are picked up at the Natural Capital Project
forums.

If you have any questions or comments about ``rios_preprocessor``
specifically, you can contact me at leon.baruah@landmark.co.uk.

Changelog
=========

Version 0.1.0
-------------

-  Hello World!
