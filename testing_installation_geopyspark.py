# -*- coding: utf-8 -*-
"""
Spyder Editor.
"""


import geopyspark as gps

from pyspark import SparkContext
from shapely.geometry import box


# Create the SparkContext
conf = gps.geopyspark_conf(appName="geopyspark-example", master="local[*]")
#sc = SparkContext(conf=conf) #already exists changebelow

sc = SparkContext.getOrCreate(conf)
sqlContext = SQLContext(sc)


# Read in the NLCD tif that has been saved locally.
# This tif represents the state of Pennsylvania.
raster_layer = gps.geotiff.get(layer_type=gps.LayerType.SPATIAL,
                               uri='/tmp/NLCD2011_LC_Pennsylvania.tif',
                               num_partitions=100)

# Tile the rasters within the layer and reproject them to Web Mercator.
tiled_layer = raster_layer.tile_to_layout(layout=gps.GlobalLayout(), target_crs=3857)

# Creates a Polygon that covers roughly the north-west section of Philadelphia.
# This is the region that will be masked.
area_of_interest = box(-75.229225, 40.003686, -75.107345, 40.084375)

# Mask the tiles within the layer with the area of interest
masked = tiled_layer.mask(geometries=area_of_interest)

# We will now pyramid the masked TiledRasterLayer so that we can use it in a TMS server later.
pyramided_mask = masked.pyramid()

# Save each layer of the pyramid locally so that it can be accessed at a later time.
for pyramid in pyramided_mask.levels.values():
    gps.write(uri='file:///tmp/pa-nlcd-2011',
              layer_name='north-west-philly',
              tiled_raster_layer=pyramid)