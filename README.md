# remove-trend-geotiff
Remove regional trend from potential field data using least squares
fitting of 3D polynomial plane. For example, removing regional trend
from magnetic data. Uses GDAL package.

Written in Python 3.8
Christopher Mancuso
Last updated March 2020
IN:
	GeoTIFF image (works best if square dimensions)
	-filename (see raster_name below)(do not include file extention)
OUT:
	GeoTIFF image
	-filename_trend_removed.tiff
