# Coordinate import from shapefiles.
# This file is part of the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2021 Deutsches GeoForschungsZentrum Potsdam
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

import numpy as np

# See if we can import the shapefile module:
try:
	import shapefile
	_has_shapefile = True
except:
	_has_shapefile = False



def points_from_shapefile(filename, method='centroids'):
	"""
	Read points from a shapefile into numpy arrays. This can
	be used to consequently optimize a Laborde oblique Mercator
	for the points in the shapefile.

	Arguments:
	   filename : String pointing to the path of the shapefile.

	Keyword arguments:
	   method   : One of 'centroids' and 'points'.
	              'centroids' : For all geometries in the
	                  shapefile, compute and use only the centroid.
	                  This is suitable for large data sets.
	              'points'    : Use all points of all geometries.
	"""
	# Make sure the library is loaded:
	if not _has_shapefile:
		raise ImportError("Need pyshp to read shapefiles!")

	# Read shapefile:
	shape = shapefile.Reader(filename)

	shapes = shape.shapes()
	# Extract points:
	if method == 'centroids':
		# Compute centroids of geometries (so far only averaging lon/lat).
		# This method is suitable for large data sets.
		lon, lat = np.array([np.mean(s.points,axis=0) for s in shapes]).T

	elif method == 'points':
		# Collect all points.
		lon, lat = np.array([p for s in shapes for p in s.points]).T

	return lon, lat
