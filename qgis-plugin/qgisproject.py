# Project coordinate arrays between coordinate systems using the QGIS facilities
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2020 Deutsches GeoForschungsZentrum Potsdam
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from qgis.core import *

def lswkb2array(wkb):
	"""
	This method converts a LineString WKB to a numpy array of
	coordinates.
	"""
	# First determine Endianess:
	endian = 'little' if wkb[0] == 1 else 'big'

	# Assert that the WKB is a LineString:
	if not int.from_bytes(wkb[1:5],endian) == 2:
		raise TypeError("lswkb2array only works for LineString WKB.")

	# Number of points:
	N = int.from_bytes(wkb[5:9], endian)
	if 2*8*N != len(wkb) - 9:
		raise RuntimeError("Malformed WKB.")

	# Read array:
	lola = np.frombuffer(wkb[9:], dtype=np.double).reshape((-1,2))

	return lola


def project(x, y, src_crs, dst_crs):
	"""
	Projects coordinates from one CRS to another.
	"""
	# Use the rather fast line string bulk creation interface:
	ls = QgsLineString(x,y)

	# Create the transformation:
	transform = QgsCoordinateTransform(src_crs, dst_crs,
									   QgsProject.instance())

	# Transform the geometry:
	success = ls.transform(transform)

	# Obtain coordinates:
	bytes = ls.asWkb().data()
	lola = lswkb2array(bytes)

	lon = lola[:,0].reshape(-1)
	lat = lola[:,1].reshape(-1)

	return lon, lat
