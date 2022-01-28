# Convenience methods for the Laborde oblique Mercator projection.
# This file is part of the DOOMERCAT python module.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
#          Sebastian von Specht
#
# Copyright (C) 2019-2021 Deutsches GeoForschungsZentrum Potsdam,
#                         Sebastian von Specht
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

def ellipsoid_projection_radian(lon, lat, h, a, f):
	"""
	Project longitude and latitude coordinates to Euclidean reference
	system.

	Parameters:
	   lon, lat: Geographic coordinates in radians.
	   h :       Height above ellipsoid.
	"""
	coslon = np.cos(lon)
	sinlon = np.sin(lon)
	coslat = np.cos(lat)
	sinlat = np.sin(lat)

	b = (1.0 - f)*a
	CF = a**2 / np.sqrt(a**2 * coslat**2 + b**2 * sinlat**2)

	xyz = np.zeros((lon.size,3))
	xyz[:,0] = (CF + h) * coslat * coslon
	xyz[:,1] = (CF + h) * coslat * sinlon
	xyz[:,2] = (b**2/a**2 * CF + h) * sinlat

	return xyz


def ellipsoid_projection(lon, lat, h, a, f):
	"""
	Project longitude and latitude coordinates to Euclidean reference
	system.

	Parameters:
	   lon, lat: Geographic coordinates in degrees.
	   h :       Height above ellipsoid.
	"""
	return ellipsoid_projection_radian(np.deg2rad(lon), np.deg2rad(lat),
	                                   h, a, f)


def laborde_variables(lambda_, phi, vize, lambda_c, f):
	"""
	Angles in radians
	"""

	# Ellipsoid:
	e2 = f*(2-f)
	e = np.sqrt(e2)

	Gd_inv = lambda x : np.arctanh(np.sin(x))

	B = np.sqrt(1.0 + e2*np.cos(vize)**4/(1-e2))
	fies = np.arcsin(np.sin(vize)/B)
	C = Gd_inv(fies) - B*(Gd_inv(vize) + 0.5*e*np.log((1-e*np.sin(vize)) / (1+e*np.sin(vize))))
	q = C + B*(Gd_inv(phi) + 0.5*e*np.log((1-e*np.sin(phi)) / (1+e*np.sin(phi))))
	P = 2*np.arctan(np.exp(q)) - 0.5*np.pi

	sin_vize = np.sin(vize)
	L = B * (lambda_ - lambda_c)

	cosP = np.cos(P)

	return e, B, fies, C, q, P, sin_vize, L, cosP


def laborde_sub(P, cosP, L):
	"""
	Perform the U,V,W projection given the P and L variables.
	"""
	U = cosP * np.cos(L)
	V = cosP * np.sin(L)
	W = np.sin(P)

	return U, V, W


def laborde_projection(lambda_, phi, vize, lambda_c, f):
	"""
	Compute Laborde coordinates (and intermediates) from
	geographic coordinates.

	lon, lat : Coordinates in degrees.
	"""
	# Compute the Laborde projection variables fro
	e, B, fies, C, q, P, sin_vize, L, cosP = laborde_variables(lambda_, phi, vize, lambda_c, f)

	return laborde_sub(P, cosP, L)
