# Convenience class for an optimized Laborde oblique Mercator projection.
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
from .defs import _ellipsoids

# Optional dependency on pyproj.
# This dependency is problematic: In at least one configuration,
# due to the multiple sources of installation pathways, importing
# the pyproj package from within QGIS throws a std::bad_alloc.
# Hence, check whether we are calling from QGIS and, in that case,
# default to _has_pyproj=False. The check can be performed by trying
# to import from the namespace that QGIS provides to the plugins:
try:
	from qgis.core import Qgis # type: ignore
	_is_qgis = True
except:
	_is_qgis = False

if _is_qgis:
	_has_pyproj = False
else:
	try:
		from pyproj import Proj
		_has_pyproj = True
	except:
		_has_pyproj = False



# Some helper functions:
def _tfun(phi0,e):
	# Method to calculate t or t0 from Snyder (1987):
	e_sin_phi = e*np.sin(phi0)
	return np.tan(0.5*(0.5*np.pi - phi0)) / \
		   np.power((1.0 - e_sin_phi) / (1.0 + e_sin_phi),0.5*e)

def _arcsin(x):
	# An arcsin fixing numerical instability under the assumption
	# that |x|>1 can only happen by numerical instability, not
	# due to wrong math.
	if abs(x) <= 1.0:
		return np.arcsin(x)
	elif x > 1.0:
		return 0.5*np.pi
	else:
		return -0.5*np.pi

def _preprocess_coordinates(x0,x1, cname0, cname1):
	# Make sure we can apply all the numpy routines and ndarray methods:
	x0 = np.atleast_1d(x0)
	x1 = np.atleast_1d(x1)

	# Handle incompatible shapes:
	if not np.array_equal(x0.shape,x1.shape):
		if x0.size != 1 and x1.size != 1:
			raise RuntimeError("Incompatible shapes of %s (%s) and %s (%s)!"
			                   % (cname0, str(x0.shape), cname1, str(x1.shape)))
		elif x0.size == 1:
			x0 = x0 * np.ones_like(x1)
		else:
			x1 = x1 * np.ones_like(x0)

	# Work on one-dimensional arrays:
	shape = list(x0.shape)
	if x0.squeeze().ndim != 1:
		x0 = x0.flatten()
		x1 = x1.flatten()

	return x0, x1, shape





# Class definition:

class LOMBase:
	"""
	Base class for LOM.
	"""
	def __init__(self, ellipsoid='WGS84', projection_tolerance=1e-7, f=None,
	             a=None, lonc=None, lat_0=None, alpha=None, k0=None,
	             _invert_v=False, _invert_u=False):
		# Initialization.
		# 1) Sanity checks:

		self._alpha = alpha
		self._lonc = lonc
		self._lat_0 = lat_0
		self._k0 = k0
		self._f = f
		self._a = a

		self._ellipsoid = ellipsoid
		self._invert_v = _invert_v
		self._invert_u = _invert_u

		self._projection_tolerance = projection_tolerance

		# 3) Initialize projection, if available:
		if _has_pyproj:
			self._proj = Proj(self.proj4_string())
		else:
			self._proj = None

		# 4) Also prepare for applying the fallback implementation
		#    (that also returns the local scale factor)
		self._check_and_set_constants()


	def _project_to_uvk(self, lon, lat, return_k=False):
		"""
		Project lon/lat to x/y using a Hotine oblique Mercator
		projection. Follows alternte B from Snyder (1987), which
		is EPSG standard 9815.
		Projection center lies at (0,0) in projected coordinates.

		Required parameters:
		   lon          : Array of longitude coordinates to convert
		   lat          : Array of latitude coordinates to convert

		Optional parameters:
		   return_k : Whether to calculate and return the scale factor
			          k. Default: False

		Returns:
		   If return_k == False:
			  x,y
		   otherwise:
			  x,y,k

		Reference:
		Snyder, J. P. (1987). Map projections: A working manual.
		U.S. Geological Survey Professional Paper (1395).
		doi: 10.3133/pp1396
		"""
		lon, lat, shape = _preprocess_coordinates(lon, lat, "lon", "lat")

		# Unravel constants:
		e,phi0,lambda_c,alpha_c,A,B,t0,D,E,F,G,gamma0,lambda0,uc = self._constants
		tol = self._projection_tolerance

		# Degree to radians:
		phi = np.deg2rad(lat)
		lbda = np.deg2rad(lon)

		# (9-25) to (9-30), respecting note:
		mask = np.logical_and(lat < 90.0-tol,lat > -90.0+tol)
		t = _tfun(phi[mask],e)
		Q = E / np.power(t,B) # Corresponds to W in PROJ
		iQ = 1.0/Q
		S = 0.5*(Q - iQ)
		T = 0.5*(Q + iQ)

		# Note about handling longitude wrapping:
		dlambda = lbda -lambda0
		dlambda[dlambda < -np.pi] += 2*np.pi
		dlambda[dlambda > np.pi] -= 2*np.pi
		V = np.sin(B*dlambda[mask])
		cosgamma0 = np.cos(gamma0)
		singamma0 = np.sin(gamma0)
		U = (-V * cosgamma0 + S*singamma0) / T

		# Calculate v
		v = np.zeros_like(phi)
		v[mask] = A * np.log((1.0-U) / (1.0+U)) / (2.0*B)
		v[~mask] = (A/B) * np.log(np.tan(0.25*np.pi + 0.5 * np.sign(phi[~mask])
		                                                  * gamma0))

		# Invert:
		if self._invert_v:
			v = -v

		# Calculate u:
		u = np.zeros_like(phi)
		W = np.cos(B*dlambda) # Called 'temp' in PROJ
		mask2 = np.abs(W) > tol
		mask3 = mask2[mask]
		mask4 = np.logical_and(mask,mask2)
		u[mask4] = A/B * np.arctan2((S[mask3]*cosgamma0 + V[mask3]*singamma0),
			                        W[mask4])
		# The following line is different from what is given by
		# Snyder (1987). Following him, the line would have to be
		# u[~mask2] = A*B*dlambda[~mask2]
		# Instead, we follow the PROJ issue by iskander,
		# https://github.com/OSGeo/PROJ/issues/114,
		# which instead writes
		u[~mask2] = A*dlambda[~mask2]
		u[~mask] = A*phi[~mask]/B

		# Return coordinates:
		if not return_k:
			u -= uc
			if self._invert_u:
				return -u.reshape(shape), v.reshape(shape)
			return u.reshape(shape), v.reshape(shape)

		# Calculate k, the scale factor:
		k = A * np.cos(B*u/A) * np.sqrt(1-e**2 * np.sin(phi)**2) / \
			(self._a * np.cos(phi) * np.cos(B*dlambda))
		u -= uc

		# Circumvent numerical problems:
		k = np.maximum(k,self._k0)

		# Now return coordinates:
		if self._invert_u:
			return -u.reshape(shape), v.reshape(shape), k.reshape(shape)
		return u.reshape(shape), v.reshape(shape), k.reshape(shape)


	def _check_and_set_constants(self):
		"""
		Perform conversions from degree to radians and do some
		consistency checks (from Snyder (1987))!
		"""
		# For readability: Local variables.
		f = self._f
		a = self._a
		lat0 = self._lat_0
		lon0 = self._lonc
		azimuth = self._alpha
		tol = self._projection_tolerance

		# Convert flattening to eccentricity:
		e = np.sqrt(2*f - f**2)

		# Convert to radians.
		# Follow notation from Snyder (1987)
		phi0 = np.deg2rad(lat0)
		lambda_c = np.deg2rad(lon0)
		alpha_c  = np.deg2rad(azimuth)

		# Works for -90 < azimuth_c < 90, so transform to that range:
		if azimuth > 90.0:
			alpha_c -= np.pi
		if azimuth < -90.0:
			alpha_c += np.pi

		# Calculate the constants A,B,D,E,F,G,gamma0, and lambda0.

		# Calculate intermediate values from Snyder (1987):
		cosphi0 = np.cos(phi0)
		sinphi0 = np.sin(phi0)
		B = np.sqrt(1.0 + e**2 * cosphi0**4 / (1.0 - e**2))
		A = a * B * self._k0 * np.sqrt(1.0 - e**2) \
		    / (1.0 - e**2 * sinphi0**2)

		# Lambda to calculate t and t0:
		t0 = _tfun(phi0,e)
		D = B * np.sqrt(1.0 - e**2) / (cosphi0 * np.sqrt(1.0 - e**2 * sinphi0**2))

		# Ensure D >= 1:
		if abs(D) < 1.0:
			D = np.sign(D)

		F = D + np.sign(phi0) * np.sqrt(D**2 - 1.0)
		E = F * np.power(t0,B)
		G = 0.5 * (F - 1.0/F)
		gamma0 = _arcsin(np.sin(alpha_c) / D)
		lambda0 = lambda_c - _arcsin(G * np.tan(gamma0)) / B

		# Compute offset of center in u:
		uc = np.sign(phi0) * A/B * np.arctan2(np.sqrt(D**2-1), np.cos(alpha_c))

		# Save constants:
		self._constants = (e, phi0, lambda_c, alpha_c, A, B, t0, D, E, F, G,
		                   gamma0, lambda0, uc)
