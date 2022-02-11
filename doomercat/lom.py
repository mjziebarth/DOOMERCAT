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
from math import atan2, degrees
from .cppextensions import bfgs_optimize
from .lombase import LOMBase, _has_pyproj
from .defs import _ellipsoids
from .initial import initial_parameters


class LabordeObliqueMercator(LOMBase):
	"""
	A Laborde oblique Mercator projection (LOM) optimized for a
	geographical data set. The projection's definition follows
	Laborde (1928) as laid out by Roggero (2009) and is compatible with the
	Hotine oblique Mercator projection (EPSG Guidance Notes 7-2, (2019);
	Snyder (1987)).

	Call signatures:

	(1) Optimize the LOM for a set of points:

	LabordeObliqueMercator(lon, lat, weight=None, pnorm=2, k0_ap=0.98,
	                       sigma_k0=0.02, ellipsoid='WGS84', f=None, a=None,
	                       Nmax=200)

	   lon, lat       : Iterable sets of longitude and latitude coordinates
	                    of the data set.
	   weight         : Optional iterable set of multiplicative weights assigned
	                    to the data points in the cost function. Need to be
	                    positive real weights.
	                    Default: None
	   pnorm          : Power by which residuals are weighted in the cost
	                    function that is minimized. Let dx_i be the i'th
	                    residual between projected and ellipsoidal space.
	                    Then the sum
	                       sum(dx_i ** pnorm)
	                    is minimized to obtain an optimal projection.
	                    Default: 2
	   k0_ap          : The k_0 threshold beyond which smaller k_0
	                    are constrained using a quadratic potential.
	                    This constraint helps preventing the solver
	                    from approaching small-circle solutions.
	                    Setting k0_ap to zero removes the constraint.
	                    Default: 0.95
	   sigma_k0       : The scale of the quadratic constrains for
	                    small k_0, i.e. the standard deviation of
	                    the quadratic branch of the k_0 potential.
	                    Default: 0.02
	   ellipsoid      : Name of the reference ellipsoid. Must be one of
	                    'WGS84' and 'GRS80'. Can be overriden by using the
	                    a and f parameters.
	                    Default: 'WGS84'
	   f              : If not None, the flattening of the reference rotational
	                    ellipsoid.
	                    Default: None
	   a              : If not None, the large axis of of the reference
	                    rotational ellipsoid. Only used for projection,
	                    irrelevant for the optimization.
	                    Default: None
	   Nmax           : Maximum number of iterations of the BFGS algorithm.
	                    Default: 200
	   cyl_lon0       : Initial longitude of the cylinder axis when starting
	                    the optimization.
	                    Default: 0.0
	   cyl_lat0       : Initial latitude of the cylinder axis.
	                    Default: 10.0


	(2) Give the parameters of the LOM to allow projecting:

	LabordeObliqueMercator(lonc=lonc, lat_0=lat0, alpha=alpha, k0=k0,
	                       ellipsoid='WGS84', a=None, f=None)
	    lonc   : Longitude of the central point.
	    lat_0  : Latitude of the central point.
	    alpha  : Azimuth of the central line at the central point.
	    k0     : Scale factor at the central point.



	Methods:
	   lonc()
	   lat0()
	   azimuth()
	   k0()
	   ellipsoid()
	   project()
	   inverse()
	   distortion()

	References:
	Laborde, J. (1928). La nouvelle projection du Service Geographique
	de Madagascar. Madagascar, Cahiers du Service geographique de Madagascar,
	Tananarive 1:70

	Roggero, M. (2009). Laborde projection in Madagascar cartography and its
	recovery in WGS84 datum. Appl Geomat 1, 131. doi:10.1007/s12518-009-0010-4

	Snyder, J. P. (1987). Map projections: A working manual.
	U.S. Geological Survey Professional Paper (1395).
	doi: 10.3133/pp1396
	"""
	def __init__(self, lon=None, lat=None, weight=None, pnorm=2, k0_ap=0.98,
	             sigma_k0=0.002, ellipsoid=None, f=None, a=None,
	             cyl_lon0=0.0, cyl_lat0=10.0, lonc=None, lat_0=None, alpha=None,
	             k0=None, Nmax=200, logger=None):
		# Initialization.
		# 1) Sanity checks:
		assert ellipsoid in _ellipsoids or ellipsoid is None
		assert pnorm >= 2
		Nmax = int(Nmax)
		assert Nmax > 0

		# Ellipsoid parameters:
		if ellipsoid is None:
			if f is None and a is None:
				ellipsoid = "WGS84"
			elif f is None or a is None:
				raise RuntimeError("If one of 'a' and 'f' is given, both have "
				                   "to be specified.")
		else:
			if f is not None or a is not None:
				raise RuntimeError("Conflicting definition: Ellipsoid defined "
				                   "both through 'ellipsoid' keyword and 'a' "
				                   "and 'f' keywords.")
		if f is None:
			f = 1.0 / _ellipsoids[ellipsoid][1]
		else:
			f = float(f)
		if a is None:
			a = _ellipsoids[ellipsoid][0]
		else:
			a = float(a)

		# Check whether lon/lat given:
		if lon is not None:
			# Case 1: Data is given as lon/lat, so we optimize the projection.
			assert lat is not None
			if not isinstance(lon,np.ndarray):
				lon = np.array(lon)
			if not isinstance(lat,np.ndarray):
				lat = np.array(lat)
			if weight is not None and not isinstance(weight,np.ndarray):
				weight = np.array(weight)
				assert weight.shape == lon.shape

			assert lon.shape == lat.shape

			# Initial guess for the cylinder axis and central point:
			(cyl_lon0, cyl_lat0), (lonc0, lat_0) = initial_parameters(lon, lat)
			cyl_lon0 = float(cyl_lon0)
			cyl_lat0 = float(cyl_lat0)
			lonc0 = float(lonc0)

			if logger is not None:
				logger.log(20, "Starting BFGS optimization.")

			# Optimize the Laborde oblique Mercator:
			result = \
			    bfgs_optimize(lon, lat, weight, pnorm, k0_ap, sigma_k0, f,
			                  cyl_lon0, cyl_lat0, lonc0, Nmax)
			lonc = result.lonc
			lat_0 = result.lat_0
			alpha = result.alpha
			k0 = result.k0

		else:
			# Case 2: The parameters are given directly.
			assert lat is None
			lonc = float(lonc)
			lat_0 = float(lat_0)
			k0 = float(k0)
			alpha = float(alpha)

		# 3) Call superclass, which implements projection etc.
		super().__init__(ellipsoid=ellipsoid, lonc=lonc, lat_0=lat_0, k0=k0,
		                 alpha=alpha, f=f, a=a)


	def lonc(self):
		"""
		Longitude of the projection's central point.

		Returns:
		   Float
		"""
		return self._lonc


	def lat_0(self):
		"""
		Latitude of the projection's central point.

		Returns:
		   Float
		"""
		return self._lat_0


	def alpha(self):
		"""
		Azimuth of projection's equator at central point (lonc,lat_0).

		Returns:
		   Float
		"""
		return self._alpha

	def k0(self):
		"""
		Scaling of the central line great circle compare to the ellipsoid.

		Returns:
		   Float
		"""
		return self._k0


	def ellipsoid(self):
		"""
		Name of the reference ellipsoid for this projection.

		Returns:
		   String
		"""
		return self._ellipsoid


	def a(self):
		"""
		Length of the large half axis of the reference ellipsoid.

		Returns:
		   Float
		"""
		return self._a


	def f(self):
		"""
		Flattening of the reference ellipsoid.

		Returns:
		   Float
		"""
		return self._f


	def proj4_string(self):
		"""
		Return a projection string for use with PROJ/GDAL.

		Returns:
		   String

		The projection string uses the 'omerc' projection,
		which in turn uses the Hotine oblique Mercator projection
		(HOM) (EPSG 9815) given by Snyder (1987) instead of the
		Laborde oblique Mercator projection (LOM) (EPSG 9813).
		For practical purposes, the differences between the LOM
		optimized by this class and the HOM should not be significant.

		Reference:
		Snyder, J. P. (1987). Map projections: A working manual.
		U.S. Geological Survey Professional Paper (1395).
		doi: 10.3133/pp1396
		"""
		ellps = self.ellipsoid()
		if ellps not in (None,'IERS2003'):
			return "+proj=omerc +lat_0=%.8f +lonc=%.8f +alpha=%.8f +k_0=%.8f " \
			       "+ellps=%s" % (self.lat_0(), self.lonc(),self.alpha(),
				                   self.k0(),self.ellipsoid())
		else:
			return "+proj=omerc +lat_0=%.8f +lonc=%.8f +alpha=%.8f +k_0=%.8f " \
			       "+a=%.8f +b=%.8f" % (self.lat_0(), self.lonc(), self.alpha(),
			                          self.k0(), self.a(),
			                          self.a()*(1.0-self.f()))


	def project(self, lon, lat):
		"""
		Project a geographical coordinate set.

		Returns:
		   x, y : Coordinates in projected coordinate system.

		Note: This method uses the PROJ 'omerc' projection
		which in turn uses the equations given by Snyder (1987)
		for the Hotine oblique Mercator projection (EPSG 9815).
		For most practical purposes, the difference should be
		negligible (EPSG Guidance Notes 7-2, 2019).

		Reference:
		Snyder, J. P. (1987). Map projections: A working manual.
		U.S. Geological Survey Professional Paper (1395).
		doi: 10.3133/pp1396
		"""
		if not _has_pyproj:
			return self._project_to_uvk(lon, lat)
		return self._proj(lon, lat)


	def inverse(self, x, y):
		"""
		Inverse projection.

		Note: This method uses the PROJ 'omerc' projection
		which in turn uses the equations given by Snyder (1987)
		for the Hotine oblique Mercator projection (EPSG 9815).
		For most practical purposes, the difference should be
		negligible (EPSG Guidance Notes 7-2, 2019).

		Reference:
		Snyder, J. P. (1987). Map projections: A working manual.
		U.S. Geological Survey Professional Paper (1395).
		doi: 10.3133/pp1396
		"""
		if not _has_pyproj:
			raise ImportError("Module 'pyproj' is not available!")
		return self._proj(x, y, inverse=True)


	def distortion(self, lon, lat):
		"""
		Calculate distortion of the oblique Mercator projection
		at given geographical coordinates.

		Note: This computation follows the equations given by
		Snyder (1987) which are derived for the Hotine oblique
		Mercator projection. For practical purposes, this can
		often be equivalent to the Laborde oblique Mercator
		(EPSG Guidance Notes 7-2, 2019).
		In any case, the relevant implementation of the oblique
		Mercator projection in PROJ follows the Hotine oblique
		Mercator equations by Snyder (1987). Hence, the distortion
		as returned by this method represents most practical
		use cases.

		Reference:
		Snyder, J. P. (1987). Map projections: A working manual.
		U.S. Geological Survey Professional Paper (1395).
		doi: 10.3133/pp1396
		"""
		return self._project_to_uvk(lon, lat, True)[2]
