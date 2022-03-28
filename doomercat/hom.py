# Convenience class for an optimized Hotine oblique Mercator projection.
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
from math import atan2, degrees, isinf
from .defs import _ellipsoids
from .initial import initial_parameters
from .enclosingsphere import BoundingSphere
from .hotineproject import hotine_project
from .hotine import grad


class HotineObliqueMercator:
    """
    A Hotine oblique Mercator projection (HOM) optimized for a
    geographical data set. The projection's definition follows
    Snyder (1987).

    Call signatures:

    (1) Optimize the HOM for a set of points:

    HotineObliqueMercator(lon, lat, weight=None, pnorm=2, k0_ap=0.98,
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

    HotineObliqueMercator(lonc=lonc, lat_0=lat0, alpha=alpha, k0=k0,
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

    Note: This computation follows the equations given by
    Snyder (1987) which are derived for the Hotine oblique
    Mercator projection. For practical purposes, this can
    often be equivalent to the Laborde oblique Mercator
    (EPSG Guidance Notes 7-2, 2019; Laborde, 1928; Roggero, 2009).
    In any case, the relevant implementation of the oblique
    Mercator projection in PROJ follows the Hotine oblique
    Mercator equations by Snyder (1987). Hence, the distortion
    as returned by this method represents most practical
    use cases.

    References:
    Snyder, J. P. (1987). Map projections: A working manual.
    U.S. Geological Survey Professional Paper (1395).
    doi: 10.3133/pp1396

    Laborde, J. (1928). La nouvelle projection du Service Geographique
    de Madagascar. Madagascar, Cahiers du Service geographique de Madagascar,
    Tananarive 1:70

    Roggero, M. (2009). Laborde projection in Madagascar cartography and its
    recovery in WGS84 datum. Appl Geomat 1, 131. doi:10.1007/s12518-009-0010-4
    """
    def __init__(self, lon=None, lat=None, weight=None, pnorm=2, k0_ap=0.98,
                 sigma_k0=0.002, ellipsoid=None, f=None, a=None,
                 lonc0=None, lat_00=None, alpha0=None, k00=None, lonc=None,
                 lat_0=None, alpha=None, k0=None, Nmax=1000,
                 proot=False, logger=None,
                 backend='C++', fisher_bingham_use_weight=False,
                 compute_enclosing_sphere=False, bfgs_epsilon=1e-3):
        # Initialization.
        # 1) Sanity checks:
        assert ellipsoid in _ellipsoids or ellipsoid is None
        assert pnorm > 0
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

        self._backend_loaded = False

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
                weight /= weight.sum()
                assert weight.shape == lon.shape

            assert lon.shape == lat.shape

            # Initial guess for the parameters:
            if any(p is None for p in (lonc0, lat_00, alpha0, k00)):
                if fisher_bingham_use_weight:
                    w_initial = weight
                else:
                    w_initial = None
                lonc0, lat_00, alpha0, k00 = initial_parameters(lon, lat,
                                                                w_initial,
                                                                pnorm, f)

            if backend in ('c++','C++'):
                # Call the C++ BFGS backend.
                if logger is not None:
                    logger.log(20, "Starting BFGS optimization.")

                # Load the C++ backend:
                self._load_backend()

                # If p=inf, pre-optimize with p=80 norm:
                if isinf(pnorm):
                    pre_res = \
                        self._bfgs_hotine(lon, lat, weight, 80, k0_ap,
                                          sigma_k0, f, lonc0, lat_00, alpha0,
                                          k00, Nmax, proot,
                                          epsilon=bfgs_epsilon)
                    lonc0  = pre_res.lonc
                    lat_00 = pre_res.lat_0
                    alpha0 = pre_res.alpha
                    k00    = pre_res.k0

                # Optimize the Hotine oblique Mercator:
                result = \
                    self._bfgs_hotine(lon, lat, weight, pnorm, k0_ap,
                                      sigma_k0, f, lonc0, lat_00, alpha0,
                                      k00, Nmax, proot, epsilon=bfgs_epsilon)

            elif backend in ('python','Python'):
                # Call the Python Levenberg-Marquardt backend.
                if logger is not None:
                    logger.log(20, "Starting Levenberg-Marquardt "
                                   "optimization.")
                if weight is None:
                    weight = np.ones_like(lon)
                if isinf(pnorm):
                    k00 = initial_parameters(lon, lat, w_initial, 2, f)[3]
                result = grad(np.deg2rad(lon), np.deg2rad(lat),
                              weight, np.deg2rad(lat_00), np.deg2rad(lonc0),
                              np.deg2rad(alpha0), k00, f,
                              0 if isinf(pnorm) else pnorm, Nmax,
                              False, k0_ap, sigma_k0)
            else:
                raise ValueError("Backend unkown!")

            lonc = result.lonc
            lat_0 = result.lat_0
            alpha = result.alpha
            k0 = result.k0
            self.optimization_result = result

            # Save the central point in terms of the enclosing
            # sphere:
            if compute_enclosing_sphere:
                bscenter = BoundingSphere(lon, lat, a, f)
            else:
                bscenter = None

        else:
            # Case 2: The parameters are given directly.
            assert lat is None
            lonc = float(lonc)
            lat_0 = float(lat_0)
            k0 = float(k0)
            alpha = float(alpha)
            bscenter = None


        # Save all attributes:
        self._alpha = alpha
        self._lonc = lonc
        self._lat_0 = lat_0
        self._k0 = k0
        self._f = f
        self._a = a
        self._ellipsoid = ellipsoid
        self._bscenter = bscenter


    def __getstate__(self):
        return self._alpha, self._lonc, self._lat_0, self._k0, self._f, \
               self._a, self._ellipsoid, self._bscenter


    def __setstate__(self, state):
        print("__setstate__ tuple ",len(state))
        self._alpha, self._lonc, self._lat_0, self._k0, self._f, self._a, \
           self._ellipsoid, self._bscenter = state
        self._backend_loaded = False


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


    def proj4_string(self, orient_north=None):
        """
        Return a projection string for use with PROJ/GDAL.

        Returns:
           String

        The projection string uses the 'omerc' projection.

        Reference:
        Snyder, J. P. (1987). Map projections: A working manual.
        U.S. Geological Survey Professional Paper (1395).
        doi: 10.3133/pp1396
        """
        ellps = self.ellipsoid()

        if ellps not in (None,'IERS2003'):
            projstr= "+proj=omerc +lat_0=%.8f +lonc=%.8f +alpha=%.8f " \
                     "+k_0=%.8f +ellps=%s" % (self.lat_0(), self.lonc(),
                                              self.alpha(), self.k0(),
                                              self.ellipsoid())
        else:
            projstr = "+proj=omerc +lat_0=%.8f +lonc=%.8f +alpha=%.8f " \
                      "+k_0=%.8f +a=%.8f +b=%.8f" % (self.lat_0(),
                                              self.lonc(), self.alpha(),
                                              self.k0(), self.a(),
                                              self.a()*(1.0-self.f()))

        # Handle computation of gamma:
        if orient_north is not None:
            gamma = self.north_gamma(*orient_north)
            projstr += (" +gamma=%.8f" % (gamma,))

        return projstr


    def project(self, lon, lat):
        """
        Project a geographical coordinate set.

        Returns:
           x, y : Coordinates in projected coordinate system.

        Reference:
        Snyder, J. P. (1987). Map projections: A working manual.
        U.S. Geological Survey Professional Paper (1395).
        doi: 10.3133/pp1396
        """
        self._load_backend()
        return self._project_hotine(lon, lat, self._lonc, self._lat_0,
                                    self._alpha, self._k0, self._f)


    def north_azimuth(self, lon: float, lat: float,
                      delta: float = 1e-7) -> float:
        """
        Compute, at a location, the local 'north azimuth', that is,
        the clockwise angle from the y axis to the local north vector.

        Returns:
           azimuth : The local azimuth.
        """
        # Compute the local vectors in north direction:
        lon = float(lon)
        lat = float(lat)
        u,v = hotine_project(np.array((lon,lon)),
                             np.array((lat-delta, lat+delta)),
                             self._lonc, self._lat_0, self._alpha,
                             self._k0, self._f)
        dx = u[1] - u[0]
        dy = v[1] - v[0]

        # From this, compute the angle:
        azimuth = degrees(atan2(dx,dy))

        return azimuth


    def north_gamma(self, lon: float, lat: float) -> float:
        """
        Compute the oblique Mercator rectification gamma of Snyder (1987)
        so that the projected map points northwards in positive y direction
        ("up") at the location specified by `lon` and `lat`.
        """
        # Compute the north azimuth in u,v coordinates:
        angle: float = self.north_azimuth(lon, lat)

        # Now take into account that the rotation spelled out by Snyder (1987)
        # actually perform a 90° rotation from u->y and v->x.
        return angle - 90.0


    def distortion(self, lon, lat):
        """
        Calculate distortion of the oblique Mercator projection
        at given geographical coordinates.

        Reference:
        Snyder, J. P. (1987). Map projections: A working manual.
        U.S. Geological Survey Professional Paper (1395).
        doi: 10.3133/pp1396
        """
        self._load_backend()
        return self._compute_k_hotine(lon, lat, self.lonc, self._lat_0,
                                      self._alpha, self._k0, self._f)


    def enclosing_sphere_center(self):
        """
        Return, if computed, the center of the sphere enclosing
        the data on the ellipsoid - projected to the ellipsoid's
        surface.
        """
        return self._bscenter



    def _load_backend(self):
        """
        Import functions from the C++ backend.
        """
        if not self._backend_loaded:
            from .cppextensions import (bfgs_hotine, project_hotine,
                                        compute_k_hotine)
            self._bfgs_hotine = bfgs_hotine
            self._project_hotine = project_hotine
            self._compute_k_hotine = compute_k_hotine

        self._backend_loaded = True
