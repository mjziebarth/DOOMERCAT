# Convenience class for an optimized Hotine oblique Mercator projection.
# This file is part of the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam,
#               2024      Technical University of Munich
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
"""The main class of DOOMERCAT, HotineObliqueMercator.

This class bundles all the functionality of optimizing the
Hotine oblique Mercator projection, setting north at a desired
position, generating the PROJ string, projecting, and computing
the distortion of the projection.
"""


import numpy as np
from math import atan2, degrees, isinf
from typing import Optional, Iterable
from ._typing import ndarray64
from .config import _default
from .defs import _ellipsoids
from .initial import initial_parameters_fisher_bingham
from .enclosingsphere import bounding_sphere
from .hotineproject import hotine_project_uv
from .hotine import lm_adamax_optimize
from .geometry import desired_scale_factor


class HotineObliqueMercator:
    """A Hotine oblique Mercator projection (HOM) optimized for a
    geographical data set. The projection's definition follows
    Snyder [1]_, variant B.

    **Typical Call Signature**

    The following parameters are used to optimize the HOM for
    a set of points.

    Parameters
    ----------
    lon : array_like
        Longitudes of the data set.
    lat : array_like
        Latitudes of the data set.
    h : array_like, optional
        Elevations of the data points with respect to the reference
        ellipsoid. If not given, zero elevation will be assumed for
        each data point.
    weight : array_like, optional
       Multiplicative weights assigned to the data points in the cost
       function. Need to be positive real weights.
    pnorm : float, optional
       Power by which scale factors are weighted in the cost function.
       Let :math:`k_i` be the local scale factor, corrected for height,
       at the i'th residual between projected and ellipsoidal space.
       Then the sum

       .. math:: \sum_i |k_i - 1|^p

       is minimized to obtain an optimal projection. Can be :math:`\infty`
       to use the infinity (sup) norm.
    k0_ap : float,optional
       The threshold beyond which smaller global scale factors
       :math:`k_0` are constrained using a quadratic potential.
       This constraint helps preventing the solver from approaching
       small-circle solutions. Setting **k0_ap** to zero removes
       the constraint.
    sigma_k0 : float, optional
       The scale of the quadratic constrains for small :math:`k_0`,
       that is, the standard deviation of the quadratic branch of the
       :math:`k_0` potential.
    ellipsoid : str, optional
       Name of the reference ellipsoid. Must be one of 'WGS84' and
       'GRS80'. Can be overriden by using the **a** and **f** parameters.
    f : float, optional
       Flattening of the reference rotational ellipsoid.
    a : float, optional
       The large half-axis of of the reference rotational
       ellipsoid.
    lonc0 : float, optional
       Starting value for **lonc** parameter. If not given, will be
       determined by Fisher-Bingham estimator.
    lat_00 : float, optional
       Starting value for **lat_0** parameter. If not given, will be
       determined by Fisher-Bingham estimator.
    alpha0 : float, optional
       Starting value for **alpha** parameter. If not given, will be
       determined by Fisher-Bingham estimator.
    k00 : float, optional
       Starting value for **k0** parameter. If not given, will be
       determined by Fisher-Bingham estimator.
    Nmax : int, optional
       Maximum number of iterations of the optimization algorithms.
    backend : str, optional
       The optimization backend. Either ``'Python'``, a pure Python and
       NumPy implementation using the Levenberg-Marquard algorithm (or
       AdaMax in case that :math:`p=\infty`), or ``'C++'``, a compiled backend
       using the BFGS algorithm.
    fisher_bingham_use_weight : bool, optional
       If `True`, use data weights when computing the starting value of
       the optimization using the Fisher-Bingham distribution.
    compute_enclosing_sphere : bool, optional
       If `True`, compute the geographic center of the enclosing sphere,
       which can be used to set north at its center.
    bfgs_epsilon : float, optional
       Tolerance parameter for the BFGS exit condition.


    **Calling Without Optimization**

    The other parameters can be used to use the functionality of the class
    for given projection parameters. These two sets of parameters should be
    used exclusively:

    Parameters
    ----------
    lonc : float, optional
       Longitude of central point.
    lat_0 : float, optional
       Latitude of the central point.
    alpha : float, optional
       Azimuth of the oblique equator at the central point.
    k0 : float, optional
       Global scale factor of the projection.


    Notes
    -----
    This computation follows the equations given by Snyder [1]_,
    variant B, which are derived for the Hotine oblique Mercator
    projection. For practical purposes, this can often be equivalent
    to the Laborde oblique Mercator [2]_ [3]_ [4]_.
    In any case, the relevant implementation of the oblique
    Mercator projection in PROJ follows the Hotine oblique
    Mercator equations by Snyder (1987). Hence, the distortion
    as returned by this method represents most practical
    use cases.

    References
    ----------
    .. [1] Snyder, J. P. (1987). Map projections: A working manual.
       U.S. Geological Survey Professional Paper (1395).
       doi: 10.3133/pp1396

    .. [2] Laborde, J. (1928). La nouvelle projection du Service
       Geographique de Madagascar. Madagascar, Cahiers du Service
       geographique de Madagascar, Tananarive 1:70

    .. [3] Roggero, M. (2009). Laborde projection in Madagascar
       cartography and its recovery in WGS84 datum. Appl Geomat 1,
       131. doi:10.1007/s12518-009-0010-4

    .. [4] EPSG Guidance Notes 7-2.
    """
    def __init__(self,
                 lon: Optional[Iterable[float]] = None,
                 lat: Optional[Iterable[float]] = None,
                 h: Optional[Iterable[float]] = None,
                 weight: Optional[Iterable[float]] = None,
                 pnorm: int = _default["pnorm"],
                 k0_ap: float = _default["k0_ap"],
                 sigma_k0: float = _default["sigma_k0"],
                 ellipsoid: Optional[str] = _default["ellipsoid"],
                 f: Optional[float] = _default["f"],
                 a: Optional[float] = _default["a"],
                 lonc0: Optional[float] = None,
                 lat_00: Optional[float] = None,
                 alpha0: Optional[float] = None,
                 k00: Optional[float] = None, lonc: Optional[float] = None,
                 lat_0: Optional[float] = None, alpha: Optional[float] = None,
                 k0: Optional[float] = None,
                 Nmax: int = _default["Nmax"],
                 proot: bool = _default["proot"],
                 logger: object = None,
                 backend: str = _default["backend"],
                 fisher_bingham_use_weight: bool
                     = _default["fisher_bingham_use_weight"],
                 compute_enclosing_sphere: bool
                     = _default["compute_enclosing_sphere"],
                 bfgs_epsilon: float = _default["bfgs_epsilon"],
                 Nmax_pre_adamax: int = _default["Nmax_pre_adamax"]):
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

        # For type checking purposes:
        lon_array: ndarray64
        lat_array: ndarray64

        # Check whether lon/lat given:
        if lon is not None:
            # Case 1: Data is given as lon/lat, so we optimize the projection.
            assert lat is not None
            lon_array = np.array(lon, copy=False)
            lat_array = np.array(lat, copy=False)
            if weight is not None:
                if not isinstance(weight,np.ndarray):
                    weight = np.array(weight)
                weight /= weight.sum()
                assert isinstance(weight, np.ndarray)
                assert weight.shape == lon_array.shape
            if h is not None:
                if not isinstance(h,np.ndarray):
                    h = np.array(h)
                assert h.shape == lon_array.shape

            assert lon_array.shape == lat_array.shape

            # Initial guess for the parameters:
            if any(p is None for p in (lonc0, lat_00, alpha0, k00)):
                if fisher_bingham_use_weight:
                    w_initial = weight
                else:
                    w_initial = None
                lonc0, lat_00, alpha0, k00 \
                    = initial_parameters_fisher_bingham(
                        lon_array,
                        lat_array,
                        w_initial,
                        pnorm,
                        f
                        )
            else:
                # Could be required in Python backend:
                w_initial = None

            if backend in ('c++','C++'):
                # Call the C++ BFGS backend.
                if logger is not None:
                    logger.log(20, "Starting BFGS optimization.")

                # Load the C++ backend:
                self._load_backend()

                # If p=inf, pre-optimize with p=80 norm:
                if isinf(pnorm):
                    pre_res = \
                        self._bfgs_optimize(
                            lon_array,
                            lat_array,
                            h,
                            weight,
                            80,
                            k0_ap,
                            sigma_k0,
                            a,
                            f,
                            lonc0,
                            lat_00,
                            alpha0,
                            k00,
                            Nmax,
                            proot,
                            epsilon=bfgs_epsilon
                        )
                    lonc0  = pre_res.lonc
                    lat_00 = pre_res.lat_0
                    alpha0 = pre_res.alpha
                    k00    = pre_res.k0

                # Optimize the Hotine oblique Mercator:
                result = \
                    self._bfgs_optimize(
                        lon_array,
                        lat_array,
                        h,
                        weight,
                        pnorm,
                        k0_ap,
                        sigma_k0,
                        a,
                        f,
                        lonc0,
                        lat_00,
                        alpha0,
                        k00,
                        Nmax,
                        proot,
                        epsilon=bfgs_epsilon
                    )

            elif backend in ('truong2020',):
                # Call the C++ two-way backtracking gradient descent
                if logger is not None:
                    logger.log(20, "Starting Truong & Nguyen (2020) "
                                   "optimization.")

                # Load the C++ backend:
                self._load_backend()

                # Optimize the Hotine oblique Mercator:
                result = \
                    self._truong2020_optimize(
                        lon_array,
                        lat_array,
                        h,
                        weight,
                        pnorm,
                        k0_ap,
                        sigma_k0,
                        a,
                        f,
                        lonc0,
                        lat_00,
                        alpha0,
                        k00,
                        Nmax,
                        proot,
                        epsilon=bfgs_epsilon
                    )

            elif backend in ('python','Python'):
                # Call the Python Levenberg-Marquardt backend.
                if logger is not None:
                    logger.log(20, "Starting Levenberg-Marquardt "
                                   "optimization.")
                if weight is None:
                    weight = np.ones_like(lon_array)
                if h is None:
                    h = np.zeros_like(lon_array)
                print(np.deg2rad(lat_00),
                        np.deg2rad(lonc0),
                        np.deg2rad(alpha0),
                        k00,
                        a,
                        f,
                        pnorm,
                        Nmax,
                        k0_ap,
                        sigma_k0)
                result \
                    = lm_adamax_optimize(
                        np.deg2rad(lon_array),
                        np.deg2rad(lat_array),
                        h,
                        weight,
                        np.deg2rad(lat_00),
                        np.deg2rad(lonc0),
                        np.deg2rad(alpha0),
                        k00,
                        a,
                        f,
                        pnorm,
                        Nmax,
                        Nmax_pre_adamax,
                        False,
                        k0_ap,
                        sigma_k0
                    )
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
                bscenter = bounding_sphere(lon, lat, a, f)
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
            self.optimization_result = None


        # Save all attributes, with some type checking
        # beforehand:
        assert alpha is not None
        assert lonc is not None
        assert lat_0 is not None
        assert k0 is not None
        assert f is not None
        assert a is not None
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
               self._a, self._ellipsoid, self._bscenter, self.optimization_result


    def __setstate__(self, state):
        print("__setstate__ tuple ",len(state))
        self._alpha, self._lonc, self._lat_0, self._k0, self._f, self._a, \
           self._ellipsoid, self._bscenter, self.optimization_result = state
        self._backend_loaded = False


    def lonc(self) -> float:
        """
        Longitude of the projection's central point.

        Returns
        -------
        float
        """
        return self._lonc


    def lat_0(self) -> float:
        """
        Latitude of the projection's central point.

        Returns
        -------
        float
        """
        return self._lat_0


    def alpha(self) -> float:
        """
        Azimuth of projection's equator at central point (lonc,lat_0).

        Returns
        -------
        float
        """
        return self._alpha

    def k0(self) -> float:
        """
        Scaling of the central line great circle compare to the ellipsoid.

        Returns
        -------
        float
        """
        return self._k0


    def ellipsoid(self) -> Optional[str]:
        """
        Name of the reference ellipsoid for this projection.

        Returns
        -------
        str or ``None``
        """
        return self._ellipsoid


    def a(self) -> float:
        """
        Length of the large half axis of the reference ellipsoid.

        Returns
        -------
        float
        """
        return self._a


    def f(self) -> float:
        """
        Flattening of the reference ellipsoid.

        Returns
        -------
        float
        """
        return self._f


    def proj4_string(self, orient_north=None) -> str:
        """
        Return a projection string for use with PROJ/GDAL.

        Returns
        -------
        str
            Projection string in (old-style) PROJ syntax
            using the 'omerc' projection (see [1]_).
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


    def project(self, lon: Iterable[float], lat: Iterable[float],
                gamma: Optional[float] = None
        ) -> tuple[ndarray64, ndarray64]:
        """
        Project a geographical coordinate set.

        Parameters
        ----------
        lon : array_like
            Longitudes in degrees.
        lat : array_like
            Latitudes in degrees. Has to be same shape as ``lon``.
        gamma : float, optional
            Angle of rotation in the projected plane. If ``None``,
            will be set to the projection's :math:`\\alpha`. The
            method ``north_gamma`` can be used to compute a ``gamma``
            such that at a point of choice points, the *y* direction
            points north.

        Returns
        -------
        x : np.ndarray
            Projected *x* coordinates.
        y : np.ndarray
            Projected *y* coordinates.

        Notes
        -----
        Uses the Hotine oblique Mercator equations as described by
        Snyder [1]_. Can only be used if the compiled C++ backend
        is available.
        """
        self._load_backend()
        if gamma is None:
            gamma = self._alpha
        return self._project_hotine(lon, lat, self._lonc, self._lat_0,
                                    self._alpha, self._k0, gamma, self._f)


    def north_azimuth(self, lon: float, lat: float,
                      delta: float = 1e-7) -> float:
        """
        Compute, at a location, the local 'north azimuth', that is,
        the clockwise angle from the *y* axis to the local north vector.

        Parameters
        ----------
        lon : float
            Longitude of the point in degrees.
        lat : float
            Latitude of the point in degrees.
        delta : float, optional
            Step width used for numerical estimation of the
            coordinate gradient.

        Returns
        -------
        azimuth : float
           The local azimuth in degrees.
        """
        # Compute the local vectors in north direction:
        lon = float(lon)
        lat = float(lat)
        u,v = hotine_project_uv(np.array((lon,lon)),
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
        Compute the oblique Mercator rectification gamma [1]_
        so that the projected map points northwards in positive
        *y* direction ("up") at the location specified by
        ``lon`` and ``lat``.

        Returns
        -------
        float
        """
        # Compute the north azimuth in u,v coordinates:
        angle: float = self.north_azimuth(lon, lat)

        # Now take into account that the rotation spelled out by Snyder (1987)
        # actually perform a 90° rotation from u->y and v->x.
        return angle - 90.0


    def distortion(self, lon: Iterable[float], lat: Iterable[float],
                   h: Optional[Iterable[float]] = None) -> ndarray64:
        """
        Distortion of the Hotine oblique Mercator projection [1]_.

        Parameters
        ----------
        lon : array_like
            Longitudes :math:`\lambda_i`, in degrees, at which the distortion
            should be computed.
        lat : array_like
            Corresponding latitudes :math:`\phi_i` in degrees.
        h : array_like, optional
            Corresponding elevations :math:`h_i` with respect to the reference
            ellipsoid in meters.

        Returns
        -------
        np.ndarray
            Distortion at the data points,

            .. math:: \\frac{k(\lambda_i, \phi_i)}
                            {k_\\mathrm{des}(h_i, \phi_i)} - 1,

            where :math:`k_\\mathrm{dest}(h_i, \phi_i)` is the desired scale
            factor at the location due to the local elevation with respect to
            the reference ellipsoid.
        """
        self._load_backend()
        # The Hotine k, a citizen of the ellipsoid:
        k = self._compute_k_hotine(lon, lat, self.lonc, self._lat_0,
                                   self._alpha, self._k0, self._f)
        # Advancing it to height:
        if h is not None:
            k /= desired_scale_factor(h, lat, self._a, self._f)
        return k - 1.0


    def enclosing_sphere_center(self) -> Optional[tuple[float,float]]:
        """
        The center of the sphere enclosing the data on the ellipsoid.

        Returns
        -------
        (float,float), optional
            Center of the enclosing sphere projected to the ellipsoid's
            surface. If ``compute_enclosing_sphere=False`` has been
            passed at the construction of this object, or it has not been
            generated by optimizing for some data, ``None`` is returned.
        """
        return self._bscenter



    def _load_backend(self):
        """
        Import functions from the C++ backend.
        """
        if not self._backend_loaded:
            from .cppextensions import (bfgs_optimize, project_hotine,
                                        compute_k_hotine, truong2020_optimize)
            self._bfgs_optimize = bfgs_optimize
            self._project_hotine = project_hotine
            self._compute_k_hotine = compute_k_hotine
            self._truong2020_optimize = truong2020_optimize

        self._backend_loaded = True
