# Initial parameter estimates.
# This file is part of the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#         Sebastian von Specht
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
#                    Sebastian von Specht
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
from math import asin, degrees, sqrt, isinf
from .fisherbingham import fisher_bingham_mom, fisher_bingham_angles
from .euclid import _lola2xyz, _Rx, _Ry, _Rz



def initial_k0(phi0, lmbdc, alphac, X, wdata, pnorm, is_p2opt=True,
               pnorm_p0=None):
    """
    Computes the initial k0 given the initial central latitude,
    longitude, and azimuth (phi0, lmbdc, alphac), the point set,
    and the p-norm.

    Paramters:
       phi0   : Initial central latitude (in radians)
       lmbdc  : Initial central longitude (in radians)
       alphac : Initial azimuth at the central line (in radians)
    """
    X0 = (_Rz(lmbdc) @ _Ry(phi0) @ _Rx(alphac-np.pi/2)).T @ X

    k0v = np.sqrt(np.maximum(1-(X0[2])**2,0))
    k0_init = np.mean(k0v)

    for i in range(100):
        if isinf(pnorm):
            if is_p2opt:
                w = np.zeros_like(k0v)
                I = [np.argmax(k0_init-k0v), np.argmin(k0_init-k0v)]
                w[I] = 1.
            else:
                w = np.abs(k0_init-k0v)**(pnorm_p0-2)
        elif pnorm > 0. and pnorm <= 1:
            w = (np.abs(k0_init-k0v)+1e-15)**(pnorm-2)
        elif pnorm == 2:
            w = np.ones_like(k0v)
        else:
            w = np.abs(k0_init-k0v)**(pnorm-2)

        k0_init -= .1* np.sum(w*wdata * (k0_init-k0v))/np.sum(w*wdata)

    return k0_init


def initial_parameters_fisher_bingham(lon, lat, w, pnorm, f):
    """
    Computes an initial estimate of the parameters using parameter
    estimates of the Fisher-Bingham distribution.

    Returns:
       lonc, lat_0, alpha, k_0
    """
    # Sanity:
    if w is None:
        w = 1.0
    # Use Fisher-Bingham distribution to compute the central
    # latitude, longitude, and azimuth:
    X = _lola2xyz(lon, lat, f)
    phi0, lmbdc, alphac = fisher_bingham_angles(X, w)

    # Compute k0:
    k0 = initial_k0(phi0, lmbdc, alphac, X, w, pnorm)

    return np.rad2deg(lmbdc), np.rad2deg(phi0), np.rad2deg(alphac), k0


def compute_azimuth(cylinder_axis, central_axis):
    """
    Compute, at the central point, the azimuth of the tangent
    great circle in which an oblique cylinder, specified through
    its symmetry axis, touches a sphere.
    """
    # We need the sine of the latitude of the cylinder axis,
    # which is the cylinder axis z component:
    sin_phi_cyl = cylinder_axis[2]

    # Furthermore, we need the cosine of the central coordinate
    # latitude:
    cos_fies = sqrt(central_axis[0]**2 + central_axis[1]**2)

    # Now, we can compute the azimuth of the cylinder equator at the
    # central coordinate using spherical trigonometry:
    azimuth = asin(max(min(sin_phi_cyl / cos_fies, 1.0), -1.0));

    # The spherical geometry used does not consider the correct
    # sign of the azimuth. Thus, we may have to multiply by -1 if
    # the cylinder axis is to the east of the central axis.
    # Whether the cylinder axis is west or east of the central axis
    # can be decided by its projection to a westward vector.
    # Such a vector is the cross product of the central axis and
    # the North pole:
    westward = np.cross(central_axis, (0,0,1))
    if np.dot(westward, cylinder_axis) > 0.0:
        return degrees(azimuth);

    return -degrees(azimuth);



def initial_parameters(lon, lat, w, pnorm, f, how='fisher-bingham'):
    """
    Computes an initial estimate of the parameters.

    Returns:
       (lon_cyl, lat_cyl), (lonc, lat_0)
    """
    if how == 'fisher-bingham':
        return initial_parameters_fisher_bingham(lon, lat, w, pnorm, f)

    elif how == 'fisher-bingham-alt':
        if w is None:
            w = 1.0
        X = _lola2xyz(lon, lat, f)
        g1, g2, g3 = fisher_bingham_mom(X,w)
        azimuth = compute_azimuth(g3, g1)

        # Code from Fisher-Bingham angles:
        phi0 = asin(g1[2])
        lmbdc = np.arctan2(g1[1],g1[0])
        k0 = initial_k0(phi0, lmbdc, np.deg2rad(azimuth), X, w, pnorm)

        return degrees(lmbdc), degrees(phi0), azimuth, k0

    else:
        raise NotImplementedError("Only 'fisher-bingham' implemented.")
