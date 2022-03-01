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
from math import asin, degrees, sqrt
from .fisherbingham import fisher_bingham_mom


def _xyz2lola(xyz):
    """
    Takes a vector (N,3) and computes the spherical coordinates.
    """
    xyz /= np.linalg.norm(xyz, axis=0)
    return np.rad2deg(np.arctan2(xyz[1], xyz[0])), np.rad2deg(np.arcsin(xyz[2]))


def initial_axes_cross_product(lon,lat):
    """
    Computes an initial estimate of the parameters using the average
    cross product between data point pairs.

    Returns:
       (lon_cyl, lat_cyl), (lonc, lat_0)
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    v0 = np.stack((np.cos(lon) * np.cos(lat),
                   np.sin(lon) * np.cos(lat),
                   np.sin(lat))).T
    v_cumul = np.cumsum(v0, axis=0)

    # estimate of the cylinder axis:
    est0 = np.cross(v0, v_cumul).sum(axis=0)
    est0 /= np.linalg.norm(est0)

    # estimate the lonc,latc:
    est1 = v_cumul[-1,:] / np.linalg.norm(v_cumul[-1,:])

    return _xyz2lola(est0), _xyz2lola(est1)



def initial_axes_fisher_bingham(lon, lat, w):
    """
    Computes an initial estimate of the parameters using parameter
    estimates of the Fisher-Bingham distribution.

    Returns:
       cylinder_axis, central_axis
    """
    g1, g2, g3 = fisher_bingham_mom(lon, lat, w)

    return g3, g1


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


def initial_parameters(lon, lat, w, how='fisher-bingham'):
    """
    Computes an initial estimate of the parameters.

    Returns:
       (lon_cyl, lat_cyl), (lonc, lat_0)
    """
    if how == 'fisher-bingham':
        cylinder_axis, central_axis \
           = initial_axes_fisher_bingham(lon, lat, w)

    lonc, lat_0 = _xyz2lola(central_axis)
    azimuth = compute_azimuth(cylinder_axis, central_axis)
    return float(lonc), float(lat_0), float(azimuth)
