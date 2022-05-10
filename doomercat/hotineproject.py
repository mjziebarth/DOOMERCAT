# A pure Python implementation of the Hotine oblique Mercator projection.
# Follows cpp/include/hotine.hpp.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam
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
from math import sqrt, radians, sin, cos, tan, pi, asin


EPS_LARGE_PHI = 1e-9;


def hotine_project_uv(lon: np.ndarray, lat: np.ndarray, lonc: float,
                      lat_0: float, alpha: float, k0: float, f: float):
    """
    Performs the Hotine oblique Mercator projection in Python.
    Might be used in cases where loading the external Ctypes library
    is inconvenient.
    """
    # Compute the constants:
    e2 = f*(2.0-f)
    e = sqrt(e2)
    lambda_c = radians(lonc)
    phi0 = radians(lat_0)
    alpha = radians(alpha)
    sin_phi0 = sin(phi0)
    cos_phi0 = cos(phi0)
    B = sqrt(1.0 + e2 * cos_phi0**4 / (1.0 - e2))
    A = B * k0 * sqrt(1.0 - e2) / (1.0 - e2 * sin_phi0**2)
    t0 = tan(pi/4 - 0.5*phi0) * ((1.0 + e * sin_phi0)
                                 / (1.0 - e * sin_phi0))**(0.5*e);
    D = max(B * sqrt(1.0 - e2) / (cos_phi0 * sqrt(1.0 - e2 * sin_phi0**2)),
            1.0)
    F = D + sqrt(D*D - 1.0) if phi0 >= 0.0 else D - sqrt(D*D - 1.0)
    E = F * t0**B
    G = 0.5*(F - 1.0 / F)
    g0 = asin(sin(alpha) / D)
    sin_g0 = sin(g0)
    cos_g0 = cos(g0)
    l0 = lambda_c - asin(G * tan(g0)) / B
    AoB = A / B

    # Now project to u,v:
    N = lon.size
    u = np.empty(N)
    v = np.empty(N)
    phi = np.deg2rad(lat)
    lbd = np.deg2rad(lon)
    mask = (phi >  (1.0 - EPS_LARGE_PHI) * 0.5 * pi) | \
           (phi < -(1.0 - EPS_LARGE_PHI) * 0.5 * pi)

    # The zone where the limiting approximation for phi = +/- pi/2
    # is better than the actual code (1e-9 stemming from some numerical
    # investigations):
    if np.any(mask):
        sign = np.sign(phi[mask])
        u[mask] = AoB * phi[mask]
        v[mask] = AoB * np.log(np.tan(pi/4 - sign*0.5*g0))

    # For the rest, can use the full equations.
    mask = ~mask

    sp = np.sin(phi[mask]);
    t = np.sqrt((1.0 - sp) / (1.0 + sp) * ((1.0 + e*sp) / (1.0 - e*sp))**e)
    Q = E * t**(-B);
    S = 0.5*(Q - 1.0/Q)
    T = 0.5*(Q + 1.0/Q)
    # Delta lambda using the addition / subtraction rule of Snyder (p. 72)
    dlambda = lbd[mask] - l0;
    dlambda[dlambda < -pi] += 2*pi
    dlambda[dlambda > pi] -= 2*pi
    V = np.sin(B*dlambda)
    U = (S * sin_g0 - V * cos_g0) / T

    cBdl = np.cos(B * dlambda)
    u[mask] = AoB * np.arctan2(S*cos_g0 + V*sin_g0, cBdl)
    v[mask] = 0.5 * AoB * np.log((1.0-U)/(1.0 + U))

    return u,v

