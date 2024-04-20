# Initial parameter estimates.
# This file is part of the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#         Sebastian von Specht
#
# Copyright (C) 2022      Deutsches GeoForschungsZentrum Potsdam,
#               2022-2024 Sebastian von Specht
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
from math import asin, degrees, sqrt, isinf, isnan
from .fisherbingham import fisher_bingham_mom, fisher_bingham_angles
from .geometry import latitude2parametric,_lola_aux_2_xyz, _Rx, _Ry, _Rz



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

    k0v = np.sqrt(np.maximum(1-(X0[2])**2,1e-3))
    k0_init = np.mean(k0v)

    for i in range(100):
        if isinf(pnorm):
            if is_p2opt:
                w = np.zeros_like(k0v)
                I = [np.argmax(k0_init-k0v), np.argmin(k0_init-k0v)]
                # I = np.argmax(np.abs(k0_init-k0v))
                w[I] = 1.
                # k0_init = k0v[I]
                # continue
            else:
                w = np.abs(k0_init-k0v)**(pnorm_p0-2)
        elif pnorm > 0. and pnorm <= 1:
            w = (np.abs(k0_init-k0v)+1e-15)**(pnorm-2)
        elif pnorm == 2:
            w = np.ones_like(k0v)
        else:
            w = np.abs(k0_init-k0v)**(pnorm-2)

        if isinf(pnorm) and is_p2opt:
            k0_init -= .1* np.sum(w*wdata * (k0_init-k0v))/np.sum(w*wdata)
        else:
            k0_init -= .1* np.sum(w*wdata * (k0_init-k0v)**2)/np.sum(w*wdata)

    if isnan(k0_init):
        return np.mean(k0v)

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

    # Convert geographic coordinates to parametric Cartesian coordinates
    X = _lola_aux_2_xyz(lon, lat, f)
      
    # Use Fisher-Bingham distribution to compute the central
    # latitude, longitude, and azimuth:
    phi0, lmbdc, alphac = fisher_bingham_angles(X, w)

    # Compute k0:
    k0 = initial_k0(phi0, lmbdc, alphac, X, w, pnorm)

    # Convert parametric latitude phi0 to geographic latitude
    phi0 = latitude2parametric(phi0, f, 'backward')
  
    return np.rad2deg(lmbdc), np.rad2deg(phi0), np.rad2deg(alphac), k0
