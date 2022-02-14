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

def _xyz2lola(xyz):
    """
    Takes a vector (N,3) and computes the spherical coordinates.
    """
    xyz /= np.linalg.norm(xyz)
    return np.rad2deg(np.arctan2(xyz[1], xyz[0])), np.rad2deg(np.arcsin(xyz[2]))


def initial_parameters_cross_product(lon,lat):
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



def initial_parameters_fisher_bingham(lon, lat, w):
    """
    Computes an initial estimate of the parameters using parameter
    estimates of the Fisher-Bingham distribution.

    Returns:
       (lon_cyl, lat_cyl), (lonc, lat_0)
    """
    if w is None:
        w = 1.0
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    v1 = np.stack((np.cos(lon) * np.cos(lat),
                   np.sin(lon) * np.cos(lat),
                   np.sin(lat)))

    mv = np.sum(w*v1,axis=1)/np.sum(w)
    mv /= np.linalg.norm(mv)

    Sv = 1/np.sum(w) * w * v1 @ v1.T

    ct = mv[2]
    st = np.sqrt(1-mv[2]**2)

    cp = mv[1]/np.sqrt(mv[0]**2+mv[1]**2)
    sp = mv[0]/np.sqrt(mv[0]**2+mv[1]**2)

    H1 = np.array([
        [1,0,0.],
        [0,ct,-st],
        [0,st,ct]
    ])

    H2 = np.array([
        [cp,-sp,0],
        [sp,cp,0],
        [0,0.,1]
    ])

    H = H1 @ H2

    Hv1 = H @ v1
    Hmv = H @ mv

    B = H @ Sv @ H.T
    Bl = B[:-1,:-1]

    ee,ev = np.linalg.eig(Bl)
    ee = np.sqrt(ee)

    K = np.block([[ev.T,np.zeros((2,1))],
                  [np.zeros((1,2)),1.],])

    G = K @ H

    g1 = G.T[:,2] # central axis
    g2 = G.T[:,0] # equator axis
    g3 = G.T[:,1] # pole axis

    return _xyz2lola(g3), _xyz2lola(g1)


def initial_parameters(lon, lat, w, how='fisher-bingham'):
    """
    Computes an initial estimate of the parameters.

    Returns:
       (lon_cyl, lat_cyl), (lonc, lat_0)
    """
    if how == 'fisher-bingham':
        return initial_parameters_fisher_bingham(lon, lat, w)
