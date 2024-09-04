# Code for working within Euclidean coordinates and
# traversing to and fro.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#         Sebastian von Specht
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
#               2022 Sebastian von Specht,
#               2024 Technische Universität München
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
from ._typing import ndarray64
#
# Converting between geographic and Euclidean coordinates:
#


def _xyz2lola(xyz: ndarray64) -> tuple[ndarray64, ndarray64]:
    """
    Takes a vector (3,N) and computes the spherical coordinates.
    """
    xyz /= np.linalg.norm(xyz, axis=0)
    return np.rad2deg(np.arctan2(xyz[1], xyz[0])), np.rad2deg(np.arcsin(xyz[2]))


def _lola2xyz(lo: ndarray64, la: ndarray64, f: float) -> ndarray64:
    """
    Convert geographic coordinates to 3D Euclidean space.

    Returns
    -------
    ndarray
       Array of the Euclidean coordinates in shape (3,N).
    """
    e2 = 2*f-f**2

    lo = np.deg2rad(lo)
    la = np.deg2rad(la)

    N = 1/np.sqrt(1-e2*np.sin(la)**2)

    X = np.zeros((3,lo.size))

    X[0] = N*np.cos(lo)*np.cos(la)
    X[1] = N*np.sin(lo)*np.cos(la)
    X[2] = (N*(1-e2))*np.sin(la)

    return X

def latitude2parametric(la,f,transform='forward'):
    '''
    Conversion between geographic latitude and parametric latitude to
    solve problems for geodesics on the ellipsoid by transforming them
    to an equivalent problem for spherical geodesics.

    'forward'  : from geographic to parametric latitude (in rad)
    'backward' : from parametric to geographic latitude (in rad)
    '''

    if transform == 'forward':
        return np.arctan((1-f)*np.tan(la))
    elif transform == 'backward':
        return np.arctan(np.tan(la)/(1-f))
    else:
        print('Invalid input. Please enter a valid transform ("forward" or "backward").')

def _lola_aux_2_xyz(lo: ndarray64, la: ndarray64, f: float) -> ndarray64:
    """
    Convert ellipsoidal coordinates (lo,la)
    to coordinates on an auxiliary sphere (lo,la_parametric)
    and then to cartesian coordinates X (3xN).

    Returns
    -------
    ndarray
       Array of the Euclidean coordinates on the auxiliary sphere,
       in the shape (3,N).

    Notes
    -----
    Used for computations involving the Fisher-Bingham distribution,
    as this distribution is defined on the sphere rather than an ellipsoid.
    """

    lo = np.deg2rad(lo)
    la = np.deg2rad(la)
    la_parametric = latitude2parametric(la,f)

    X = np.zeros((3,lo.size))

    X[0] = np.cos(lo)*np.cos(la_parametric)
    X[1] = np.sin(lo)*np.cos(la_parametric)
    X[2] = np.sin(la_parametric)

    return X

#
# Rotational matrices:
#


def _Rx(a: float) -> ndarray64:
    R = np.array([
        [1,0,0.],
        [0.,np.cos(a),np.sin(a)],
        [0.,-np.sin(a),np.cos(a)]
    ])
    return R


def _Ry(a: float) -> ndarray64:
    R = np.array([
        [np.cos(a),0.,-np.sin(a)],
        [0,1,0.],
        [np.sin(a),0.,np.cos(a)]
    ])
    return R


def _Rz(a: float) -> ndarray64:
    R = np.array([
        [np.cos(a),-np.sin(a),0.],
        [np.sin(a),np.cos(a),0.],
        [0,0.,1]
    ])
    return R


#
# Ellipsoid scale factor:
#
def desired_scale_factor(
        h: ndarray64,
        lat: ndarray64,
        a: float,
        f: float,
        batch: int = 1000000) -> ndarray64:
    """
    Computes the desired scale factor at a given height and latitude.
    """
    # Make this code suitable for very large input arrays by
    # batch calling (on the expense of a little overhead)
    h = np.asarray(h)
    lat = np.asarray(lat)
    n = max(h.size, lat.size)
    if n > batch:
        bc = np.broadcast(h,lat)
        k_des = np.empty(bc.shape)
        i = 0
        while i < n:
            k_des.flat[i:i+batch] = desired_scale_factor(bc.iters[0][i:i+batch],
                                                         bc.iters[1][i:i+batch],
                                                         a, f, batch)
            i += batch

        return k_des
    else:
        # Don't worry too much.
        e2 = 2*f - f**2
        lat = np.deg2rad(lat)
        clat = np.cos(lat)
        slat = np.sin(lat, out=lat)
        # N = a / np.sqrt(1.0 - e2*np.sin(lat)**2)
        N = slat * slat
        N *= (-e2)
        N += 1
        N = np.sqrt(N, out=N)
        N **= -1
        N *= a
        # x = (N+h)*np.cos(lat)
        x = N + h
        x *= clat
        # z = ((1.0-e2)*N + h) * np.sin(lat)
        z = N
        z *= (1.0 - e2)
        z += h
        z *= slat
        # A = z / (x*(1.0-f))
        A = 1/x
        A /= (1.0 - f)
        A *= z
        # r = np.sqrt(x**2 + z**2)
        z *= z
        x *= x
        x += z
        r = np.sqrt(x, out=x)
        # re = np.sqrt(1.0 - e2 * A**2 / (1.0 + A**2))
        A *= A             # A -> A**2
        np.add(1,A,out=z)  # z = 1 + A**2
        re = A             # A**2
        re *= (-e2)        # -e2 * A**2
        re /= z            # -e2 * A**2 / (1 + A**2)
        re += 1.0          # 1.0 - e2 * A**2 / (1 + A**2)
        re = np.sqrt(re, out=re)
        # k_des = r / (a*re)
        k_des = r
        k_des /= re
        k_des /= a
        return k_des
