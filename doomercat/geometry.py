# Code for working within Euclidean coordinates and
# traversing to and fro.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#         Sebastian von Specht
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
#               2022 Sebastian von Specht,
#               2024 Technical University of Munich
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

#
# Converting between geographic and Euclidean coordinates:
#


def _xyz2lola(xyz):
    """
    Takes a vector (N,3) and computes the spherical coordinates.
    """
    xyz /= np.linalg.norm(xyz, axis=0)
    return np.rad2deg(np.arctan2(xyz[1], xyz[0])), np.rad2deg(np.arcsin(xyz[2]))


def _lola2xyz(lo,la,f):
    e2 = 2*f-f**2

    lo = np.deg2rad(lo)
    la = np.deg2rad(la)

    N = 1/np.sqrt(1-e2*np.sin(la))

    X = np.zeros((3,lo.size))

    X[0] = N*np.cos(lo)*np.cos(la)
    X[1] = N*np.sin(lo)*np.cos(la)
    X[2] = (N*(1-e2))*np.sin(la)

    return X


#
# Rotational matrices:
#


def _Rx(a):
    R = np.array([
        [1,0,0.],
        [0.,np.cos(a),np.sin(a)],
        [0.,-np.sin(a),np.cos(a)]
    ])
    return R


def _Ry(a):
    R = np.array([
        [np.cos(a),0.,-np.sin(a)],
        [0,1,0.],
        [np.sin(a),0.,np.cos(a)]
    ])
    return R


def _Rz(a):
    R = np.array([
        [np.cos(a),-np.sin(a),0.],
        [np.sin(a),np.cos(a),0.],
        [0,0.,1]
    ])
    return R


#
# Ellipsoid scale factor:
#
def desired_scale_factor(h: np.ndarray, lat: np.ndarray, a: float,
                         f: float, batch: int = 1000000) -> np.ndarray:
    """
    Computes the desired scale factor at a given height and latitude.
    """
    # Make this code suitable for very large input arrays by
    # batch calling (on the expense of a little overhead)
    h = np.array(h, copy=False)
    lat = np.array(lat, copy=False)
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
    k_des = r / (a*re)
