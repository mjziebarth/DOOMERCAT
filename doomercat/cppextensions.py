# Import of the C++ backend code.
# This file is part of the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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
import os
import numpy as np
from ctypes import CDLL, c_double, c_size_t, c_uint, POINTER
from pathlib import Path


# Load the C++ code.
# In setup.py, we compiled the C++ code into an extension named like this
# Python file.
# The extension's name differs depending on the operating system:

if os.name == 'posix':
    paths = list(Path(__file__).resolve().parent.glob('_cppextensions*.so'))
    if len(paths) > 1:
        raise ImportError("Could not find a unique binary _cppextensions "
                          "library.")
    elif len(paths) == 0:
        raise ImportError("Could not find any binary _cppextensions library.")
    _cppextensions_so = CDLL(paths[0])

elif os.name == 'nt':
    raise NotImplementedError()

#
# Define the Python interface:
#

def bfgs_optimize(data_lon, data_lat, w, pnorm, k0_ap, sigma_k0, f, lon0, lat0,
                  Nmax, return_full_history=False):
    """
    Computes the cost function and its derivatives.
    """
    # Ensure types:
    if w is None:
        w = np.ones_like(data_lon)
    data_lon = np.ascontiguousarray(data_lon, dtype=np.double)
    data_lat = np.ascontiguousarray(data_lat, dtype=np.double)
    w = np.ascontiguousarray(w, dtype=np.double)
    N = data_lon.size
    assert data_lat.size == N
    assert w.size == N
    pnorm = int(pnorm)
    k0_ap = float(k0_ap)
    sigma_k0 = float(sigma_k0)
    lat0 = float(lat0)
    f = float(f)
    Nmax = int(Nmax)
    assert Nmax > 0

    result = np.zeros((Nmax,6))
    M = np.zeros(1,dtype=np.uint)
    res = _cppextensions_so.perform_bfgs(c_size_t(N),
                                     data_lon.ctypes.data_as(POINTER(c_double)),
                                     data_lat.ctypes.data_as(POINTER(c_double)),
                                     w.ctypes.data_as(POINTER(c_double)),
                                     c_double(f), c_uint(pnorm),
                                     c_double(k0_ap), c_double(sigma_k0),
                                     c_double(lon0), c_double(lat0),
                                     c_size_t(Nmax),
                                     result.ctypes.data_as(POINTER(c_double)),
                                     M.ctypes.data_as(POINTER(c_uint)))
    M = int(M)

    if return_full_history:
        return result[:M,:]

    cost, lat_0, alpha, k0, _, _ = result[M-1,:]
    return float(lat_0), float(alpha), float(k0)
