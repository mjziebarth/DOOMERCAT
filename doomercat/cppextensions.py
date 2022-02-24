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
import platform
import numpy as np
from ctypes import CDLL, c_double, c_size_t, c_uint, POINTER
from pathlib import Path
from .messages import info


# Load the C++ code.
# In setup.py, we compiled the C++ code into an extension named like this
# Python file.
# The extension's name differs depending on the operating system:

_cppextensions_so = None

def load_cppextensions():
    """
    Reload the extension.
    """
    global _cppextensions_so
    if _cppextensions_so is not None:
        return

    # Now load:
    system = platform.system()
    if system == 'Linux':
        paths = list(Path(__file__).resolve().parent.glob('_cppextensions*.so'))
        if len(paths) > 1:
            raise ImportError("Could not find a unique binary _cppextensions "
                              "library.")
        elif len(paths) == 0:
            raise ImportError("Could not find any binary _cppextensions library.")
        _cppextensions_so = CDLL(paths[0])
        info("Loaded shared library \"" + str(paths[0]) + "\"")

    elif system == 'Windows':
        raise NotImplementedError()
    elif system == 'Darwin':
        raise NotImplementedError()



#
# Define the Python interface:
#

class LabordeResult:
    """
    Result of the optimization.
    """
    def __init__(self, cost, lonc, lat_0, alpha, k0, lon_cyl, lat_cyl, steps):
        self.cost = cost
        self.lonc = lonc
        self.lat_0 = lat_0
        self.alpha = alpha
        self.k0 = k0
        self.lon_cyl = lon_cyl
        self.lat_cyl = lat_cyl
        self.N = 1 if isinstance(cost,float) else cost.size
        self.steps = steps

    def last(self):
        if self.N == 1:
            return self
        return LabordeResult(cost=self.cost[-1], lonc=self.lonc[-1],
                             lat_0=self.lat_0[-1], alpha=self.alpha[-1],
                             k0=self.k0[-1], lon_cyl=self.lon_cyl[-1],
                             lat_cyl=self.lat_cyl[-1], steps=self.steps,
                             qr=self.qr, qi=self.qi, qj=self.qj, qk=self.qk,
                             f=self.f)


class HotineResult:
    """
    Result of the optimization.
    """
    def __init__(self, cost, lonc, lat_0, alpha, k0, grad_lonc, grad_lat0,
                 grad_alpha, grad_k0, steps, f, mode):
        self.cost = cost
        self.lonc = lonc
        self.lat_0 = lat_0
        self.alpha = alpha
        self.k0 = k0
        self.grad_lonc = grad_lonc
        self.grad_lat0 = grad_lat0
        self.grad_alpha = grad_alpha
        self.grad_k0 = grad_k0
        self.N = 1 if isinstance(cost,float) else cost.size
        self.steps = steps
        self.f = f
        self.mode = mode

    def last(self):
        if self.N == 1:
            return self
        return HotineResult(cost=self.cost[-1], lonc=self.lonc[-1],
                            lat_0=self.lat_0[-1], alpha=self.alpha[-1],
                            k0=self.k0[-1], grad_lonc=self.grad_lonc[-1],
                            grad_lat0=self.grad_lat0[-1],
                            grad_alpha=self.grad_alpha[-1],
                            grad_k0=self.grad_k0[-1], steps=self.steps,
                            f=self.f, mode=self.mode[-1])


def adam_optimize(data_lon, data_lat, w, pnorm, k0_ap, sigma_k0, f, lon0, lat0,
                  lonc0, k00, Nmax, return_full_history=False):
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
    lon0 = float(lon0)
    lat0 = float(lat0)
    lonc0 = float(lonc0)
    k00 = float(k00)
    f = float(f)
    Nmax = int(Nmax)
    assert Nmax > 0

    # Make sure that we have loaded the CDLL:
    load_cppextensions()

    result = np.zeros((Nmax,7))
    M = np.zeros(1,dtype=np.uint)
    final_cylinder = np.zeros(6)
    res = _cppextensions_so.perform_adam(c_size_t(N),
                                     data_lon.ctypes.data_as(POINTER(c_double)),
                                     data_lat.ctypes.data_as(POINTER(c_double)),
                                     w.ctypes.data_as(POINTER(c_double)),
                                     c_double(f), c_uint(pnorm),
                                     c_double(k0_ap), c_double(sigma_k0),
                                     c_double(lon0), c_double(lat0),
                                     c_double(lonc0), c_double(k00),
                                     c_size_t(Nmax),
                                     result.ctypes.data_as(POINTER(c_double)),
                                     M.ctypes.data_as(POINTER(c_uint)),
                                     final_cylinder.ctypes
                                            .data_as(POINTER(c_double)))
    M = int(M)

    if return_full_history:
        return LabordeResult(cost=result[:M,0],
                             lonc=result[:M,1],
                             lat_0=result[:M,2],
                             alpha=result[:M,3],
                             k0=result[:M,4],
                             lon_cyl=result[:M,5],
                             lat_cyl=result[:M,6],
                             steps=M,
                             qr=final_cylinder[0],
                             qi=final_cylinder[1],
                             qj=final_cylinder[2],
                             qk=final_cylinder[3],
                             f=final_cylinder[5])

    cost, lonc, lat_0, alpha, k0, lon_cyl, lat_cyl = result[M-1,:]
    return LabordeResult(cost=float(cost),
                         lonc=float(lonc),
                         lat_0=float(lat_0),
                         alpha=float(alpha),
                         k0=float(k0),
                         lon_cyl=float(lon_cyl),
                         lat_cyl=float(lat_cyl),
                         steps=M,
                         qr=final_cylinder[0],
                         qi=final_cylinder[1],
                         qj=final_cylinder[2],
                         qk=final_cylinder[3],
                         f=final_cylinder[5])



def bfgs_hotine(data_lon, data_lat, w, pnorm, k0_ap, sigma_k0, f, lonc_0,
                lat_0_0, alpha_0, k_0_0, Nmax, return_full_history=False):
    """
    Perform BFGS on Hotine oblique Mercator cost function.
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
    lonc_0  = float(lonc_0)
    lat_0_0 = float(lat_0_0)
    alpha_0 = float(alpha_0)
    k_0_0   = float(k_0_0)
    f = float(f)
    Nmax = int(Nmax)
    assert Nmax > 0

    # Make sure that we have loaded the CDLL:
    load_cppextensions()

    result = np.zeros((Nmax,10))
    M = np.zeros(1,dtype=np.uint)

    res = _cppextensions_so.hotine_bfgs(c_size_t(N),
                                 data_lon.ctypes.data_as(POINTER(c_double)),
                                 data_lat.ctypes.data_as(POINTER(c_double)),
                                 w.ctypes.data_as(POINTER(c_double)),
                                 c_double(f), c_uint(pnorm), c_double(k0_ap),
                                 c_double(sigma_k0), c_double(lonc_0),
                                 c_double(lat_0_0), c_double(alpha_0),
                                 c_double(k_0_0), c_size_t(Nmax),
                                 result.ctypes.data_as(POINTER(c_double)),
                                 M.ctypes.data_as(POINTER(c_uint)))
    M = int(M)

    if return_full_history:
        return HotineResult(cost=result[:M,0],
                            lonc=result[:M,1],
                            lat_0=result[:M,2],
                            alpha=result[:M,3],
                            k0=result[:M,4],
                            grad_lonc=result[:M,5],
                            grad_lat0=result[:M,6],
                            grad_alpha=result[:M,7],
                            grad_k0=result[:M,8],
                            steps=M,
                            f=f,
                            mode=result[:M,9].astype(int))

    cost, lonc, lat_0, alpha, k0, grad_lonc, grad_lat0,\
       grad_alpha, grad_k0, mode = result[M-1,:]
    return HotineResult(cost=float(cost),
                        lonc=float(lonc),
                        lat_0=float(lat_0),
                        alpha=float(alpha),
                        k0=float(k0),
                        grad_lonc=float(grad_lonc),
                        grad_lat0=float(grad_lat0),
                        grad_alpha=float(grad_alpha),
                        grad_k0=float(grad_k0),
                        steps=M,
                        f=f,
                        mode=int(mode))




def bfgs_optimize(data_lon, data_lat, w, pnorm, k0_ap, sigma_k0, f, lon0, lat0,
                  lonc0, k00, Nmax, return_full_history=False):
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
    lon0 = float(lon0)
    lat0 = float(lat0)
    lonc0 = float(lonc0)
    k00 = float(k00)
    f = float(f)
    Nmax = int(Nmax)
    assert Nmax > 0

    # Make sure that we have loaded the CDLL:
    load_cppextensions()

    result = np.zeros((Nmax,7))
    M = np.zeros(1,dtype=np.uint)
    final_cylinder = np.zeros(6)
    res = _cppextensions_so.perform_bfgs(c_size_t(N),
                                     data_lon.ctypes.data_as(POINTER(c_double)),
                                     data_lat.ctypes.data_as(POINTER(c_double)),
                                     w.ctypes.data_as(POINTER(c_double)),
                                     c_double(f), c_uint(pnorm),
                                     c_double(k0_ap), c_double(sigma_k0),
                                     c_double(lon0), c_double(lat0),
                                     c_double(lonc0), c_double(k00),
                                     c_size_t(Nmax),
                                     result.ctypes.data_as(POINTER(c_double)),
                                     M.ctypes.data_as(POINTER(c_uint)),
                                     final_cylinder.ctypes
                                            .data_as(POINTER(c_double)))
    M = int(M)

    if return_full_history:
        return LabordeResult(cost=result[:M,0],
                             lonc=result[:M,1],
                             lat_0=result[:M,2],
                             alpha=result[:M,3],
                             k0=result[:M,4],
                             lon_cyl=result[:M,5],
                             lat_cyl=result[:M,6],
                             steps=M)

    cost, lonc, lat_0, alpha, k0, lon_cyl, lat_cyl = result[M-1,:]
    return LabordeResult(cost=float(cost),
                         lonc=float(lonc),
                         lat_0=float(lat_0),
                         alpha=float(alpha),
                         k0=float(k0),
                         lon_cyl=float(lon_cyl),
                         lat_cyl=float(lat_cyl),
                         steps=M,
                         qr=final_cylinder[0],
                         qi=final_cylinder[1],
                         qj=final_cylinder[2],
                         qk=final_cylinder[3],
                         f=final_cylinder[5])


def laborde_project(lon: np.ndarray, lat: np.ndarray, qr: float, qi: float,
                    qj: float, qk: float, k0: float, f: float, a: float):
    """
    Perform Laborde projection as used in the optimization.
    """
    # Input sanitization.
    lon = np.ascontiguousarray(lon, dtype=np.double)
    lat = np.ascontiguousarray(lat, dtype=np.double)
    N = lon.size
    assert lat.size == N
    qr = float(qr)
    qi = float(qi)
    qj = float(qj)
    qk = float(qk)
    k0 = float(k0)
    f = float(f)
    a = float(a)

    # Result vector:
    xy = np.empty((N,2))

    # Call the library:
    _cppextensions_so.laborde_project(c_size_t(N),
                                      lon.ctypes.data_as(POINTER(c_double)),
                                      lat.ctypes.data_as(POINTER(c_double)),
                                      c_double(qr), c_double(qi),
                                      c_double(qj), c_double(qk),
                                      c_double(k0), c_double(f),
                                      c_double(a),
                                      xy.ctypes.data_as(POINTER(c_double)))

    return xy


def compute_cost_k0_iter(k0: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                         qr: float, qi: float, qj: float, qk: float,
                         f: float, pnorm: int, k0_ap: float,
                         sigma_k0: float):
    """
    Computes the cost function for different k0.
    """
    # Input sanitization.
    k0 = np.ascontiguousarray(k0, dtype=np.double)
    lon = np.ascontiguousarray(lon, dtype=np.double)
    lat = np.ascontiguousarray(lat, dtype=np.double)
    w = np.ones_like(lon)
    N = lon.size
    assert lat.size == N
    M = k0.size
    qr = float(qr)
    qi = float(qi)
    qj = float(qj)
    qk = float(qk)
    f = float(f)
    k0_ap = float(k0_ap)
    sigma_k0 = float(sigma_k0)

    # Result vector:
    cost = np.empty(M)

    _cppextensions_so.compute_cost_k0_iter(c_size_t(N),
            lon.ctypes.data_as(POINTER(c_double)),
            lat.ctypes.data_as(POINTER(c_double)),
            w.ctypes.data_as(POINTER(c_double)),
            c_size_t(M), k0.ctypes.data_as(POINTER(c_double)),
            c_double(qr), c_double(qi), c_double(qj), c_double(qk),
            c_double(f), c_uint(pnorm), c_double(k0_ap),
            c_double(sigma_k0), cost.ctypes.data_as(POINTER(c_double)))

    return cost


def compute_cost_hotine(lonc: np.ndarray, lat_0: np.ndarray,
                        alpha: np.ndarray, k_0: np.ndarray,
                        lon: np.ndarray, lat: np.ndarray,
                        w: np.ndarray,
                        f: float, pnorm: int, k0_ap: float,
                        sigma_k0: float):
    """
    Computes the cost function for different k0.
    """
    # Input sanitization.
    lonc = np.ascontiguousarray(lonc, dtype=np.double)
    lat_0 = np.ascontiguousarray(lat_0, dtype=np.double)
    alpha = np.ascontiguousarray(alpha, dtype=np.double)
    k_0 = np.ascontiguousarray(k_0, dtype=np.double)
    lon = np.ascontiguousarray(lon, dtype=np.double)
    lat = np.ascontiguousarray(lat, dtype=np.double)
    N = lon.size
    if w is None:
        w = np.ones_like(lon)
    else:
        w = np.ascontiguousarray(w, dtype=np.double)
        assert w.size == N
    assert lat.size == N
    M = lonc.size
    assert lat_0.size == M
    assert alpha.size == M
    assert k_0.size == M
    f = float(f)
    k0_ap = float(k0_ap)
    sigma_k0 = float(sigma_k0)

    # Result vector:
    cost = np.empty(M)

    _cppextensions_so.compute_cost_hotine_batch(c_size_t(N),
            lon.ctypes.data_as(POINTER(c_double)),
            lat.ctypes.data_as(POINTER(c_double)),
            w.ctypes.data_as(POINTER(c_double)),
            c_size_t(M),
            lonc.ctypes.data_as(POINTER(c_double)),
            lat_0.ctypes.data_as(POINTER(c_double)),
            alpha.ctypes.data_as(POINTER(c_double)),
            k_0.ctypes.data_as(POINTER(c_double)),
            c_double(f), c_uint(pnorm), c_double(k0_ap),
            c_double(sigma_k0),
            cost.ctypes.data_as(POINTER(c_double)));

    return cost

def compute_k_hotine(lon: np.ndarray, lat: np.ndarray,
                     lonc: float, lat_0: float, alpha: float, k_0: float,
                     f: float):
    # Input sanitization.
    lon = np.ascontiguousarray(lon, dtype=np.double)
    lat = np.ascontiguousarray(lat, dtype=np.double)
    w = np.ones_like(lon)
    N = lon.size
    assert lat.size == N
    lonc = float(lonc)
    lat_0 = float(lat_0)
    alpha = float(alpha)
    k_0 = float(k_0)
    f = float(f)

    # Result vector:
    k = np.empty(N)

    _cppextensions_so.compute_k_hotine(c_size_t(N),
            lon.ctypes.data_as(POINTER(c_double)),
            lat.ctypes.data_as(POINTER(c_double)),
            w.ctypes.data_as(POINTER(c_double)),
            c_double(lonc), c_double(lat_0), c_double(alpha),
            c_double(k_0), c_double(f),
            k.ctypes.data_as(POINTER(c_double)))

    return k
