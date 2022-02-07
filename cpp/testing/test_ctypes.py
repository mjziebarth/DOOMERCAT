from ctypes import CDLL, c_double, c_size_t, c_uint, POINTER
from pathlib import Path

import numpy as np


p = Path(__file__).resolve().parent / 'doomercat.so'

_doomercat_so = CDLL(p)


def compute_cost(lon, lat, data_lon, data_lat, w, k0, pnorm, k0_ap, sigma_k0,
                 f):
    """
    Computes the cost function.
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
    lon = float(lon)
    lat = float(lat)
    k0 = float(k0)
    pnorm = int(pnorm)
    k0_ap = float(k0_ap)
    sigma_k0 = float(sigma_k0)
    f = float(f)

    result = np.ones(1)
    res = _doomercat_so.compute_cost(c_size_t(N),
                                     data_lon.ctypes.data_as(POINTER(c_double)),
                                     data_lat.ctypes.data_as(POINTER(c_double)),
                                     w.ctypes.data_as(POINTER(c_double)),
                                     c_double(lon),
                                     c_double(lat), c_double(k0), c_double(f),
                                     c_uint(pnorm), c_double(k0_ap),
                                     c_double(sigma_k0),
                                     result.ctypes.data_as(POINTER(c_double)))

    return float(result)


def compute_cost_and_gradient(lon, lat, data_lon, data_lat, w, k0, pnorm,
                              k0_ap, sigma_k0, f):
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
    lon = float(lon)
    lat = float(lat)
    k0 = float(k0)
    pnorm = int(pnorm)
    k0_ap = float(k0_ap)
    sigma_k0 = float(sigma_k0)
    f = float(f)

    result = np.zeros(6)
    res = _doomercat_so.compute_cost_and_derivatives(c_size_t(N),
                                     data_lon.ctypes.data_as(POINTER(c_double)),
                                     data_lat.ctypes.data_as(POINTER(c_double)),
                                     w.ctypes.data_as(POINTER(c_double)),
                                     c_double(lon), c_double(lat), c_double(k0),
                                     c_double(f), c_uint(pnorm),
                                     c_double(k0_ap), c_double(sigma_k0),
                                     result.ctypes.data_as(POINTER(c_double)))

    return float(result[0]), result[1:]


def billo_gradient_descent(data_lon, data_lat, pnorm, k0_ap, sigma_k0, f, lon0, lat0, Nmax):
    """
    Computes the cost function and its derivatives.
    """
    # Ensure types:
    data_lon = np.ascontiguousarray(data_lon, dtype=np.double)
    data_lat = np.ascontiguousarray(data_lat, dtype=np.double)
    N = data_lon.size
    assert data_lat.size == N
    pnorm = int(pnorm)
    k0_ap = float(k0_ap)
    sigma_k0 = float(sigma_k0)
    lon0 = float(lon0)
    lat0 = float(lat0)
    f = float(f)
    Nmax = int(Nmax)
    assert Nmax > 0

    result = np.zeros((Nmax,4))
    res = _doomercat_so.perform_billo_gradient_descent(c_size_t(N),
                                     data_lon.ctypes.data_as(POINTER(c_double)),
                                     data_lat.ctypes.data_as(POINTER(c_double)),
                                     c_double(f), c_uint(pnorm),
                                     c_double(k0_ap), c_double(sigma_k0),
                                     c_double(lon0), c_double(lat0),
                                     c_size_t(Nmax),
                                     result.ctypes.data_as(POINTER(c_double)))

    return result


def bfgs_optimize(data_lon, data_lat, pnorm, k0_ap, sigma_k0, f, lon0, lat0, Nmax):
    """
    Computes the cost function and its derivatives.
    """
    # Ensure types:
    data_lon = np.ascontiguousarray(data_lon, dtype=np.double)
    data_lat = np.ascontiguousarray(data_lat, dtype=np.double)
    N = data_lon.size
    assert data_lat.size == N
    pnorm = int(pnorm)
    k0_ap = float(k0_ap)
    sigma_k0 = float(sigma_k0)
    lon0 = float(lon0)
    lat0 = float(lat0)
    f = float(f)
    Nmax = int(Nmax)
    assert Nmax > 0

    result = np.zeros((Nmax,6))
    M = np.zeros(1,dtype=np.uint)
    res = _doomercat_so.perform_bfgs(c_size_t(N),
                                     data_lon.ctypes.data_as(POINTER(c_double)),
                                     data_lat.ctypes.data_as(POINTER(c_double)),
                                     c_double(f), c_uint(pnorm),
                                     c_double(k0_ap), c_double(sigma_k0),
                                     c_double(lon0), c_double(lat0),
                                     c_size_t(Nmax),
                                     result.ctypes.data_as(POINTER(c_double)),
                                     M.ctypes.data_as(POINTER(c_uint)))
    M = int(M)

    return result[:M,:]
