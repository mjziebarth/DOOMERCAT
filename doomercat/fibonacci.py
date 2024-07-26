# Fibonacci lattice to generate uniformly distributed points on a sphere.
#
# Author: Sebastian von Specht (s.von.specht@protonmail.com),
#         Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022-2024 Sebastian von Specht,
#               2024      Technical University of Munich
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
from numba import njit

@njit
def fibonacci_lattice(D: float, 
                      a: float = 6378.137, 
                      phi_min: float = -90,
                      phi_max: float = 90,
                      lam_min: float = -180,
                      lam_max: float = 180):
    """
    Returns (almost) equally spaced points with a Fibonacci lattice
    on a sphere with radius 'a' in longitude and latitude. 

    Parameters
    ----------
    D : float
        Average distance of points on the sphere/ellipsoid (unit follows 
        from unit of radius/semi-major axis, default is in kilometer).
    a : float, optional
        Radius of the sphere (semi-major axis of the ellipsoid). 
        Default is the WGS84 radius in kilometer, f = 6378.137 km.
    phi_min, phi_max, lam_min, lam_max : float, optional
        extent (in degrees) of a region for which the 
        lattice should be computed.
        Default is global coverage; phi_min = -90, phi_max = 90, 
        lam_min = 180, lam_max = 180.
        If a region is crossing the +/-180th meridian, then set 
        lam_min > lam_max, e.g., lam_min = 170, lam_max = -170 
        covers the range from 170 deg to 180 deg and from 
        -180 deg (= 180 deg) to -170 deg.
    

    Returns
    -------
    longitude: ndarray
       Longitudes of the point set in degrees.
    latitude: ndarray
       Latitudes of the point set in degrees.
    """

    # golden ratio
    PHI = (1+np.sqrt(5))/2

    # radius (standard Earth radius [WGS84])
    a_earth = 6378.137
    
    # parameters of model to relate number of points to average distance 
    # on the sphere with radius "a_earth"
    p = np.array([ 1.06113340e+00,  1.81909121e-04,  6.87745830e-01,  2.73178890e-02,
        1.80008060e+00,  1.31022975e+01, -2.89844396e+00])

    # adjust distance for sphere with radius "a"
    D *= a_earth/a
    
    # number-distance model
    c = p[0]+p[1]*D**p[2] + p[3]*np.log(p[4]+np.sin(p[5]*np.log(D)+p[6]))
    
    # convert "c" to half-number of points on a sphere with radius "a"
    N = int(np.round((4*np.pi*a_earth**2/(c*D**2)-1)/2))
    
    # total numbers of point (always odd)
    P = 2*N+1

    lam = []
    phi = []

    # latitude range converted to sines
    sphi_min = np.sin(np.deg2rad(phi_min))
    sphi_max = np.sin(np.deg2rad(phi_max))

    # sine range converted to indices
    iphi_min = int(np.floor(sphi_min*P/2))
    iphi_max = int(np.ceil(sphi_max*P/2))
    
    for i in range(iphi_min,iphi_max):
        if 2*i/P >= sphi_min and 2*i/P <= sphi_max:

            lam_i = np.mod(i,PHI)*360/PHI-180

            if lam_min < lam_max:
                if lam_i >= lam_min and lam_i <= lam_max:
                    lam.append(lam_i)
                    phi_i = np.arcsin(2*i/P)
                    phi.append(phi_i*180/np.pi)
            else:
                if lam_i >= lam_min or lam_i <= lam_max:
                    lam.append(lam_i)
                    phi_i = np.arcsin(2*i/P)
                    phi.append(phi_i*180/np.pi)
    
    return (np.array(lam),np.array(phi))
