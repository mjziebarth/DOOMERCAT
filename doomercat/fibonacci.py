# Fibonacci lattice to generate uniformly distributed points on a sphere.
#
# Author: Sebastian von Specht,
#         Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022-2024 Sebastian von Specht,
#               2022 Technical University of Munich
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

def fibonacci_lattice(D: float, a: float = 6378137.0):
    """
    Returns (nearly) equally spaced points with a Fibonacci lattice
    on a sphere.

    Parameters
    ----------
    D : float
       Average distance of points on the sphere (unit follows from unit
       of radius, default is in meter).
    a : float, optional
       Radius of the sphere. Default is the WGS84 in meter.

    Returns
    -------
    longitude: ndarray
       Longitudes of the point set in degrees.
    latitude: ndarray
       Latitudes of the point set in degrees.
    """

    # golden ratio
    PHI = (1+np.sqrt(5))/2

    # parameters of model to relate number of points to average distance on the sphere
    p = np.array([ 1.03033653e+00,  9.08039046e-05,  1.57110979e+00,  1.29553736e-02,
            1.78518128e+00,  3.01690251e+01, -2.89932149e+00])

    # number-distance model
    ldf = np.log10(D)
    c = p[0]+p[1]*np.exp(p[2]*ldf) + p[3]*np.log(p[4]+np.sin(p[5]*ldf+p[6]))

    # convert "c" to half-number of points on a sphere with radius "a"
    N = np.round((4*np.pi*a**2/(c*D)**2-1)/2).astype(int)

    # total numbers of point (always odd)
    P = 2*N+1

    i = np.arange(-N,N+1)

    # latitude (phi), longitude (lam) in degrees
    phi = np.arcsin(2*i/P)*180/np.pi
    lam = np.mod(i,PHI)*360/PHI

    return (lam,phi)
