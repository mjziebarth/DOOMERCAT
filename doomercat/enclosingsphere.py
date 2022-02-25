# Iterative search for the center of the smalles enclosing sphere for a
# set of points on an ellipsoid.
#
# Author: Sebastian von Specht (specht3@uni-potsdam.de),
#         Malte Ziebarth
#
# Copyright (C) 2022 Sebastian von Specht,
#                    Deutsches GeoForschungsZentrum Potsdam
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

def BoundingSphere(lon,lat,a,f,N=100):
    '''
    Iterative search for center of bounding sphere for points on an ellipsoid in
    geographic coordinates.
    lon,lat: geographic coordinates in degree
    a: major semi-axis in m; used for termination criterion only
    f: flattening of the ellipsoid
    N: maximum number of iterations

    The algorithm terminates latest after N iterations or when the difference in
    center coordinates is less or equal 1 m.

    Returns geographic coordinates of the bounding sphere center projected onto
    the ellipsoid.
    '''

    e2 = 2*f-f**2
    Nlat = 1/np.sqrt(1-e2*np.sin(lat)**2)

    x = np.stack([Nlat*np.cos(lon)*np.cos(lat),
                  Nlat*np.sin(lon)*np.cos(lat),
                  Nlat*(1-e2)*np.sin(lat)]).T

    xnorm = x/np.sqrt(np.sum(x**2,axis=1))[:,np.newaxis]

    xc = np.sum(x,axis=0)
    xc /= np.sqrt(np.sum(xc**2))

    R = np.sqrt(np.max(np.sum((xc-x)**2,axis=1)))

    fac = np.array([.5,1.,2.])

    al = 1e-2

    Rold = 1.
    eps = 1./a # precision down to a real-world meter
    i = 0
    while (np.abs(Rold-R)>eps) and (i<N):

        imax = np.argmax(np.sum((xc-x)**2,axis=1))
        dRdxc = xc-x[imax]

        R3 = np.zeros_like(fac)
        for j in range(fac.size):
            neoxc = xc-fac[j]*al*dRdxc
            R3[j] = np.max(np.sum((neoxc-x)**2,axis=1))

        j = np.argmin(R3)

        al *= fac[j]
        xc -= al*dRdxc

        Rold = R
        R = np.sqrt(np.max(np.sum((xc-x)**2,axis=1)))

        i += 1

    xcnorm = xc/np.linalg.norm(xc)
    phic = np.arctan(np.tan(np.arcsin(xcnorm[2]))/(1-e2))
    lamc = np.arctan2(xcnorm[0],xcnorm[1])

    return np.rad2deg(lamc),np.rad2deg(phic)
