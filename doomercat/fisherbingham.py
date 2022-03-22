# Fitting the Fisher-Bingham distribution to data distributed on the
# sphere using the method of moments.
#
# Author: Sebastian von Specht (specht3@uni-potsdam.de),
#         Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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
import scipy.special as sp
from warnings import warn

#
# Main code to compute the Fisher-Bingham central axes
# (and possibly dispersion measures) using the method of moments:
#


def fisher_bingham_mom(X, w):
    """
    Computes

    Arguments:
       X : 3D Euclidean coordinates of the data points on the ellipsoid.

    Returns:
       phi0   : Estimate of central latitude in radians
       lmbdc  : Estimate of central longitude in radians
       alphac : Estimate of azimuth at central point, in radians

    Kent (1982)
    """
    if w is None:
        w = 1.0

    mv = np.sum(w*X,axis=1)/np.sum(w)
    Sv = 1/np.sum(w) * w*X@X.T

    mv_ = mv/np.linalg.norm(mv)

    ct = mv_[0]
    st = np.sqrt(1-ct**2)

    cp = mv_[1]/st
    sp = mv_[2]/st

    H = np.array([
        [ct,-st,0.],
        [st*cp,ct*cp,-sp],
        [st*sp,ct*sp,cp]
    ])

    B = H.T @ Sv @ H

    psi = .5*np.arctan2(2*B[1,2],(B[1,1]-B[2,2]))

    K = np.array([[1,       0,               0],
                  [0, np.cos(psi),-np.sin(psi)],
                  [0, np.sin(psi),np.cos(psi)]])

    G = H @ K

    V = G.T @ mv
    T = G.T @ Sv @ G

    r1 = V[0]

    r2 = T[1,1]-T[2,2]

    g1 = G[:,0] # central axis
    g2 = G[:,1] # equator axis
    # g3 = G[:,2] # pole axis

    # Compute the angles:
    phi0 = np.arcsin(g1[2])
    lmbdc = np.arctan2(g1[1],g1[0])

    n = np.stack([
        -np.cos(lmbdc)*np.sin(phi0),
        -np.sin(lmbdc)*np.sin(phi0),
        np.cos(phi0)
    ])

    alphac = np.arccos(np.sum(n*g2))
    alphac = (alphac + np.pi/2) % np.pi - np.pi/2

    return phi0,lmbdc,alphac
