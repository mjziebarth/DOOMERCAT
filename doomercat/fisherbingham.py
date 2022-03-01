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
# Code for Adam optimization of the dispersion measures of the
# Fisher-Bingham distribution.
#
# The code is currently not in use in the plugin and has not
# been tested extensively.
#

def _fb_norm(ka,be):
    ckb = 0.
    if ka>0 and be>0:
        nv = np.arange(10.)
        ckb = np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1) * be**(2*nv)
                     * (.5*ka)**(-2*nv-.5) * sp.iv(2*nv+.5,ka))
    elif ka > 0. and be == 0.:
        ckb = 4*np.pi/ka*np.sinh(ka)
    elif ka == 0.:
        ckb = 4*np.pi

    return ckb


def _fb_norm_dka(ka,be):
    dckb = 0.
    if ka>0 and be>=0:
        nv = np.arange(10.)
        nvi = 2*nv+.5
        dIdka = .5*(sp.iv(nvi-1,ka)+sp.iv(nvi+1,ka))
        dckb = (  np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1) * be**(2*nv)
                         * .5*(-2*nv-.5)*(.5*ka)**(-2*nv-1.5) * sp.iv(nvi,ka))
                + np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1) * be**(2*nv)
                         * (.5*ka)**(-2*nv-.5) * dIdka))
    elif ka > 0. and be == 0.:
        dckb = -4*np.pi/ka**2*np.sinh(ka) + 4*np.pi/ka*np.cosh(ka)
    elif ka == 0.:
        dckb = 0.

    return dckb


def _fb_norm_dbe(ka,be):
    dckb = 0.
    if ka>0 and be>=0:
        nv = np.arange(10.)
        dckb = np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1)
                      * (2*nv)*be**(2*nv-1) * (.5*ka)**(-2*nv-.5)
                      * sp.iv(2*nv+.5,ka))
    else:
        dckb = 0.

    return dckb


def _fb_norm_dka2(ka,be):
    dckb = 0.
    if ka>0 and be>0:
        nv = np.arange(10.)
        nvi = 2*nv+.5
        dIdka = .5*(sp.iv(nvi-1,ka)+sp.iv(nvi+1,ka))
        d2Idka2 = .25*(sp.iv(nvi-2,ka)+2*sp.iv(nvi,ka) + sp.iv(nvi+2,ka))
        dckb = (  np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1) * be**(2*nv)
                         * .5*(-2*nv-.5)*(.5*ka)**(-2*nv-1.5) * dIdka)
                + np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1) * be**(2*nv)
                         * .5*(-2*nv-1.5)*.5*(-2*nv-.5)*(.5*ka)**(-2*nv-2.5)
                         * sp.iv(nvi,ka))
                + np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1) * be**(2*nv)
                         * (.5*ka)**(-2*nv-.5) * d2Idka2)
                + np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1) * be**(2*nv)
                         * .5*(-2*nv-.5)*(.5*ka)**(-2*nv-1.5) * dIdka))
    elif ka > 0. and be == 0.:
        dckb = (8*np.pi/ka**3*np.sinh(ka) -4*np.pi/ka**2*np.cosh(ka)
                - 4*np.pi/ka**2*np.cosh(ka) + 4*np.pi/ka*np.sinh(ka))
    elif ka == 0.:
        dckb = 4*np.pi/3

    return dckb


def _fb_norm_dkabe(ka,be):
    dckb = 0.
    if ka>0 and be>0:
        nv = np.arange(10.)
        nvi = 2*nv+.5
        dIdka = .5*(sp.iv(nvi-1,ka)+sp.iv(nvi+1,ka))
        dckb = (  np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1)
                         * (2*nv)*be**(2*nv-1) * .5*(-2*nv-.5)
                         *(.5*ka)**(-2*nv-1.5) * sp.iv(nvi,ka))
                + np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1)
                         * (2*nv)*be**(2*nv-1) * (.5*ka)**(-2*nv-.5) * dIdka))
    else:
        dckb = 0

    return dckb

def _fb_norm_dbe2(ka,be):
    dckb = 0.
    if ka>0 and be>0:
        nv = np.arange(10.)
        dckb = np.sum(2*np.pi*sp.gamma(nv+.5)/sp.gamma(nv+1)
                      * (2*nv-1)*(2*nv)*be**(2*nv-2) * (.5*ka)**(-2*nv-.5)
                      * sp.iv(2*nv+.5,ka))
    else:
        dckb = np.nan


    return dckb


def _compute_ka0_be0(r1, r2):
    """
    Fits the dispersion parameters kappa and beta using the
    method of moments due to Kent (1982). To solve the nonlinear
    equation, the Adam method is used.

    Kent, J. T. (1982). The Fisher-Bingham Distribution on the Sphere.
       Journal of the Royal Statistical Society. Series B (Methodological),
       44(1), 71–80. http://www.jstor.org/stable/2984712
    """
    warn("Calling compute_ka0_be0. This code has not been extensively "
         "tested.")

    # Starting values:
    ka0 = 1/(1-r1)
    be0 = .01*ka0

    # ADAM memory:
    be1 = .9
    be2 = .999
    eps = 1e-10
    ma = 0.
    va = 0.

    def funS(r1,r2,ka0,be0):
        if ka0>0. and be0>0.:
            c = _fb_norm(ka0,be0)
            ck = _fb_norm_dka(ka0,be0)
            cb = _fb_norm_dbe(ka0,be0)

            return (r1-ck/c)**2 + (r2-cb/c)**2
        else:
            return np.nan


    # Normal vector giving the "sensible" distribution
    # parameter space constraint:
    cv = np.array([0.5,-1.0])
    cv /= np.linalg.norm(cv)
    gold = None
    goldold = None

    for i in range(1001):
        if i == 1000:
            raise RuntimeError("Failed to converge when estimating "
                               "Fisher-Bingham parameters.")

        # Compute the value and derivatives:
        c = _fb_norm(ka0,be0)
        ck = _fb_norm_dka(ka0,be0)
        cb = _fb_norm_dbe(ka0,be0)
        ckk = _fb_norm_dka2(ka0,be0)
        ckb = _fb_norm_dkabe(ka0,be0)
        cbb = _fb_norm_dbe2(ka0,be0)

        S = (r1-ck/c)**2 + (r2-cb/c)**2
        dSdk =   2*(r1-ck/c)*-(ckk*c-ck**2)/c**2 \
               + 2*(r2-cb/c)*-(ckb*c-ck*cb)/c**2
        dSdb =   2*(r1-ck/c)*-(ckb*c-ck*cb)/c**2 \
               + 2*(r2-cb/c)*-(cbb*c-cb**2)/c**2

        g = np.array([dSdk,dSdb])
        ma = be1*ma + (1-be1)*g
        va = be2*va + (1-be2)*g**2
        mat = ma/(1-be1**(i+1))
        vat = va/(1-be2**(i+1))

        adam = mat/(np.sqrt(vat)+eps)

        # ensure positivity of parameter estimates
        while (ka0 - adam[0]) <= 0.:
            ma /= 2
            va /= 2
            adam[0] /= 2

        while (be0 - adam[1]) <= 0.:
            ma /= 2
            va /= 2
            adam[1] /= 2

        # ensure the condition ka>2*be. If this is not met, the FB5 still
        # exists but is bimodal.
        # In this step, adam is simply redirected by Gram-Schmidt projection
        # rather than slowed down.
        if (ka0 - adam[0]) < 2*(be0 - adam[1]):
            adam -= np.dot(cv,adam)*cv
            ma -= np.dot(cv,ma)*cv
            va -= np.dot(cv,va)*cv

        # ensure decrease of cost function by slowing down adam
        while funS(r1,r2,ka0 - adam[0],be0 - adam[1]) > S:
            adam /= 2.


        # Stopping condition:
        if goldold is not None and np.linalg.norm(goldold-g) < 1e-15:
            break

        # Memorize latest history:
        goldold = gold
        gold = g

        ka0 -= adam[0]
        be0 -= adam[1]

    return ka0, be0


#
# Main code to compute the Fisher-Bingham central axes
# (and possibly dispersion measures) using the method of moments:
#


def fisher_bingham_mom(lon, lat, w, compute_dispersion=False):
    """
    Computes

    Returns:
       g1    : Central axis
       g2    : Equator axis
       g3    : Pole axis
       kappa : Dispersion
       beta  : Dispersion

    Kent (1982)
    """
    if w is None:
        w = 1.0
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    v1 = np.stack((np.cos(lon) * np.cos(lat),
                   np.sin(lon) * np.cos(lat),
                   np.sin(lat)))

    mv = np.sum(w*v1,axis=1)/np.sum(w)
    Sv = 1/np.sum(w) * w * v1 @ v1.T

    mv_ = mv/np.linalg.norm(mv)

    ct = mv_[0]
    st = np.sqrt(1-ct**2)

    if st == 0.0:
        # define cp & sp to zero:
        cp = 0.0
        sp = 0.0
    else:
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
    g3 = G[:,2] # pole axis

    if not compute_dispersion:
        return g1, g2, g3

    # Now fit the dispersion parameters:
    ka, be = _compute_ka0_be0(r1, r2)

    return g1, g2, g3, ka, be
