# Optimization of the Hotine oblique Mercator projection in Python.
#
# Author: Sebastian von Specht (specht3@uni-potsdam.de),
#         Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022-2024 Sebastian von Specht,
#               2022      Deutsches GeoForschungsZentrum Potsdam,
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
import sys
from math import degrees, isinf
from .geometry import _lola_aux_2_xyz, _Rx, _Ry, _Rz
from .initial import initial_k0
from ._typing import ndarray64
from typing import Optional


class HotineResultPy:
    """
    Result of the optimization.
    """
    def __init__(self, cost, lonc, lat_0, alpha, k0, steps, f, error_flag):
        self.cost = cost
        self.lonc = lonc
        self.lat_0 = lat_0
        self.alpha = alpha
        self.k0 = k0
        self.N = 1
        self.steps = steps
        self.f = f
        self.mode = 2
        self.error_flag = error_flag


def _B(phi0,e):
    return np.sqrt(1+(e**2*np.cos(phi0)**4)/(1-e**2))

def _A(B,k0,e,phi0):
    return B*k0*np.sqrt(1-e**2) / (1-e**2*np.sin(phi0)**2)

def _t0(phi0,e):
    return np.tan(np.pi/4-phi0/2) \
           / ((1-e*np.sin(phi0))/(1+e*np.sin(phi0)))**(e/2)

def _t(phi,e):
    return np.tan(np.pi/4-phi/2) / ((1-e*np.sin(phi))/(1+e*np.sin(phi)))**(e/2)

def _D(B,e,phi0):
    return B*np.sqrt(1-e**2) / (np.cos(phi0)*np.sqrt(1-e**2*np.sin(phi0)**2))

def _F(D,phi0):
    return D + np.sign(phi0)*np.sqrt(D**2-1)

def _E(F,t0,B):
    return F*t0**B

def _G(F):
    return (F-1/F)/2

def _gamma0(alphac,D):
    return np.arcsin(np.sin(alphac)/D)

def _lmbd0(lmbdc,G,gamma0,B,alphac=0,phi0=0):
    # the limit of G*tan(gamma0) as alphac-> +/- pi/2 is +/- 1. However, for
    # large G and/or tan(gamma0) round-off errors may render the expression >|1|
    X = G*np.tan(gamma0)
    X = 1. if X > 1. else X
    X = -1. if X < -1. else X
    return lmbdc-np.arcsin(X)/B


def _Q(E,t,B):
    return E/t**B

def _S(Q):
    return (Q-1/Q)/2

def _T(Q):
    return (Q+1/Q)/2

def _V(B,lmbd,lmbd0):
    return np.sin(B*(lmbd-lmbd0))

def _U(V,gamma0,S,T):
    return (-V*np.cos(gamma0)+S*np.sin(gamma0))/T

def _u(A,S,gamma0,V,B,lmbd,lmbd0):
    return A*np.arctan2((S*np.cos(gamma0)+V*np.sin(gamma0)),
                         np.cos(B*(lmbd-lmbd0)))/B

def _h(dudp,dvdp,e,phi,k0):
    return np.sqrt(dudp**2 + dvdp**2) * (1-e**2*np.sin(phi)**2)**1.5 / (1-e**2)

def _k(dudl,dvdl,e,phi,k0):
    return np.sqrt(dudl**2 + dvdl**2) * (1-e**2*np.sin(phi)**2)**0.5 \
           / np.cos(phi)

def _ks(A,B,u,e,phi,lmbd,lmbd0):
    return A*np.cos(B*u/A)*np.sqrt(1-e**2*np.sin(phi)**2)\
           /(np.cos(phi)*np.cos(B*(lmbd-lmbd0)))




def _f_d_k_cse(lmbd,phi,phi0,lmbdc,alphac,k0,e,noJ=False):

    if np.abs(alphac) >= np.deg2rad((90-1/3600)):
        alphac = np.deg2rad((90-1/3600))*np.sign(alphac)

    B = _B(phi0, e)
    A = _A(B, k0, e, phi0)
    D = _D(B,e,phi0)
    F = _F(D,phi0)
    t0 = _t0(phi0,e)
    t = _t(phi,e)
    E = _E(F,t0,B)
    G = _G(F)

    gamma0 = _gamma0(alphac,D)
    lmbd0 = _lmbd0(lmbdc,G,gamma0,B)
    Q = _Q(E,t,B)
    S = _S(Q)
    V = _V(B,lmbd,lmbd0)
    u = _u(A,S,gamma0,V,B,lmbd,lmbd0)

    ks = _ks(A,B,u,e,phi,lmbd,lmbd0)

    if noJ:
        return ks
    else:

        x0 = np.cos(phi)
        x1 = lmbd - lmbd0
        x2 = B*x1
        x3 = np.sin(x2)
        x4 = A**(-1)
        x5 = B*u
        x6 = x4*x5
        x7 = np.sin(x6)
        x8 = e**2
        x9 = np.sqrt(-x8*np.sin(phi)**2 + 1)
        x10 = x7*x9
        x11 = np.cos(x2)
        x12 = x11**(-1)
        x13 = x12/x0
        x14 = np.cos(phi0)
        x15 = x14**3
        x16 = np.sin(phi0)
        x17 = x16*x8
        x18 = 2*x17/(B*(x8 - 1))
        x19 = x15*x18
        x20 = 2*phi0
        x21 = -x8
        x22 = x21 + x8*np.cos(x20) + 2
        x23 = x16**2*x8
        x24 = (x23 - 1)**(-1)
        x25 = np.sqrt(x21 + 1)
        x26 = x18*x25
        x27 = 2*A*x8*np.sin(x20)/x22 - k0*x15*x24*x26
        x28 = A*np.cos(x6)
        x29 = x5*x7
        x30 = x13*x4*x9
        x31 = np.tan(gamma0)
        x32 = B**(-1)
        x33 = np.sqrt(-G**2*x31**2 + 1)
        x34 = x33**(-1)
        x35 = x32*x34
        x36 = D**2
        x37 = x36**(-1)
        x38 = np.sin(alphac)
        x39 = 1/np.sqrt(-x37*x38**2 + 1)
        x40 = np.cos(gamma0)
        x41 = x40**2
        x42 = x39/x41
        x43 = 1 - x23
        x44 = B*x25
        x45 = D*np.tan(phi0) + x14**2*x26/np.sqrt(x43) + x17*x44/x43**(3/2)
        x46 = x37*x38*x45
        x47 = np.sqrt(x36 - 1)
        x48 = x45*((1) if (phi0 == 0) else (D*phi0/(x47*abs(phi0)) + 1))
        x49 = F**2
        x50 = G*x35*x42*x46 - 1/2*x31*x35*x48*(x49 + 1)/x49 \
              + x19*np.arcsin(G*x31)/B**2
        x51 = B*x50
        x52 = ks*np.tan(x2)
        x53 = np.sin(gamma0)
        x54 = S*x40 + V*x53
        x55 = np.arctan2(x54,x11)
        x56 = x3*x54
        x57 = A*x56
        x58 = x11**2
        x59 = x54**2 + x58
        x60 = x59**(-1)
        x61 = x32*x60
        x62 = x11*(S*x53 - V*x40)
        x63 = A*x61
        x64 = Q**2
        x65 = e*x16
        x66 = (x65 + 1)**(-1)
        x67 = (x66*(1 - x65))**((1/2)*e)
        x68 = x11*x63
        x69 = B*x13
        x70 = x0*x11*x52*x59
        x71 = x53*x58 + x56
        x72 = A*x10


        f_dphi0 = -x10*x69*(
                x19*x61*(-u*x59 + x1*x57) + x27*x32*x55
                + x39*x46*x62*x63 + (1/2)*x40*x68*(x64 + 1)*(-Q*x19*np.log(t)
                    + t**(-B)*(B*E*x66*(-t0*x14*x67*x8 + (1/2)*x22/(x16 + 1))
                        /(t0*x67*(x65 - 1))
                    + E*x19*np.log(t0) + t0**B*x48))/x64
                - x50*x57*x60 + x53*x68*(x1*x11*x19 - x11*x51)
            ) + x13*x19*(ks*x0*x1*x3 - u*x10) + x27*x30*(x28 + x29) - x51*x52
        f_dlmbdc = x60*x69*(-x70 + x71*x72)
        f_dalphac = x13*x34*x42*x60*(
                G*x70 - x72*(G*x71 - x33*x41*x62)
            )*np.cos(alphac)/D
        f_dk0 = x24*x30*x44*(A*x55*x7 - x28 - x29)


        J = np.stack([f_dphi0,
                      f_dlmbdc,
                      f_dalphac,
                      f_dk0]).T

        return (ks,J)


def _confine(p):

    # winding number for phi (p[0])
    wn = np.floor((p[0]+np.pi/2) / np.pi).astype(int)

    p[0] = (((p[0]+np.pi/2) % np.pi) - np.pi/2)*(-1.)**wn

    if (wn % 2) == 1:
        p[1] -= np.sign(p[0])*np.pi
        p[2] += np.pi

    p[1] = ((p[1]+np.pi) % (2*np.pi)) - np.pi
    p[2] = (p[2] + np.pi/2) % np.pi - np.pi/2

    return p

def _subbatch(X,p,xper=.025,nper=50):
    X0 = (_Rz(p[1]) @ _Ry(p[0]) @ _Rx(p[2]-np.pi/2)).T @ X
    Z0 = np.abs(X0[2])

    I = np.ones_like(Z0,dtype=bool)

    if Z0.size*(xper)>nper:
        I1 = int(xper*Z0.size)
        I2 = int((1-xper)*Z0.size)
        I[I1:I2] = False
        I_batch = np.argpartition(Z0,(I1,I2))[I]
    else:
        I_batch = np.arange(Z0.size)

    return I_batch

def lm_adamax_optimize(
        lon: ndarray64,
        lat: ndarray64,
        h: ndarray64,
        wdata: ndarray64,
        phi0: float,
        lmbdc: float,
        alphac: float,
        k0: float,
        a: float,
        f: float,
        pnorm: int = 2,
        Niter: int = 100,
        Nmax_pre_adamax: int = 50,
        diagnostics: bool = False,
        k0_ap: Optional[float] = None,
        k0_ap_std: Optional[float] = None,
        debug_prints: bool = False
    ):

    # normalize data weights to sum(wdata) = number of data points
    wdata /= wdata.sum()

    # Compute the local destination scale factor:
    e2 = 2*f - f**2
    one_over_k_e = (a+h) / a

    e = np.sqrt(e2)

    p = np.array([phi0,lmbdc,alphac,k0])

    x = 2.**np.arange(-1,2.)
    al = 1e-4
    la = 1e-1

    x0 = np.concatenate((np.array([-1.]), 2.**np.arange(-1,2.)))
    alm = .9
    bem = .999
    mm = 0.
    um = 0.
    eps = 1e-10
    ti = 1
    X = None
    iswitch = None

    lminfloat = np.log10(sys.float_info.min)

    pnorm_p0 = 10.
    is_p2opt = False
    switch_to_p0 = True
    error_flag = None

    Ssd = 1.
    Ssd_th = 1e-9
    Nsd = 10
    # If Nmax_pre_adamax < Nsd, the exit equality check further
    # below never triggers.
    Nmax_pre_adamax = max(Nmax_pre_adamax, 4*Nsd)
    Sv = []
    S33 = np.zeros((1,1)) + np.nan
    Ij = Ik = 0

    if diagnostics:
        alv = []
        lav = []
        P = []
        L = []

    if k0_ap is None:
        P_ap = np.zeros((4,4))
        v_ap = np.zeros(4)
    else:
        if k0_ap_std is None:
            print('standard deviation of a priori k0 not specified. Terminate.')
            return None

        P_ap = np.zeros((4,4))
        P_ap[3,3] = 1/k0_ap_std**2
        v_ap = np.array([0.,0.,0.,k0_ap])

    for i in range(Niter):

        if not isinf(pnorm) or not is_p2opt:
            fk,J = _f_d_k_cse(lon,lat,p[0],p[1],p[2],p[3],e)
            yk = one_over_k_e[:]
        else:
            I_batch = _subbatch(X,p)

            fk = _f_d_k_cse(lon[I_batch],lat[I_batch],p[0],p[1],
                           p[2],p[3],e,noJ = True)
            yk = one_over_k_e[I_batch]


        if isinf(pnorm) and is_p2opt:
            res = np.abs(fk-yk)
            iresmax = np.argmax(res)
            fk,J = _f_d_k_cse(lon[I_batch][iresmax],lat[I_batch][iresmax],
                             p[0],p[1],p[2],p[3],e)
            J.shape = (1,4)
            yk = one_over_k_e[I_batch][iresmax]

        if isinf(pnorm):
            if is_p2opt:
                w = np.array([1.])
            else:
                w = np.abs(fk-yk)**(pnorm_p0-2)
                w *= wdata
        elif pnorm > 0. and pnorm <= 1:
            w = (np.abs(fk-yk)+1e-15)**(pnorm-2)
            w *= wdata
        elif pnorm == 2:
            w = wdata*1.
        else:
            w = np.abs(fk-yk)**(pnorm-2)
            w *= wdata

        w /= w.sum()

        if isinf(pnorm) and is_p2opt:

            Theta =  (np.sign(v_ap[3]-p[3])+1)/2
            dp = J*np.sign(fk-yk) + 2 * P_ap @ (p-v_ap) * -Theta

            mm = alm*mm+(1-alm)*dp
            um = np.maximum(bem*um,np.abs(dp))
            ti += 1

            neodp = np.zeros((1,4))
            neodp[0] = 1/(1-alm**ti) * mm/um

            neop = np.zeros((x0.size,4))
            S33 = np.zeros((x0.size,1))+np.inf

            for k in range(x0.size):
                neop[k] = _confine(p - x0[k]*al*neodp[0])

                I_batch = _subbatch(X,neop[k])
                neofk = _f_d_k_cse(lon[I_batch], lat[I_batch],
                                  neop[k,0], neop[k,1], neop[k,2],neop[k,3],
                                  e, noJ=True)
                yk = one_over_k_e[I_batch]

                Theta =  (np.sign(v_ap[3]-neop[k,3])+1)/2
                S33[k,0] = (np.abs(w*(neofk-yk)).max()) \
                           + P_ap[3,3]*(neop[k,3]-v_ap[3])**2 * Theta
                # S33[k,0] = np.maximum(np.abs(neofk-yk).max(), (np.sqrt(P_ap[3,3])*np.abs(neop[k,3]-v_ap[3]) * Theta).max())

            I = np.argmin(S33)
            al *= x0[I]

            p1 = neop[I]


            Ij,Ik = I,0 # for diagnostics format
            if np.any(np.isnan(p1)) or np.any(np.isinf(p1)):
                # Exit before breaking parameters:
                Sv.append(S33[Ij,Ik])
                if diagnostics:
                    alv.append(al)
                    lav.append(la)
                    P.append(p*1.)
                error_flag = "NaNParameter"
                break
            p = p1*1

        else:

            JTJ = (J.T*w)@J
            D = np.diag(np.diag(JTJ)) # Preconditioner

            S33 = np.zeros((x.size,x.size))
            neop = np.zeros((x.size,x.size,4))
            neodp = np.zeros((x.size,4))

            Theta =  (np.sign(v_ap[3]-p[3])+1)/2

            for j in range(x.size):

                try:
                    neodp[j] = np.linalg.solve(JTJ + x[j] * la * D
                                                 + P_ap * -Theta,
                                              (J.T*w)@(fk-yk)
                                                 + P_ap @ (p-v_ap) * -Theta)
                except np.linalg.LinAlgError:
                    error_flag = "LinAlgError"
                    break

                neodp[j] = neodp[j]

                for k in range(x.size):
                    neop[j,k] = _confine(p-x[k]*al*neodp[j])

                    neofk = _f_d_k_cse(lon,lat,neop[j,k,0],neop[j,k,1],
                                      neop[j,k,2],neop[j,k,3],e,noJ=True)
                    yk = one_over_k_e[:]

                    Theta = (np.sign(v_ap[3]-neop[j,k,3])+1)/2
                    if isinf(pnorm):
                        S33[j,k] = np.sum(wdata*np.abs(neofk-yk)**pnorm_p0) \
                                    + P_ap[3,3]*(neop[j,k,3]-v_ap[3])**2 * Theta
                    else:
                        S33[j,k] = np.sum(wdata*np.abs(neofk-yk)**pnorm) \
                                    + P_ap[3,3]*(neop[j,k,3]-v_ap[3])**2 * Theta

                    if np.isnan(S33[j,k]):
                        S33[j,k] = np.inf


            I = np.argmin(S33)
            Ij,Ik = np.unravel_index(I,S33.shape)

            al *= x[Ik]
            la *= x[Ij]

            p1 = neop[Ij,Ik]
            if np.any(np.isnan(p1)) or np.any(np.isinf(p1)):
                # Exit before breaking parameters:
                Sv.append(S33[Ij,Ik])
                if diagnostics:
                    alv.append(al)
                    lav.append(la)
                    P.append(p*1.)
                error_flag = "NaNParameter"
                break
            p = p1

        Sv.append(S33[Ij,Ik])
        if debug_prints:
            print(Sv[-1],p)
        if diagnostics:
            alv.append(al)
            lav.append(la)
            P.append(p*1.)

        if error_flag is not None:
            break

        if i >= Nsd:

            Ssd = np.std(np.log(Sv[-Nsd:]),ddof=1)
            # print('var',np.log(Sv[-Nsd:]),Ssd)

            # Especially for high pnorm, it might happen that the cost
            # evaluation of S33 gives 0.0 for all data points if the distortions
            # are already fairly small.
            # Catch this case here:
            #if min(Sv[-Nsd:]) == 0.0:
            if np.isnan(Ssd):
                Ssd = 0.0

            if not isinf(pnorm) and Ssd<Ssd_th:
                break
            elif isinf(pnorm) and (Ssd<Ssd_th or i == Nmax_pre_adamax):
                is_p2opt = True

                if switch_to_p0:
                    X = _lola_aux_2_xyz(np.rad2deg(lon),np.rad2deg(lat),f)
                    p[3] = initial_k0(p[0], p[1], p[2], X, wdata, np.inf,
                                      is_p2opt, pnorm_p0)

                    switch_to_p0 = False
                    iswitch = i*1
                    al = 1e-6
                    la = .1
                    mm = 0
                    um = 0
                else:
                    break

            elif Ssd<Ssd_th:
                break

    if diagnostics:
        alv = np.array(alv)
        lav = np.array(lav)
        P = np.array(P)
        Sv = np.array(Sv)
        return p,alv,lav,Sv,P
    else:
        return HotineResultPy(cost=S33[Ij,Ik], lonc=degrees(p[1]),
                              lat_0=degrees(p[0]), alpha=degrees(p[2]),
                              k0=p[3], steps=i, f=f, error_flag=error_flag)
