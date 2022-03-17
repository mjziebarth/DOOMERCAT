# Optimization of the Hotine oblique Mercator projection in Python.
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
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import sys

def Rx(a):
    R = np.array([
        [1,0,0.],
        [0.,np.cos(a),np.sin(a)],
        [0.,-np.sin(a),np.cos(a)]
    ])
    return R

def Ry(a):
    R = np.array([
        [np.cos(a),0.,-np.sin(a)],
        [0,1,0.],
        [np.sin(a),0.,np.cos(a)]
    ])
    return R

def Rz(a):
    R = np.array([
        [np.cos(a),-np.sin(a),0.],
        [np.sin(a),np.cos(a),0.],
        [0,0.,1]
    ])
    return R

def lola2xyz(lo,la,f):
    e2 = 2*f-f**2

    N = 1/np.sqrt(1-e2*np.sin(la))

    X = np.zeros((3,lon.size))

    X[0] = N*np.cos(lo)*np.cos(la)
    X[1] = N*np.sin(lo)*np.cos(la)
    X[2] = (N*(1-e2))*np.sin(la)

    return X


def _B(phi0,e):
    return np.sqrt(1+(e**2*np.cos(phi0)**4)/(1-e**2))

def _A(B,k0,e,phi0):
    return B*k0*np.sqrt(1-e**2) / (1-e**2*np.sin(phi0)**2)

def _t0(phi0,e):
    return np.tan(np.pi/4-phi0/2) / ((1-e*np.sin(phi0))/(1+e*np.sin(phi0)))**(e/2)

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
    # the limit of G*tan(gamma0) as alphac-> +/- pi/2 is +/- 1. However, for large G and/or tan(gamma0) round-off errors may render the expression >|1|
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

def _v(A,U,B):
    return A*log((1-U)/(1+U))/(2*B)

def _u(A,S,gamma0,V,B,lmbd,lmbd0):
    return A*np.arctan2((S*np.cos(gamma0)+V*np.sin(gamma0)),np.cos(B*(lmbd-lmbd0)))/B

def _h(dudp,dvdp,e,phi,k0):
    return np.sqrt(dudp**2 + dvdp**2) * (1-e**2*np.sin(phi)**2)**1.5 / (1-e**2)

def _k(dudl,dvdl,e,phi,k0):
    return np.sqrt(dudl**2 + dvdl**2) * (1-e**2*np.sin(phi)**2)**0.5 / np.cos(phi)

def _ks(A,B,u,e,phi,lmbd,lmbd0):
    return A*np.cos(B*u/A)*np.sqrt(1-e**2*np.sin(phi)**2)/(np.cos(phi)*np.cos(B*(lmbd-lmbd0)))




def f_d_k_cse(lmbd,phi,phi0,lmbdc,alphac,k0,e,noJ=False):

    if np.abs(alphac) >= np.deg2rad((90-1/3600)):
        alphac = np.deg2rad((90-1/3600))*np.sign(alphac)

    B = _B(phi0, e)
    A = _A(B, k0, e, phi0)
    D = _D(B,e,phi0)
    F = _F(D,phi0)
    t0 = _t(phi0,e)
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
        x50 = G*x35*x42*x46 - 1/2*x31*x35*x48*(x49 + 1)/x49 + x19*np.arcsin(G*x31)/B**2
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


        f_dphi0 = -x10*x69*(x19*x61*(-u*x59 + x1*x57) + x27*x32*x55 + x39*x46*x62*x63 + (1/2)*x40*x68*(x64 + 1)*(-Q*x19*np.log(t) + t**(-B)*(B*E*x66*(-t0*x14*x67*x8 + (1/2)*x22/(x16 + 1))/(t0*x67*(x65 - 1)) + E*x19*np.log(t0) + t0**B*x48))/x64 - x50*x57*x60 + x53*x68*(x1*x11*x19 - x11*x51)) + x13*x19*(ks*x0*x1*x3 - u*x10) + x27*x30*(x28 + x29) - x51*x52
        f_dlmbdc = x60*x69*(-x70 + x71*x72)
        f_dalphac = x13*x34*x42*x60*(G*x70 - x72*(G*x71 - x33*x41*x62))*np.cos(alphac)/D
        f_dk0 = x24*x30*x44*(A*x55*x7 - x28 - x29)


        J = np.stack([f_dphi0,f_dlmbdc,f_dalphac,f_dk0]).T

        return (ks,J)


def confine(p):

    # winding number for phi (p[0])
    wn = np.floor((p[0]+np.pi/2) / np.pi).astype(int)

    p[0] = (((p[0]+np.pi/2) % np.pi) - np.pi/2)*(-1.)**wn

    if (wn % 2) == 1:
        p[1] -= np.sign(p[0])*np.pi
        p[2] += np.pi

    p[1] = ((p[1]+np.pi) % (2*np.pi)) - np.pi
    p[2] = (p[2] + np.pi/2) % np.pi - np.pi/2

    return p

def initalphac(lon,lat,phi0,lmbdc):

    a = np.stack([
        np.cos(lmbdc)*np.cos(phi0),
        np.sin(lmbdc)*np.cos(phi0),
        np.sin(phi0)
    ])

    x = np.stack([
        np.cos(lon)*np.cos(lat),
        np.sin(lon)*np.cos(lat),
        np.sin(lat)
    ])

    b = np.sum(x,axis=1)
    b /= np.linalg.norm(b)

    lat_b = np.arcsin(b[2])
    lon_b = np.arctan2(b[1],b[0])

    n = np.stack([
        -np.cos(lmbdc)*np.sin(phi0),
        -np.sin(lmbdc)*np.sin(phi0),
        np.cos(phi0)
    ])

    h = 1/np.dot(a,b)

    c = h*b-a
    c /= np.linalg.norm(c)

    alphac = np.arccos(np.sum(n*c))
    alphac = (alphac + np.pi/2) % np.pi - np.pi/2

    return alphac

def FBaxes_initials(X,w=None):

    if w is None:
        w = 1.

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

    B = H.T@Sv@H
    psi = .5*np.arctan2(2*B[1,2],(B[1,1]-B[2,2]))
    K = np.array([[1,0,0],[0,np.cos(psi),-np.sin(psi)],[0,np.sin(psi),np.cos(psi)]])
    G = H@K

    g1 = G[:,0] # central axis, basis
    g2 = G[:,1] # equator axis
    # g3 = G[:,2] # pole axis

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

    # basis for dispersion estimators
#     V = G.T@mv
#     T = G.T@Sv@G

#     r1 = V[0]
#     r2 = T[1,1]-T[2,2]


def initk0(phi0, lmbdc, alphac, X, wdata, pnorm, is_p2opt):

    X0 = (Rz(lmbdc)@Ry(phi0)@Rx(alphac-np.pi/2)).T@X

    plt.figure()
    plt.plot(X0[1],X0[2],'.')
    plt.show()

    k0v = np.sqrt(1-(X0[2])**2)
    k0_init = np.mean(k0v)

    for i in range(100):

        if pnorm == 0  and is_p2opt:
            w = np.zeros_like(k0v)
            I = [np.argmax(k0_init-k0v), np.argmin(k0_init-k0v)]
            w[I] = 1.
        elif pnorm > 0. and pnorm <= 1:
            w = (np.abs(k0_init-k0v)+1e-15)**(pnorm-2)
        elif pnorm == 2 and not is_p2opt:
            w = np.ones_like(k0v)
        else:
            w = np.abs(k0_init-k0v)**(pnorm-2)

        k0_init -= .1* np.sum(w*wdata * (k0_init-k0v))/np.sum(w*wdata)

    return k0_init


def grad(lon,lat,wdata,phi0,lmbdc,alphac,k0,f,pnorm=2,Niter = 100,
         diagnostics=False, k0_ap=None, k0_ap_std=None):

    # normalize data weights to sum(wdata) = number of data points
    wdata /= wdata.mean()

    e = np.sqrt(2*f-f**2)
    X = lola2xyz(lon,lat,f)

    p = np.array([phi0,lmbdc,alphac,k0])

    x = 2.**np.arange(-1,2.)
    al = 1e-2
    la = 1e-2

    o = 1.

    alm = .9
    bem = .999
    mm = 0.
    vm = 0.
    eps = 1e-10
    ti = 1
    signS = np.zeros(Niter)
    signS_N = 50
    signS_i = 0

    lminfloat = np.log10(sys.float_info.min)

    is_p2opt = False
    switch_to_p0 = True

    p[3] = initk0(p[0],p[1],p[2],X,wdata,pnorm,is_p2opt)
    print('k0 init',p[3])

    if diagnostics:
        alv = []
        lav = []
        Sv = []
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

    print(v_ap,P_ap,k0_ap_std,k0_ap)

    for i in range(Niter):

        if pnorm != 0 or not is_p2opt:
            fk,J = f_d_k_cse(lon,lat,p[0],p[1],p[2],p[3],e)
        else:
            X0 = (Rz(p[1])@Ry(p[0])@Rx(p[2]-np.pi/2)).T@X
            Z0 = np.abs(X0[2])

            I = np.ones_like(Z0,dtype=bool)
            xper = .025
            I[int(xper*Z0.size):int((1-xper)*Z0.size)] = False
            I_batch = np.argsort(Z0)[I]

            fk = f_d_k_cse(lon[I_batch],lat[I_batch],p[0],p[1],p[2],p[3],e,noJ = True)



        if pnorm == 0. and is_p2opt:
            res = np.abs(fk-1)
            iresmax = np.argmax(res)
            fk,J = f_d_k_cse(lon[I_batch][iresmax],lat[I_batch][iresmax],p[0],p[1],p[2],p[3],e)
            J.shape = (1,4)


        if pnorm == 0  and is_p2opt:
            w = np.array([1.])
        elif pnorm > 0. and pnorm <= 1:
            w = (np.abs(fk-1)+1e-15)**(pnorm-2)
            w *= wdata
        elif pnorm == 2 and not is_p2opt:
            w = np.ones_like(fk)
            w *= wdata
        else:
            w = np.abs(fk-1)**(pnorm-2)
            w *= wdata

        w /= w.mean()


        if pnorm == 0 and is_p2opt:

            if signS_i == 0:
                al = 1e-3

#             if signS_i > signS_N:
#                 signS_m = np.sign(np.diff(signS[signS_i-signS_N:signS_i])).mean()

#                 if signS_m > -.5:
#                     al *= .9
#                 else:
#                     al /= .9

                # print(signS_m)


            dp = (J*np.sign(fk-1)).flatten()
            mm = alm*mm + (1-alm)*dp
            vm = bem*vm + (1-bem)*dp**2
            mt = mm/(1.-alm**ti)
            vt = vm/(1.-bem**ti)
            ti += 1.

            neodp = np.zeros((1,4))
            neodp[0] = mt/(np.sqrt(vt)+eps)
            p -= al*neodp[0]
            p = confine(p)
            S33 = np.zeros((1,1))
            Ij,Ik = 0,0
            S33[0,0] = (np.abs(fk-1))

            signS[signS_i] = fk-1
            signS_i += 1
        else:

            JTJ = (J.T*w)@J
            D = np.diag(np.diag(JTJ))
            L = np.tril(JTJ)
            M = 1/(2-o)*(D/o+L)@np.linalg.inv(D/o)@(D/o+L).T
            PJi = np.linalg.inv(M)

            S33 = np.zeros((x.size,x.size))
            neop = np.zeros((x.size,x.size,4))


            neodp = np.zeros((x.size,4))
            for j in range(x.size):
                neodp[j] = np.linalg.solve(JTJ@PJi + x[j]*la*np.eye(4) + P_ap, (J.T*w)@(fk-1) + P_ap@(p-v_ap) * (np.sign(p[3]-v_ap[3])+1)/2)
                neodp[j] = PJi@neodp[j]

                for k in range(x.size):
                    neop[j,k] = confine(p-x[k]*al*neodp[j])

                    neofk = f_d_k_cse(lon,lat,neop[j,k,0],neop[j,k,1],neop[j,k,2],neop[j,k,3],e,noJ=True)

                    if pnorm == 0:
                        S33[j,k] = np.sqrt(np.sum(wdata*np.abs(neofk-1)**2) + P_ap[3,3]*(p[3]-v_ap[3])**2 * (np.sign(p[3]-v_ap[3])+1)/2)
                    else:
                        S33[j,k] = np.sum(wdata*np.abs(neofk-1)**pnorm) + P_ap[3,3]*(p[3]-v_ap[3])**2 * (np.sign(p[3]-v_ap[3])+1)/2

                    if np.isnan(S33[j,k]):
                        S33[j,k] = np.inf


            I = np.argmin(S33)
            Ij,Ik = np.unravel_index(I,S33.shape)

            al *= x[Ik]
            la *= x[Ij]

            p = neop[Ij,Ik]



        if diagnostics:
            alv.append(al)
            lav.append(la)
            Sv.append(S33[Ij,Ik])
            P.append(p*1.)


        if pnorm != 0. and np.linalg.norm(neodp[Ij])<1e-7:
            break
        elif pnorm == 0. and np.linalg.norm(neodp[Ij])<1e-7:
            is_p2opt = True

            if switch_to_p0:
                p[3] = initk0(p[0],p[1],p[2],X,wdata,pnorm,is_p2opt)
                print('new k0',p[3])
                switch_to_p0 = False

        elif np.linalg.norm(neodp[Ij])<1e-7:
            break

    if diagnostics:
        alv = np.array(alv)
        lav = np.array(lav)
        Sv = np.array(Sv)
        P = np.array(P)
        return p,alv,lav,Sv,P
    else:
        return p

        
