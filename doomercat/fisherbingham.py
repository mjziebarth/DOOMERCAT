
import numpy as np
import scipy.special as sp



def compute_ka0_be0(r1, r2):
    """
    Fits the dispersion parameters kappa and beta using the
    method of moments.

    Kent (1982):
    """
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
            c = fb_norm(ka0,be0)
            ck = fb_norm_dka(ka0,be0)
            cb = fb_norm_dbe(ka0,be0)

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
            raise RuntimeError("Failed to converge when estimating Fisher-Bingham parameters.")

        # Compute the value and derivatives:
        c = fb_norm(ka0,be0)
        ck = fb_norm_dka(ka0,be0)
        cb = fb_norm_dbe(ka0,be0)
        ckk = fb_norm_dka2(ka0,be0)
        ckb = fb_norm_dkabe(ka0,be0)
        cbb = fb_norm_dbe2(ka0,be0)

        S = (r1-ck/c)**2 + (r2-cb/c)**2
        Sv[i] = S
        dSdk = 2*(r1-ck/c)*-(ckk*c-ck**2)/c**2 + 2*(r2-cb/c)*-(ckb*c-ck*cb)/c**2
        dSdb = 2*(r1-ck/c)*-(ckb*c-ck*cb)/c**2 + 2*(r2-cb/c)*-(cbb*c-cb**2)/c**2

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

        # ensure the condition ka>2*be. If this is not met, the FB5 still exists but is bimodal.
        # In this step, adam is simply redirected by Gram-Schmidt projection rather than slowed down.
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


def fisher_bingham_mom(lon, lat, w):
    """


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

    K = np.array([[1,0,0],[0,np.cos(psi),-np.sin(psi)],[0,np.sin(psi),np.cos(psi)]])

    G = H @ K

    V = G.T @ mv
    T = G.T @ Sv @ G

    r1 = V[0]

    r2 = T[1,1]-T[2,2]

    g1 = G[:,0] # central axis
    g2 = G[:,1] # equator axis
    g3 = G[:,2] # pole axis

    # Now fit the dispersion parameters:
    #ka, be = compute_ka0_be0(r1, r2)
    ka,be = 0.0, 0.0

    return g1, g2, g3, ka, be
