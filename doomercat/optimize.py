# Data-driven optimization methods for the Laborde oblique Mercator projection.
# This file is part of the DOOMERCAT python module.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
#          Sebastian von Specht
#
# Copyright (C) 2019-2021 Deutsches GeoForschungsZentrum Potsdam,
#                         Sebastian von Specht
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


from itertools import combinations
from math import sqrt
from random import sample, seed, getstate, setstate
from warnings import warn

import numpy as np

from .laborde import ellipsoid_projection_radian, laborde_variables, \
                     laborde_sub, laborde_projection
from .quaternion import Quaternion, UnitQuaternion, qrot
from .lomerror import OptimizeError


def reduce_iteratively(x, Ndest, logger=None):
    """
    Reduce the size of the point set x by iterating over
    point pairs with ascending distance and, in each step,
    removing one point of the pair until there are only
    Ndest remaining points.
    """
    # Import cKDTree only here. It is not required for the rest
    # of the code, which may work without installing the whole
    # SciPy stack (though requiring numpy, hence SciPy is likely
    # to be installed as well).
    from scipy.spatial import cKDTree

    # Logging:
    if logger is not None:
        logger.log(20, "Reducing points to " + str(Ndest))
        next_log = 1.0

    N = M = x.shape[0]
    mask = np.ones(N, dtype=bool)
    indices = np.arange(N)
    while M > Ndest:
        # Update tree and indices for all
        # remaining points:
        xl = x[mask,:]
        li = indices[mask]
        tree = cKDTree(xl)
        # First, obtain the closest 10 points for
        # each point:
        d, closest = tree.query(xl, 10)

        # The maximum distance until which we
        # consider nodes is the minimum of the
        # set of maximums of distances to the
        # ten neighbors for each node:
        maxdist = d[:,-1].min()

        # Now create a total list of pairs:
        pairs = [(j, closest[j,i], d[j,i]) for j in range(M)
                 for i in range(1,10)]

        # Remove all above maxdist:
        pairs = list(filter(lambda x : x[2] <= maxdist, pairs))

        # Sort the list ascendingly by distance:
        pairs = sorted(pairs, key=lambda x : x[2])

        # Now remove points:
        while M > Ndest and len(pairs) > 0:
            j,i,d = pairs.pop(0)
            if mask[li[j]] and mask[li[i]]:
                mask[li[i]] = False
                M -= 1

                # Logging the number of remaining points:
                if logger is not None:
                    if M <= N * next_log:
                        next_log -= 0.01
                        logger.log(20, "remaining=" + str(M))

    # Return the mask:
    return mask


def mercator_project(q, k0):
	"""
	Return a Quaternion with Mercator-projected
	vector components.
	"""
	nrm = abs(q)
	nrm_h = np.sqrt(q.i**2 + q.j**2)
	x_x = k0 * q.i / nrm_h
	x_y = k0 * q.j / nrm_h
	x_z = nrm * np.arctanh(q.k / nrm)
	q_merc = Quaternion(np.zeros_like(x_x), x_x, x_y, x_z)

	return q_merc


def rotate_mercator_residual(U, V, W, q, k0):
	"""
	Rotate the quaternion of Laborde coordinates into cylinder coordinates
	and do the Mercator projection afterwards.
	Then compute residuals between Mercator projected data and the rotated
	laborde coordinates.
	"""
	ql = UnitQuaternion(np.zeros_like(U), U, V, W)
	qv = q * (1/abs(q))

	# pq is the original point set transformed into standard cylinder coordinates
	pq = qrot(qv.conj(), ql, qv)

	# Mercator projection stretching the z component and scaling the
	# x and y component to k0:
	c = mercator_project(pq, k0)

	# Computing the residuals:
	delta = (c-pq) * (0,1,1,1)

	return delta, c, pq


def cost_function(delta, pnorm, w, k0, k0_ap, sigma_k0):
	"""
	Compute the cost from a number of residuals.
	"""
	cost = np.sum(w * abs(delta)**pnorm) ** (1./pnorm)
	if k0 < k0_ap:
		cost += (k0-k0_ap)**2 / sigma_k0**2
	return cost


def cost_function_convenient(lambda_, phi, w, q, k0, vize, lambda_c, k0_ap,
                             sigma_k0, pnorm, f=1/298.):
	"""
	Another API for cost_function.
	"""
	# 1) Laborde projection:
	U,V,W = laborde_projection(lambda_, phi, vize, lambda_c, f)
	# 2) Rotation, Mercator projection, and residual
	#    computation:
	delta, c, pq = rotate_mercator_residual(U, V, W, q, k0)


	# 3) Compute cost:
	cost = cost_function(delta, pnorm, w, k0, k0_ap, sigma_k0)

	return cost, delta


def project_and_compute_cost(lambda_, phi, vize, lambda_c, q, k0, pnorm, f=1/298.):
	"""
	Chain Laborde projection, rotation, Mercator projection, and
	the residual calculations.
	"""
	# 1) Laborde projection:
	U,V,W = laborde_projection(lambda_, phi, vize, lambda_c, f)

	# 2) Rotation, Mercator projection, and residual
	#    computation:
	delta, c, pq = rotate_mercator_residual(U, V, W, q, k0)


	# 3) Compute cost:
	cost = cost_function(delta, pnorm)

	return cost


def initial_guess(x, weights, increase_distance_weight=True):
	"""
	Calculate the cross product of the data points to obtain a first guess
	for the optimum cylinder position.
	"""
	assert isinstance(x,np.ndarray)
	assert x.ndim == 2 and x.shape[-1] == 3
	N = x.shape[0]

	# Compute cross product of all pairs of points:

	pairs = np.array([v for v in combinations(x,2)])
	if weights is None:
		wl = np.ones(pairs.shape[0])
		wr = 1
	else:
		wl = weights[[i for i,j in combinations(np.arange(N),2)]]
		wr = weights[[j for i,j in combinations(np.arange(N),2)]]
	cross_products = np.cross(pairs[:,0,:], pairs[:,1,:]) \
	                 * np.sqrt(wl * wr)[:,np.newaxis]

	if increase_distance_weight:
		mask = np.sum(pairs[:,0,:] * pairs[:,1,:], axis=-1) < 0
		if np.any(mask):
			crsnrm = np.linalg.norm(cross_products[mask,:],axis=1)
			nrm = np.linalg.norm(pairs[mask,0,:], axis=1) \
			      * np.linalg.norm(pairs[mask,1,:], axis=1)
			cross_products[mask] *= (2*nrm/crsnrm - 1)[:,np.newaxis]

	# Make sure that all cross products are in the same hemisphere:
	cross_products[cross_products @ cross_products[0,:] < 0] *= -1

	# Normed initial guess for cylinder symmetry axis:
	p = np.mean(cross_products, axis=0)
	p /= np.linalg.norm(p,2)

	return p


def jacobian(q, k0, vize, lambda_c, lambda_, phi, pnorm, f=1./298):
	"""
	Variables:

	lambda_, phi : given in radians

	...
	fies : phi_s, given in radians.
	"""
	# Inverse norm of q:

	qnrm = abs(q)
	qnrm_i = 1.0/qnrm

	# versor
	qv = q * qnrm_i

	# Laborde projection variables:
	e, B, fies, C, q_lab, P, sin_vize, L, cosP \
	   = laborde_variables(lambda_, phi, vize, lambda_c, f)

	# Laborde projection:
	U,V,W = laborde_sub(P, cosP, L)

	# Laborde vector:
	p = Quaternion(np.zeros_like(U), U, V, W)

	# pq is the Laborde point set transformed into standard cylinder coordinates
	pq = qrot(qv.conj(), p, qv)

	c = mercator_project(pq, k0)

	pqconj = pq.conj()

	dqr =  Quaternion(1,0,0,0)
	dqi =  Quaternion(0,1,0,0)
	dqj =  Quaternion(0,0,1,0)
	dqk =  Quaternion(0,0,0,1)

	# the residuals
	delta = c-pq

	# Derive pq by components:
	pqv = p * qv
	d_pq_d_qr = 2.*(( qnrm_i * dqr - q.r * q.conj() * qnrm_i**3) * pqv) * (0,1,1,1)
	d_pq_d_qi = 2.*((-qnrm_i * dqi - q.i * q.conj() * qnrm_i**3) * pqv) * (0,1,1,1)
	d_pq_d_qj = 2.*((-qnrm_i * dqj - q.j * q.conj() * qnrm_i**3) * pqv) * (0,1,1,1)
	d_pq_d_qk = 2.*((-qnrm_i * dqk - q.k * q.conj() * qnrm_i**3) * pqv) * (0,1,1,1)

	# Derive offset vector:
	pq_h = pq * (0,1,1,0)
	pq_h_nrm = abs(pq_h)
	pq_h_nrm_i = 1.0/pq_h_nrm

	d_pq_h_nrm_dqr   = pq_h_nrm_i * (pq.i*d_pq_d_qr.i   + pq.j*d_pq_d_qr.j)
	d_pq_h_nrm_dqi   = pq_h_nrm_i * (pq.i*d_pq_d_qi.i   + pq.j*d_pq_d_qi.j)
	d_pq_h_nrm_dqj   = pq_h_nrm_i * (pq.i*d_pq_d_qj.i   + pq.j*d_pq_d_qj.j)
	d_pq_h_nrm_dqk   = pq_h_nrm_i * (pq.i*d_pq_d_qk.i   + pq.j*d_pq_d_qk.j)

	pq_nrm_i = 1.0 / abs(pq)

	d_pq_nrm_dqr   = (d_pq_d_qr   * pqconj).r * pq_nrm_i
	d_pq_nrm_dqi   = (d_pq_d_qi   * pqconj).r * pq_nrm_i
	d_pq_nrm_dqj   = (d_pq_d_qj   * pqconj).r * pq_nrm_i
	d_pq_nrm_dqk   = (d_pq_d_qk   * pqconj).r * pq_nrm_i


	# Differentiate the Mercator projection.
	# Note: c = P(qlq) (very important)
	pq_h_nrm_i_2 = pq_h_nrm_i**2
	d_ci_d_qr   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqr   * pq.i + pq_h_nrm_i * d_pq_d_qr.i)
	d_ci_d_qi   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqi   * pq.i + pq_h_nrm_i * d_pq_d_qi.i)
	d_ci_d_qj   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqj   * pq.i + pq_h_nrm_i * d_pq_d_qj.i)
	d_ci_d_qk   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqk   * pq.i + pq_h_nrm_i * d_pq_d_qk.i)


	d_cj_d_qr   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqr * pq.j   + pq_h_nrm_i * d_pq_d_qr.j)
	d_cj_d_qi   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqi * pq.j   + pq_h_nrm_i * d_pq_d_qi.j)
	d_cj_d_qj   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqj * pq.j   + pq_h_nrm_i * d_pq_d_qj.j)
	d_cj_d_qk   = k0 * (- pq_h_nrm_i_2 * d_pq_h_nrm_dqk * pq.j   + pq_h_nrm_i * d_pq_d_qk.j)

	# Derivative of k component is different due to Artanh:
	pq_nrm_i_2 = pq_nrm_i**2
	d_ck_d_qr_in   = (- pq_nrm_i_2 * d_pq_nrm_dqr * pq.k   + pq_nrm_i * d_pq_d_qr.k)
	d_ck_d_qi_in   = (- pq_nrm_i_2 * d_pq_nrm_dqi * pq.k   + pq_nrm_i * d_pq_d_qi.k)
	d_ck_d_qj_in   = (- pq_nrm_i_2 * d_pq_nrm_dqj * pq.k   + pq_nrm_i * d_pq_d_qj.k)
	d_ck_d_qk_in   = (- pq_nrm_i_2 * d_pq_nrm_dqk * pq.k   + pq_nrm_i * d_pq_d_qk.k)

	C0 = 1.0 / (1.0 - (pq.k * pq_nrm_i)**2 )
	d_ck_d_qr   = C0 * d_ck_d_qr_in
	d_ck_d_qi   = C0 * d_ck_d_qi_in
	d_ck_d_qj   = C0 * d_ck_d_qj_in
	d_ck_d_qk   = C0 * d_ck_d_qk_in


	d_delta_d_qr =   dqi * d_ci_d_qr \
		           + dqj * d_cj_d_qr \
		           + dqk * d_ck_d_qr \
		           - d_pq_d_qr
	d_delta_d_qi =   dqi * d_ci_d_qi \
		           + dqj * d_cj_d_qi \
		           + dqk * d_ck_d_qi \
		           - d_pq_d_qi
	d_delta_d_qj =   dqi * d_ci_d_qj \
		           + dqj * d_cj_d_qj \
		           + dqk * d_ck_d_qj \
		           - d_pq_d_qj
	d_delta_d_qk =   dqi * d_ci_d_qk \
		           + dqj * d_cj_d_qk \
		           + dqk * d_ck_d_qk \
		           - d_pq_d_qk


	# Now derive by radius:
	d_delta_d_R = (dqi * pq.i + dqj * pq.j) * pq_h_nrm_i

	# Derive norm of residuals by single components:
	delta_nrm = abs(delta)
	delta_nrmi = 1. / delta_nrm

	d_delta_nrm_d_R    = (d_delta_d_R    * delta.conj()).r * delta_nrmi
	d_delta_nrm_d_qr   = (d_delta_d_qr   * delta.conj()).r * delta_nrmi
	d_delta_nrm_d_qi   = (d_delta_d_qi   * delta.conj()).r * delta_nrmi
	d_delta_nrm_d_qj   = (d_delta_d_qj   * delta.conj()).r * delta_nrmi
	d_delta_nrm_d_qk   = (d_delta_d_qk   * delta.conj()).r * delta_nrmi
	Jn = np.array([d_delta_nrm_d_qr, d_delta_nrm_d_qi,
					d_delta_nrm_d_qj,
					d_delta_nrm_d_qk, d_delta_nrm_d_R]).T

	return Jn, abs(delta)


def cylinder_axis(q):
	"""
	Obtain the cylinder axis in Earth-fixed coordinate system:
	"""
	qcyl = Quaternion(0,0,0,1)
	q = q * (1./abs(q))
	qcyl = qrot(q, qcyl, q.conj())

	return np.array([float(qcyl.i), float(qcyl.j), float(qcyl.k)])


def compute_vize(q, e2):
	"""
	Computes the vize given a cylinder rotation q.
	"""
	# First compute the cylinder axis:
	n = cylinder_axis(q)

	# Now we can compute fies:
	fies = -np.arctan(n[0] / n[2])

	# Test for NAN:
	if np.isnan(fies):
		raise OptimizeError(reason='convergence',
		          lonc=None, lat_0=None, alpha=None,
		          k0=None)

	# We can compute then vize:
	z = np.sin(fies)
	Gamma = np.sqrt((1-e2) * ((1-e2)/(4*e2**2*z**4) + 1/e2*(1/z**2 - 1)))
	E = (1-e2) / (2*e2*z**2) + 1
	zv = np.maximum(np.minimum(-Gamma + E,0.9999),-0.9999)
	v = np.arcsin(np.sign(fies)*np.sqrt(zv))
	return v


def vize2fies(vize, e2):
	"""
	Computes phi_s given phi_c and e^2.
	"""
	B = np.sqrt(1 + e2*np.cos(vize)**4/(1-e2))
	return np.arcsin(np.minimum(np.maximum(np.sin(vize) / B,-1),1))


def compute_azimuth(q, vize, f):
	"""
	Computes the azimuth at the central point.
	"""
	# First compute cylinder axis:
	n = cylinder_axis(q)

	# Now we need the sine of the latitude of the cylinder axis,
	# which is the cylinder axis z component:
	sin_phi_cyl = n[2]

	# Furthermore, we need the cosine of the central coordinate
	# latitude:
	fies = vize2fies(vize, f*(2-f))
	cos_fies = np.cos(fies)

	# Now, we can compute the azimuth of the cylinder equator at the
	# central coordinate using spherical trigonometry on the Laborde
	# sphere:
	azimuth = np.arcsin(np.maximum(np.minimum(sin_phi_cyl / cos_fies, 1), -1))

	# The spherical geometry used does not consider the correct
	# sign of the azimuth. Thus, we may have to multiply by -1.
	# This can be decided by considering the cross product
	# of the cylinder axis and the central axis:
	if n[1] > 0:
		azimuth *= -1

	return azimuth


def _lm_inner(q, k0, k0_ap, sigma_k0, lambda_c, pnorms, lambda_, phi, w,
              initial_lambda, lbda_min, lbda_max, nu, f, N, Nmax,
              exit_delta, logger):
    """
    Inner loop of the Levenberg-Marquardt algorithm.
    """
    print("_lm_inner Nmax:",Nmax)
    # Initialize some temporary variables:
    lbda = float(initial_lambda)
    e2 = f*(2-f)
    e = sqrt(e2)

    vize = compute_vize(q * (1./abs(q)), e2)

    best_config = (np.inf, q, k0)

    # A-priori information:
    sigma2_k0 = sigma_k0**2
    iS1 = np.diag((0,0,0,0,1./sigma2_k0))

    for p in pnorms:
        # We change norms here, so reset the cost, which is not intercomparable.
        delta_cost = 0.0
        cost = cost_function_convenient(lambda_, phi, w, q, k0, vize, lambda_c,
                                        k0_ap, sigma_k0, p)[0]
        cost_new = cost
        W = None

        lbda = max(lbda,initial_lambda)

        if logger is not None:
            logger.log(20, "Optimize pnorm=" + str(p))

        for i in range(Nmax):
            J, delta = jacobian(q, k0, vize, lambda_c, lambda_, phi, p, f=f)

            # Handle the delta:
            np.abs(delta,out=delta)

            # Reweight if pnorm != 2:
            if p != 2:
                if p > 2:
                    # Determine maximum which will dominate the sum of
                    # powered residuals for high pnorms:
                    imax = np.argmax(delta)
                    deltamax = delta[imax]

                    # Compute power of normed residuals and assure stability
                    # of dominant (maximum) residual:
                    deltapow = delta/deltamax
                    np.power(deltapow, pnorm-2, out=deltapow)
                    deltapow[imax] = 1.0
                else:
                    # Here the maximum weight just needs to be limited:
                    deltapow = delta**(pnorm-2)
                    np.minimum(deltapow, 1e4, out=deltapow)

                    # Normalize to the highest weight:
                    deltapow /= deltapow.max()

                if w is None:
                    W = deltapow
                else:
                    W = w * deltapow

            elif w is not None:
                # Handle given weights for p=2 here:
                W = w

            # A-prior information about k0:
            h = float(k0 < k0_ap)
            iS1_pdiff = -h*iS1 @ np.array((0,0,0,0, k0_ap-k0))

            # Inversion here!
            if W is None:
                JW = J.T
            else:
                # Use a memory-efficient version of JW = J.T @ np.diag(W):
                JW = J.T * W[np.newaxis,:]
            JWd = JW @ delta
            JWJ = JW @ J
            I = np.diag(np.diag(JWJ))
            # TESTING:
            # We do not want the dampening to affect k0.
            # k0 already has prior information.
            I[4,4] = 0.0
            M1 = np.linalg.inv(JWJ + lbda * I + h*iS1)
            dp0 =  -M1 @ (JWd + iS1_pdiff)
            q0 = Quaternion(q.r+dp0[0], q.i+dp0[1], q.j+dp0[2], q.k+dp0[3])
            vize0 = compute_vize(q0 * (1./abs(q0)), e2)
            # lambda_, phi, w, q, k0, vize, lambda_c, k0_ap,
            #                 sigma_k0, pnorm, f=1/298.
            cost0 = cost_function_convenient(lambda_, phi, w, q0, k0+dp0[4],
                                             vize0, lambda_c, k0_ap, sigma_k0,
                                             p, f=f)[0]

            M1 = np.linalg.inv(JWJ + lbda / nu * I + h*iS1)
            dp1 =  - M1 @ (JWd + iS1_pdiff)
            q1 = Quaternion(q.r+dp1[0], q.i+dp1[1], q.j+dp1[2], q.k+dp1[3])
            vize1 = compute_vize(q1 * (1./abs(q1)), e2)
            cost1 = cost_function_convenient(lambda_, phi, w, q1, k0+dp1[4],
                                             vize1, lambda_c, k0_ap, sigma_k0,
                                             p, f=f)[0]

            M1 = np.linalg.inv(JWJ + lbda * nu * I + h*iS1)
            dp2 =  - M1 @ (JWd + iS1_pdiff)
            q2 = Quaternion(q.r+dp2[0], q.i+dp2[1], q.j+dp2[2], q.k+dp2[3])
            vize2 = compute_vize(q2 * (1./abs(q2)), e2)
            cost2 = cost_function_convenient(lambda_, phi, w, q2, k0+dp2[4],
                                             vize2, lambda_c, k0_ap, sigma_k0,
                                             p, f=f)[0]

            if cost0 <= cost1 and cost0 <= cost2:
                q = q0
                k0 = k0+dp0[4]
                vize = vize0
                cost_new = cost0
            elif cost1 <= cost0 and cost1 <= cost2:
                q = q1
                k0 = k0+dp1[4]
                vize = vize1
                lbda /= nu
                cost_new = cost1
            elif cost2 <= cost0 and cost2 <= cost1:
                q = q2
                k0 = k0+dp2[4]
                vize = vize2
                lbda *= nu
                cost_new = cost2
            else:
                lbda /= nu

            if lbda < lbda_min:
                lbda = lbda_min
            elif lbda > lbda_max:
                lbda = lbda_max

            # Exit condition:
            if abs(cost_new - cost) < exit_delta and N is None:
                if p == pnorms[0] and i == 0:
                    # The convergence failed since for the first
                    # p-norm, the smallest one (likely 2), the convergence
                    # stopped after one iteration.
                    raise OptimizeError(reason='convergence',
                              lonc=np.rad2deg(lambda_c),
                              lat_0=np.rad2deg(vize),
                              alpha=None, k0=k0)
                break

            cost = cost_new

            if cost < best_config[0]:
                best_config = (cost, q, k0)

            # Logging:
            if logger is not None:
                if i > Nmax:
                    raise RuntimeError("i=",i,">Nmax=",Nmax)
                logger.log(20, "Step[" + str(i) + "]: cost=" + str(cost))
                if hasattr(logger, "exit") and logger.exit():
                    # Concurrent exit requested. Raise an error.
                    raise RuntimeError("Early exit requested.")

    # Take the minimum cost:
    cost, q, k0 = best_config

    return cost, q, compute_vize(q * (1./abs(q)), e2), k0



def levenberg_marquardt(lon, lat, weight=None, pnorm=2, f=1/298.257223563,
                        k0_ap=0.98, sigma_k0=0.02, initial_lambda=10,
                        nu=0.99, N=None, lbda_min=1e-10, lbda_max=1e10,
                        use='all', reproducible=True, logger=None,
                        best_of=1, k0_start=1.0):
	"""
	The main optimization routine to determine the parameters
	of the optimum Laborde oblique Mercator projection (LOM).

	Parameters:
	   lon            : Longitude coordinates of the data points
	                    to optimize the LOM for. Must be given in
	                    arcdegrees.
	   lat            : Latitude coordinates of the data points
	                    to optimize the LOM for. Must be given in
	                    arcdegrees.
	   weight         : Data weights. Can be used to vary the relative
	                    importance of the data points. Must be of same
	                    shape as lon & lat. If None, no prior
	                    weights will be used for the data points.
	                    (Default: None)
	   pnorm          : Power of the p-norm to use for the cost
	                    function that is mimized.
	                    (Default: 2)
	   f              : Flattening of the ellipsoid which is used
	                    to interpret the geographical coordinates.
	                    (Default: 1/298.257223563, corresponding
	                     to WGS84)
	   k0_ap          : The k_0 threshold beyond which smaller k_0
	                    are constrained using a quadratic potential.
	                    This constraint helps preventing the solver
	                    from approaching small-circle solutions.
	                    Setting k0_ap to zero removes the constraint.
	                    (Default: 0.98)
	   sigma_k0       : The scale of the quadratic constrains for
	                    small k_0, i.e. the standard deviation of
	                    the quadratic branch of the k_0 potential.
	                    (Default: 0.02)
	   initial_lambda : Initial value of damping factor lambda of
	                    the Levenberg-Marquardt algorithm.
	                    (Default: 10)
	   nu             : Scaling factor of lambda. In each step,
	                    lambda may be changed by a factor nu or
	                    1/nu.
	                    (Default: 0.99)
	   N              : Number of steps to take. If None, the
	                    iteration stops if the change in cost
	                    function is less than 1e-12 or if a
	                    maximum of 1400 steps have been performed.
	                    (Default: None)
	   use            : How many points to use for the optimization.
	                    If 'all', all points of the data set are
	                    used. If a number is given, only that many
	                    of the points are used, potentially speeding
	                    up and stabilizing the inversion. The points
	                    to use are then selected by iteratively
	                    removing a point of the spatially closest
	                    pair until only *use* nodes remain.
	                    (Default: 'all')
	   reproducible   : Whether or not the initial conditions
	                    should be chosen reproducibly. If True,
	                    sets a reproducible seed before choosing
	                    the random points to determine the initial
	                    parameter guesses using cross products.
	                    (Default: True)

	Returns:
	   lonc, lat_0, alpha, k0

	   lonc  : Longitude of the central point given in arcdegrees.
	   lat_0 : Latitude of the central point given in arcdegrees.
	   alpha : Azimuth of the central line at the central point
	           given in arcdegrees.
	   k0    : Scale factor at the central point.
	"""

	if k0_start < 1e-3 or k0_start > 2.0:
		raise ValueError("k0_start has to be between 1e-3 and 2.0")

	if N is None:
		Nmax = 1400
	else:
		Nmax = N
	reproducible = bool(reproducible)

	# Iterate over pnorms. The L2 norm is fairly robust to the starting
	# conditions. Afterwards, do the adjustments of the higher norms that
	# are more sensitive to starting conditions.
	if pnorm == 2:
		PNORMS = [2]
	else:
		if pnorm > 10:
			PNORMS = [2,10,pnorm]
		else:
			PNORMS = [2,pnorm]

	# Make sure we have numpy arrays:
	if weight is not None:
		w = np.array(weight,copy=True)
		# Test for NaN:
		if np.any(np.isnan(w)):
			raise ValueError("Weighting matrix contains NaN.")
		# Normalize the weight to its maximum value (this is more
		# of a numerical stability issue, to make sure that extreme
		# value ranges do not occur):
		w /= w.max()
	else:
		# Use None for none-existent weights:
		w = None

	# Convert to radians:
	phi = np.deg2rad(lat)
	lambda_ = np.deg2rad(lon)

	# Initial parameters:
	f = float(f)
	nu = float(nu)

	# Select the nodes which to use:
	x = ellipsoid_projection_radian(lambda_, phi, 0.0, 1.0, f)
	M = max(lon.size, lat.size)
	if use == 'all':
		lambda_work = lambda_
		phi_work = phi
		w_work = w
	else:
		use = int(use)
		if use < 2:
			raise ValueError("At least two points have to be used.")
		if use < M:
			mask = reduce_iteratively(x, use, logger=logger)
			x = x[mask,:]
			lambda_work = lambda_[mask]
			phi_work = phi[mask]
			if w is None:
				w_work = None
			else:
				w_work = w[mask]
			M = use
		else:
			warn("The number of points to use is larger than the actual "
			     "number of points.")

	# Convert coordinates to radian:
	lambda_c = np.arctan2(np.mean(np.sin(lambda_)), np.mean(np.cos(lambda_)))

	# Initial guess:
	if len(x) < 2:
		raise OptimizeError(reason='data_count')

	if best_of == 1:
		if len(x) <= 20:
			q = Quaternion(0, *initial_guess(x, w_work))
		else:
			# Limit the number of data points to 20, so that we compute only 400 pairs.
			# Otherwise, the quadratic increase of the pairwise computation supersedes
			# the number of operations of the main loop and the initial guess will
			# dominate the computation time consumption - not something we want.
			if reproducible:
				# Seed the RNG reproducibly. To prevent unintended consequences for
				# programs using the python RNG, save and restor its state:
				rng_state = getstate()
				seed(78962)

			q = Quaternion(0, *initial_guess(x[sample(list(range(len(x))),20),
			                                  :],w))

			if reproducible:
				setstate(rng_state)

		# Log start of optimization loop:
		if logger is not None:
			logger.log(20, "Enter optimization loop.")

		cost, q, vize, k0 \
		   = _lm_inner(q, k0_start, k0_ap, sigma_k0, lambda_c, PNORMS,
		               lambda_work, phi_work, w_work, initial_lambda, lbda_min,
		               lbda_max, nu, f, N, Nmax, 1e-8, logger)

	elif best_of < 1:
		raise ValueError("'best_of' needs to be at least 1.")

	else:
		# Multiple iterations are requested.
		#best_config = (np.inf, np.NaN, np.NaN, np.NaN)
		configs = []
		if reproducible:
			rng = np.random.default_rng(45477564783787)
		else:
			rng = np.random.default_rng()

		for i in range(best_of):
			# Get a random starting condition:
			q = UnitQuaternion(0.0, *rng.random(3))

			if use is not 'all':
				select = rng.choice(M, use)
				lambda_work = lambda_[select]
				phi_work = phi[select]
				if w is not None:
					w_work = w[select]

			if logger is not None:
				logger.log(20, "Enter optimization loop.")

			try:
				cost, q, vize, k0 \
				   = _lm_inner(q, k0_start, k0_ap, sigma_k0, lambda_c, PNORMS,
				               lambda_work, phi_work, w_work, initial_lambda,
				               lbda_min, lbda_max, nu, f, N, 200, 1e-4, logger)
			except OptimizeError as err:
				if i != best_of-1 and len(configs) > 0:
					continue
				else:
					raise err

			# Compute the full cost:
			cost = cost_function_convenient(lambda_work, phi_work, w_work, q,
			                                k0, vize, lambda_c, k0_ap, sigma_k0,
			                                PNORMS[-1], f=f)[0]

			#if cost < best_config[0]:
			#	best_config = (cost, q, vize, k0)
			configs.append((cost, q, vize, k0))

		print("best configs Nmax:",Nmax)

		# Take the best configurations:
		best_configs = sorted(configs, key=lambda x : x[0])[:10]
		for i,conf in enumerate(best_configs):
			cost, q, vize, k0 = conf

			if logger is not None:
				logger.log(20, "Enter optimization loop.")

			# One more optimization with the full data:
			cost, q, vize, k0 \
			   = _lm_inner(q, k0, k0_ap, sigma_k0, lambda_c, PNORMS,
			               lambda_, phi, w, initial_lambda,
			               lbda_min, lbda_max, nu, f, N, Nmax, 1e-8, logger)

			best_configs[i] = (cost, q, vize, k0)

		# Now choose the best of the best:
		cost, q, vize, k0 = sorted(best_configs, key=lambda x : x[0])[0]



	# Now compute the angle parameter:
	azimuth = compute_azimuth(q, vize, f)

	# Convert to degree:
	lonc = np.rad2deg(lambda_c)
	lat_0 = np.rad2deg(vize)
	alpha = np.rad2deg(azimuth)

	# Sanity checks:
	if lat_0 >= 90.0 or lat_0 <= -90.0:
		raise OptimizeError(reason='lat_0', lonc=lonc, lat_0=lat_0,
		                    alpha=alpha, k0=k0)

	if np.any(np.isnan((lonc, lat_0, alpha, k0))):
		raise OptimizeError(reason='nan')


	if logger is not None:
		logger.log(20, "Finished with lonc=" + str(lonc) + ", lat_0="
		               + str(lat_0) + ", alpha=" + str(alpha)
		               + ", k_0=" + str(k0))

	# Fin:
	return lonc, lat_0, alpha, k0
