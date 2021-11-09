# Test code for the code in optimize.py
# This code validates against the PROJ implementation.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2021 Deutsches GeoForschungsZentrum Potsdam
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
import pytest

from doomercat.quaternion import UnitQuaternion, Quaternion, qrot
from doomercat.optimize import compute_azimuth


# An old version of compute_azimuth to test new versions against:
def cylinder_axis(q):
	"""
	Obtain the cylinder axis in Earth-fixed coordinate system:
	"""
	qcyl = Quaternion(0,0,0,1)
	q = q * (1./abs(q))
	qcyl = qrot(q, qcyl, q.conj())

	return np.array([float(qcyl.i), float(qcyl.j), float(qcyl.k)])
def vize2fies(vize, e2):
	"""
	Computes phi_s given phi_c and e^2.
	"""
	B = np.sqrt(1 + e2*np.cos(vize)**4/(1-e2))
	return np.arcsin(np.minimum(np.maximum(np.sin(vize) / B,-1),1))


def compute_azimuth_old(q, vize, f):
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

	# Now, we can compute the azimuth of the cylinder axis at the
	# central coordinate using spherical trigonometry on the Laborde
	# sphere:
	azimuth = np.arccos(np.maximum(np.minimum(sin_phi_cyl / cos_fies, 1), -1))

	# The spherical geometry used does not consider the correct
	# sign of the azimuth. Thus, we may have to multiply by -1.
	# This can be decided by considering the cross product
	# of the cylinder axis and the central axis:
	central_axis = (cos_fies, 0.0, np.sin(fies))
	cxc = np.cross(central_axis,n)
	if cxc[2] < 0:
		azimuth *= -1

	# Finally, so far we computed the azimuth in direction of the
	# cylinder axis. The azimuth we are concerned with is the azimuth
	# of great circle describing the cylinder area, i.e. is
	# perpendicular to the central axis.
	azimuth += 0.5*np.pi

	# Make sure that azimuth is in the north-pointing interval
	azimuth = azimuth % (2*np.pi)
	if azimuth > 0.5*np.pi and azimuth <= 1.5 * np.pi:
		azimuth -= np.pi

	# Make sure that azimuth is given from -pi/2 to pi/2:
	if azimuth > 0.5*np.pi:
		azimuth -= 2*np.pi

	return azimuth


# The test code:
def test_compute_azimuth():
	"""
	This method tests whether the azimuth is computed correctly.
	It uses an old version of the method to compare.
	"""
	# Generate random rotation Quaternions:
	np.random.seed(197)
	N = 100
	R, I, J, K = [np.random.random(N) for i in range(4)]

	# Generate random phi_c:
	vizes = np.array([-0.45,-0.1, 0.0, 0.3, 0.499])*np.pi

	# Test:
	for r,i,j,k in zip(R,I,J,K):
		q = UnitQuaternion(r,i,j,k)
		for vize in vizes:
			az0 = compute_azimuth_old(q, vize, 1/298.)
			az1 = compute_azimuth(q, vize, 1/298.)
			# Make sure that the two are equal.
			# The only place where difference is allowed is if the azimuth
			# is close to 90° which is identical to -90°.
			# This flips the y direction but describes the same level of
			# optimization:
			assert abs(az0-az1) < 1e-8 or abs(abs(az0)-0.5*np.pi) < 1e-8
