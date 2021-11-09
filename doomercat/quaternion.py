# Quaternion class for data-driven optimizaion of the Laborde
# oblique Mercator projection.
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

import numpy as np

def qrot(q, p, q1):
	"""
	Quaternion rotation.
	"""
	return q * (p * q1)


class Quaternion:
	"""
	A quaternion representation to be used for rotational and scaling
	transformations.
	"""

	def __init__(self, *args):
		if len(args) == 4:
			self.r = args[0]
			self.i = args[1]
			self.j = args[2]
			self.k = args[3]
		elif len(args) == 1:
			assert isinstance(args[0],Quaternion)
			self.r = args[0].r
			self.i = args[1].i
			self.j = args[2].j
			self.k = args[4].k
		else:
			raise RuntimeError()

	def __len__(self):
		"""
		If this Quaternion instance represents an array of quaternions,
		return the number of quaternions, otherwise one.
		"""
		if isinstance(self.r, np.ndarray):
			return self.r.size
		return 1

	def __abs__(self):
		return np.linalg.norm(np.stack((self.r,self.i,self.j,self.k)), axis=0)

	def __mul__(self, other):
		"""
		Implementation of the 'quatmultiply' quaternion multiplication
		method.

		"""
		if isinstance(other, Quaternion):
			q = self
			p = other
			r = (q.r * p.r) - (q.i * p.i) - (q.j * p.j) - (q.k * p.k)
			i = (q.r * p.i) + (q.i * p.r) + (q.j * p.k) - (q.k * p.j)
			j = (q.r * p.j) - (q.i * p.k) + (q.j * p.r) + (q.k * p.i)
			k = (q.r * p.k) + (q.i * p.j) - (q.j * p.i) + (q.k * p.r)
		elif isinstance(other, tuple):
			if len(other) != 4:
				raise RuntimeError("A one-dimensional array multiplied with "
				                   "a quaternion must be four elements long!")
			r = self.r * other[0]
			i = self.i * other[1]
			j = self.j * other[2]
			k = self.k * other[3]
		elif isinstance(other, np.ndarray):
			if other.ndim == 1:
				# Multiply array with each individual component:
				r = self.r * other
				i = self.i * other
				j = self.j * other
				k = self.k * other
			elif other.ndim == 2:
				raise NotImplementedError()
			else:
				raise NotImplementedError()
		elif isinstance(other, float) or isinstance(other, np.floating) or isinstance(other,int):
			r = self.r * other
			i = self.i * other
			j = self.j * other
			k = self.k * other
		else:
			raise NotImplementedError()

		return Quaternion(r, i, j, k)

	def __rmul__(self, other):
		"""
		Implementation of right-sided 'quatmultiply'.
		"""
		if isinstance(other,Quaternion):
			return other.__mul__(self)
		elif isinstance(other, tuple):
			return self.__mul__(other)
		elif isinstance(other, np.ndarray):
			if other.ndim == 1:
				# Commutable if one-dimensional vector:
				return self.__mul__(other)
		elif isinstance(other, float) or isinstance(other,np.floating) or isinstance(other,int):
			return self.__mul__(other)
		raise NotImplementedError()

	def __neg__(self):
		"""
		Return negated quaternion.
		"""
		return Quaternion(-self.r, -self.i, -self.j, -self.k)

	def __add__(self, other):
		"""
		Quaternion addition.
		"""
		assert isinstance(other,Quaternion)
		return Quaternion(self.r+other.r, self.i+other.i,
		                  self.j+other.j, self.k+other.k)

	def __sub__(self, other):
		"""
		Quaternion subtraction.
		"""
		assert isinstance(other, Quaternion)
		return Quaternion(self.r-other.r, self.i-other.i,
		                  self.j-other.j, self.k-other.k)

	def __repr__(self):
		return "Quat(r=\n\t" + str(self.r) + "\ni=\n\t" + str(self.i) \
		        + "\nj=\n\t" + str(self.j) + "\nk=\n\t" + str(self.k) \
		        + "\n)"


	@staticmethod
	def from_vector(vec):
		"""
		Generate a quaternion representing a(n) (set of) three-dimensional
		vector(s).
		"""
		vec = np.atleast_1d(vec).reshape((-1,3))
		if not vec.shape[-1] == 3 or vec.ndim > 2:
			raise RuntimeError("Vector must be a (N,3)- or (3,)-shaped "
			                   "array!")
		return Quaternion(np.zeros(vec.shape[0]), *(vec.T))

	def update(self,other):
		"""
		Set this Quaternion's data from another Quaternion,
		"""
		assert isinstance(other, Quaternion)
		self.r = other.r
		self.i = other.i
		self.j = other.j
		self.k = other.k


	def copy(self):
		"""
		Return a copy of this quaternion.
		"""
		raise NotImplementedError()

	def inv(self):
		"""
		Return the inverted quaternion.
		"""
		invnorm = 1.0/abs(self)
		return Quaternion(self.r*invnorm, -self.i*invnorm, -self.j*invnorm,
		                  -self.k*invnorm)

	def conj(self):
		"""
		Return the conjugated quaternion.
		"""
		return Quaternion(self.r, -self.i, -self.j, -self.k)

	def theta(self):
		"""
		Interpret imaginary elements of this quaternion as vector in three
		dimensional Euclidean space and compute xy rotation angle theta.
		"""
		return np.arctan2(self.j, self.i)

	def z(self):
		"""
		Interpret imaginary elements of this quaternion as vector in three
		dimensional Euclidean space and return z component.
		"""
		return self.k

	def matrix(self):
		"""
		Return a (Nx4) matrix representation of self.
		"""
		return np.stack((self.r, self.i, self.j, self.k), axis=-1)

	def to_vector(self):
		"""
		Returns an (Nx3) vector representation of self.
		"""
		return np.stack((self.i, self.j, self.k), axis=-1)

	def norm(self):
		return abs(self)



class UnitQuaternion(Quaternion):
	"""
	A unit normed quaternion.
	"""
	def __init__(self, *args):
		if len(args) == 4:
			r = args[0]
			i = args[1]
			j = args[2]
			k = args[3]
		elif len(args) == 1:
			assert isinstance(args[0],Quaternion)
			r = args[0].r
			i = args[0].i
			j = args[0].j
			k = args[0].k

		norm = 1.0 / np.linalg.norm(np.stack((r,i,j,k)), axis=0)
		super().__init__(r*norm, i*norm, j*norm, k*norm)
