# Error class for the optimized Laborde oblique Mercator projection.
# This file is part of the DOOMERCAT python module.
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

class OptimizeError(Exception):
	"""
	Raised when the optimization of the Laborde oblique
	Mercator projection fails, e.g. by resulting in an
	invalid parameter combination.

	Reasons:
	   'lat_0'       : lat_0 out of bounds (<=-90° or >= 90°)
	   'convergence' : Convergence failed.
	   'data_count'  : Too little (<2) data points.
	   'nan'         : At least one parameter was nan.
	"""
	def __init__(self, reason=None,
	             lonc=None, lat_0=None, alpha=None, k0=None):
		self._lonc = lonc
		self._lat_0 = lat_0
		self._alpha = alpha
		self._k0 = k0
		self._reason = reason

	def __str__(self):
		if self._reason == 'lat_0':
			# lat_0 is the reason why the error is thrown:
			return "LOMOptimizeError(lat_0=" + str(self.lat_0()) + ")"
		else:
			# Very general error message:
			return "LOMOptimizeError(lat_0=" + str(self.lat_0()) \
			       + ", lonc=" + str(self.lonc()) \
			       + ", alpha=" + str(self.alpha()) \
			       + ", k0=" + str(self.k0()) + ")"

	def message(self):
		"""
		Print an error message.
		"""
		if self._reason == 'lat_0':
			return "Resulting central latitude (" + str(self.lat_0()) + ") is " \
			       "out of bounds."
		elif self._reason == 'convergence':
			return "Convergence failed. Initial parameters chosen were lonc=" \
			       + str(self.lonc()) + ", lat_0=" + str(self.lat_0()) \
			       + ", k0=" + str(self.k0()) + ". This likely has to do with " \
			       "an unfavorable distribution of data points. Try selecting " \
			       "points spanning a larger area."
		elif self._reason == 'data_count':
			return "Too few data points selected. Select at least two data points."
		elif self._reason == 'nan':
			return "Estimation failed: At least one of the parameters was " \
			       "estimated to be nan. Try selecting points spanning a larger " \
			       "area."

	def reason(self):
		"""
		Return the reason code.
		"""
		return self._reason

	def lonc(self):
		"""
		Returns the optimization result for the lonc
		parameter.
		"""
		return self._lonc

	def lat_0(self):
		"""
		Returns the optimization result for the lonc
		parameter.
		"""
		return self._lat_0

	def alpha(self):
		"""
		Returns the optimization result for the alpha
		parameter.
		"""
		return self._alpha

	def k0(self):
		"""
		Returns the optimization result for the k0
		parameter.
		"""
		return self._k0
