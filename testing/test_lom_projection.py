# Test code for the projection code in lombase.py
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

from doomercat import LabordeObliqueMercator
from pyproj import Proj

def test_project_uvk():
	"""
	Test the LOMBase._project_to_uvk() method.
	"""
	# Create some test data:
	np.random.seed(1793)
	lon = 360.0 * np.random.random(100)
	lat = 180.0 * np.random.random(100) - 90.0

	# Create test projections:
	LOM1 = LabordeObliqueMercator(lonc=73.0, lat_0 = 23.3, alpha=28.,
	                              k0=0.9988)
	proj1 = Proj("+proj=omerc +lonc=73.0 +lat_0=23.3 +alpha=28.0 "
	             "+k_0=0.9988 +ellps=WGS84 +no_off +no_rot +datum=WGS84")
	LOM3 = LabordeObliqueMercator(lonc=-5.7, lat_0 = -73.3, alpha=89.0,
	                              k0=0.8)
	proj3 = Proj("+proj=omerc +lonc=-5.7 +lat_0=-73.3 +alpha=89.0 "
	             "+k_0=0.8 +ellps=WGS84 +no_off +no_rot +datum=WGS84")
	LOM2 = LabordeObliqueMercator(lonc=180., lat_0 = 1.0, alpha=1.0,
	                              k0=0.98)
	proj2 = Proj("+proj=omerc +lonc=180.0 +lat_0=1.0 +alpha=1.0 "
	             "+k_0=0.98 +ellps=WGS84 +no_off +no_rot +datum=WGS84")

	# Now for each of the test projections, project the
	# data points and compare against the proj result:
	i = 0
	for LOM,proj in zip([LOM1, LOM2, LOM3],[proj1, proj2, proj3]):
		uc = LOM._constants[-1]
		i += 1
		ul,vl = LOM._project_to_uvk(lon,lat)
		up,vp = proj(lon, lat)
		np.testing.assert_allclose(ul+uc,up)
		np.testing.assert_allclose(vl,vp)
