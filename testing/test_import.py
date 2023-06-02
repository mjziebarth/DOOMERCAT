# Test code for general imports of the DOOMERCAT package.
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

from doomercat import HotineObliqueMercator
import numpy as np


def test_basic_setup():
    """
    Dummy method to test the functionality of DOOMERCAT when not all
    optional packages are installed (e.g. pyproj).
    """
    lon = np.array([15, 29, 31,  18., 23])
    lat = np.array([7, 13, 13.5, 9, 10])

    HOM = HotineObliqueMercator(lon=lon, lat=lat)

    # Test agains reference values:
    assert abs(HOM.lat_0() - 11.73733039) < 1e-6
    assert abs(HOM.lonc()  - 26.70254065) < 1e-6
    assert abs(HOM.alpha() - 70.67908827) < 1e-6
    assert abs(HOM.k0()    - 0.99997646) < 1e-6
