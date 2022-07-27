# Test code for using the LabordeObliqueMercator API.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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
import pytest

def test_laborde_oblique_mercator_ellipsoid_signature():
    """
    Ensures that the ellipsoid information can be properly deciphered
    and raises errors if invalid.
    """

    # Test the predefined ellipsoids:
    for ellipsoid in ('WGS84','IERS2003','GRS80'):
        HOM = HotineObliqueMercator(lonc=20, lat_0=20, alpha=20, k0=1.0,
                                    ellipsoid='WGS84')

    # Test a custom ellipsoid:
    HOM = HotineObliqueMercator(lonc=20, lat_0=20, alpha=20, k0=1.0,
                                a=6400.0, f=1.0/300.)

    # Things that should fail:
    with pytest.raises(RuntimeError):
        HOM = HotineObliqueMercator(lonc=20, lat_0=20, alpha=20, k0=1.0,
                                    ellipsoid='WGS84',
                                    a=6400.0, f=1.0/300.)

    with pytest.raises(RuntimeError):
        HOM = HotineObliqueMercator(lonc=20, lat_0=20, alpha=20, k0=1.0,
                                    a=6400.0)

    with pytest.raises(RuntimeError):
        HOM = HotineObliqueMercator(lonc=20, lat_0=20, alpha=20, k0=1.0,
                                    f=1.0/300.)

    with pytest.raises(AssertionError):
        HOM = HotineObliqueMercator(lonc=20, lat_0=20, alpha=20, k0=1.0,
                                    ellipsoid='garbage')

