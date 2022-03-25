# Test code for some optimization examples.
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
import json


def read_geojson(filename):
    """
    Reads places from a geojson.
    """
    with open(filename,'r') as f:
        geojson = json.load(f)

    lola = np.array([f["geometry"]['coordinates'] for f in geojson["features"]])

    return lola[:,0], lola[:,1]


def test_chile_places():
    """
    Example of populated places in Chile with more then 10000 inhabitants.
    Data taken from OSM export (HOT Export Tool), see data/chile/README.txt,
    and subselected in QGIS.
    """
    lon, lat = read_geojson('data/chile/Chile-cities-select.geojson')

    HOM = HotineObliqueMercator(lon=lon, lat=lat)

    # Test agains reference values:
    assert abs(HOM.lat_0() - (-22.74393664)) < 1e-8
    assert abs(HOM.lonc()  - (-70.55450760)) < 1e-8
    assert abs(HOM.alpha() - (4.52169941)) < 1e-8
    assert abs(HOM.k0()    - (0.99997618)) < 1e-8


def test_australia_small_circle():
    """
    Example of an artificial small circle generated around Australia.

    Test here the impact of the starting point.
    """
    lon, lat = read_geojson('data/small-circle-australia.geojson')

    HOM = HotineObliqueMercator(lon=lon, lat=lat, k0_ap=0.0,
                                lonc0=170.0, lat_00=-10.0, alpha0=-45.0,
                                k00=0.5)

    # Test agains reference values:
    assert abs(HOM.lat_0() - (44.70375631)) < 1e-8
    assert abs(HOM.lonc()  - (164.90948924)) < 1e-8
    assert abs(HOM.alpha() - (164.90948924)) < 1e-8
    assert abs(HOM.k0()    - (0.75295376)) < 1e-8
