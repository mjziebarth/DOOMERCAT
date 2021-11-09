# Common definitions for the Laborde oblique Mercator projection.
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

# References for reference ellipsoids:
#
# WGS84:
# EUROCONTROL European Organization for the Safety of Air Navigation,
# Institute of Geodesy and Navigation (IfEN) (1998):
# WGS 84 IMPLEMENTATION MANUAL, version 2.4
#
# GRS80:
# Moritz, H. Journal of Geodesy (2000) 74: 128. doi:10.1007/s001900050278
#
# IERS2003:
# McCarthy, D. D., Gérard, P. (2003): IERS CONVENTIONS, IERS Technical Note(32)



_ellipsoids = {'WGS84'    : (6378137., 298.257223563),
               'GRS80'    : (6378137., 1./0.00335281068118),
               'IERS2003' : (6378136.6, 298.25642)}
