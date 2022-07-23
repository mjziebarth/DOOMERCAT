# Module loader customized to load the doomercat plugin as a relative import.
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from pathlib import Path

# Find the doomercat module inside the QGIS plugin directory:
_plugin_dir = Path(__file__).parent
sys.path.insert(0,str(_plugin_dir / "module"))

# See whether the C++ code is available:
from doomercat.cppextensions import find_cppextensions_file

try:
    HAS_CPPEXTENSIONS = find_cppextensions_file() is not None
except:
    HAS_CPPEXTENSIONS = False

# Load doomercat imports:
from doomercat import HotineObliqueMercator
from doomercat.defs import _ellipsoids
