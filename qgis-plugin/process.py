# Script that can be called stand-alone from a separate process
# to perform one inversion instance.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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

import os
import sys

# argv[1] will contain the pickled arguments.
# argv[2] will contain the path from which to execute.
if len(sys.argv) < 3:
    raise RuntimeError("No optimization data file and working directory "
                       "given.")

# Change to the working directory (the QGIS plugin path):
os.chdir(sys.argv[2])

# Now load the data and the LOM optimization code:
from pickle import Unpickler
from doomercat import LabordeObliqueMercator

with open(sys.argv[1],'rb') as f:
    args, kwargs = Unpickler(f).load()

# Perform the optimization:
LOM = LabordeObliqueMercator(*args, **kwargs)

# Return the result throught stdout:
print("PROJ{",LOM.proj4_string(),"}")
