# Setup script for the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam
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

# Imports:
from setuptools import setup, Extension
from compile import setup_compile


# Compile the C++ code:
libname = setup_compile()

# Setup:

setup(
	name='doomercat',
	version='1.0.0',
	description="Data-Optimized Oblique MERCATor",
	long_description="Algorithm and convenience class to optimize a Laborde "
	                 "oblique Mercator projection for a geospatial data set "
	                 "minimizing distortion.",
	author='Malte J. Ziebarth, Sebastian von Specht',
	author_email='ziebarth@gfz-potsdam.de',
	packages=['doomercat'],
	py_modules=['doomercat'],
	provides=['doomercat'],
	package_data={'doomercat' : [libname]},
	scripts=[],
	install_requires=['numpy'],
	license='EUPLv1.2',
)
