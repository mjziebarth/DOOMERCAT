# Public API for the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2021 Deutsches GeoForschungsZentrum Potsdam,
#               2024      Technical University of Munich
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

from .hom import HotineObliqueMercator as HotineObliqueMercator
from .shapefile import points_from_shapefile as points_from_shapefile
from .cppextensions import damped_bfgs_optimize as damped_bfgs_optimize
from .hotine import lm_adamax_optimize as lm_adamax_optimize
from .lomerror import OptimizeError as OptimizeError
from .geometry import desired_scale_factor as desired_scale_factor
from .config import change_default as change_default, \
                    save_defaults as save_defaults
from .fibonacci import fibonacci_lattice as fibonacci_lattice
