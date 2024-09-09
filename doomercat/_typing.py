# Typing.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2024 Technical University of Munich
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

# Trying to import the NDArray type from numpy.typing
# fails on Github runner with Python 3.9, so use np.ndarray
# as a backup for older Python versions:
import numpy as np
from typing import Any

def _get_dtype64():
    try:
        from numpy.typing import NDArray
        return NDArray[np.double]
    except TypeError:
        return Any

ndarray64 = _get_dtype64()