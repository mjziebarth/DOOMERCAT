# Test the results of the Hotine scale factor 'k' computation from the C++ code
# base against some reference values obtained from an independent implementation
# using high precision arithmetic (mpmath).
#
# Author: Malte J. Ziebarth (malte.ziebarth@tum.de)
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

import numpy as np
from doomercat.cppextensions import compute_k_hotine


def test_k_code_cpp_phi_near_north():
    """
    Compares the result of the C++ code computing the scale factor k
    with results obtained from high precision arithmetics (mpmath).
    This function checks the approximations for phi_0 -> +90°.
    """
    PHI0 = 89.917
    F = 1/298.
    ALPHA=20.
    LONC = 21.3235
    K_0 = 0.9993
    A = 6_378_137

    # Fibonacci-lattice, reference k computed with high precision arithmetic
    # (400 digits precision):
    LON, LAT, k_ref, _ = np.loadtxt(
        'data/validation/k-cpp-validation-data.csv'
    ).T

    # Use C++ backend to compute scale factors:
    k_cpp = compute_k_hotine(LON, LAT, LONC, PHI0, ALPHA, K_0, F)

    # Make sure that the scale factors are comparable:
    assert np.allclose(k_ref, k_cpp, rtol=1e-10, atol=0.0)

    # Make sure that the disctortions are comparable:
    assert np.allclose(k_ref-1, k_cpp-1, rtol=1e-8, atol=0.0)


def test_k_code_cpp_phi_near_south():
    """
    Compares the result of the C++ code computing the scale factor k
    with results obtained from high precision arithmetics (mpmath).
    This function checks the approximations for phi_0 -> +90°.
    """
    PHI0 = -89.92327
    F = 1/298.
    ALPHA=20.
    LONC = 21.3235
    K_0 = 0.9993

    # Fibonacci-lattice, reference k computed with high precision arithmetic
    # (400 digits precision):
    LON, LAT, _, k_ref = np.loadtxt(
        'data/validation/k-cpp-validation-data.csv'
    ).T

    # Use C++ backend to compute scale factors:
    k_cpp = compute_k_hotine(LON, LAT, LONC, PHI0, ALPHA, K_0, F)

    # Make sure that the scale factors are comparable:
    print(k_ref - k_cpp)
    assert np.allclose(k_ref, k_cpp, rtol=1e-10, atol=0)

    # Make sure that the disctortions are comparable:
    assert np.allclose(k_ref-1, k_cpp-1, rtol=1e-8, atol=0.0)