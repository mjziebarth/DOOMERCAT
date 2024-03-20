/*
 * Hotine parameters.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2024 Technische Universität München,
 *               2022 Deutsches GeoForschungsZentrum Potsdam
 *
 * Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence");
 * You may not use this work except in compliance with the Licence.
 * You may obtain a copy of the Licence at:
 *
 * https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

#ifndef DOOMERCAT_PARAMETERS_HPP
#define DOOMERCAT_PARMAETERS_HPP

#include <array>
#include <cmath>
#include <../include/constants.hpp>

namespace doomercat {



template<typename real>
real fmod(real a, real b) {
    /* True modulo operation (similar to Python's (a % b)).
        * Implemented here only for positive b (which is what we use).
        */
    real y = std::fmod(a, b);
    if (y < 0.0)
        return y+b;
    return y;
};

template<typename real>
struct HotineParameters
{
public:
    constexpr size_t d = 4;
    typedef real real_t;

	HotineParameters(real lambda_c, real phi_0, real alpha, real k0)
	{

        /* Latitude flip: */
        if (phi_0 < -PI/2 || phi_0 > PI/2){
            /* Shift latitudes to the range [0,180],
             * and its winding multiples, to correspond
             * better with integer indices:
             */
            phi_0 += PI / 2;
            /* This covers the whole plane within the
             * long range. Get the current winding index:
             */
            long winding = std::floor(phi_0 / PI);
            long abs_wind = std::abs(winding);

            /* Shift to the range [0,180]: */
            lat_0 -= PI * winding;

            /* Now handle the odd flip: */
            if ((abs_wind % 2) == 1){
                /* Need to turn around the rotation axis
                 * by 180°: */
                lonc += PI;

                /* Flip orientation of north and south: */
                lat_0 = PI - lat_0;
            }
            /* Examples:
             * -1 -> 179 -> 180 - 179 = 1
             * 181 -> 1 -> 180 - 1 = 179
             */

            /* Shift back: */
            params[1] = lat_0 - PI/2;
        } else {
            params[1] = lat_0;
        }

		/* Longitude in range [-180, 180]: */
		params[0] = fmod(lonc + PI, 2*PI) - PI;

        /* Central angle in range [-90, 90] */
        params[2] = fmod(alpha_c + PI/2, PI) - PI/2;

        params[3] = k0;
	}


    HotineParameters(const std::array<real,4>& p)
       : HotineParameters(p[0], p[1], p[2], p[3])
    {
    }

    HotineParameters(HotineParameters&&) = default;
    HotineParameters(const HotineParameters&) = default;


	operator std::array<real,4>() const
	{
		return params;
	};

    constexpr const real& operator[](uint_fast8_t i) const
    {
        return params[i];
    }

private:
	std::array<real,4> params;
};


} // end namespace

#endif