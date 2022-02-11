/*
 * Laborde-projected data set.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
 *          Sebastian von Specht
 *
 * Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam,
 *                         Sebastian von Specht
 *
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
#include <../include/projecteddata.hpp>
#include <../include/functions.hpp>
#include <cmath>
#include <iostream>

using doomercat::LabordeCylinder;
using doomercat::LabordeProjectedDataSet;
using doomercat::DataSet;

double Gd_inv(const double x) {
	return std::atanh(std::sin(x));
}


void LabordeProjectedDataSet::compute_uvw(
             const std::vector<DataSet::entry_t>& lola,
             const LabordeCylinder& cyl)
{
	_uvw.reserve(lola.size());

	const real5v& B = cyl.B();
	const real5v& C = cyl.C();
	const double e = cyl.e();
	for (const DataSet::entry_t& x : lola){
		// q = C + B*(Gd_inv(phi_) + 0.5 * e * np.log((1.0 - e * sin_phi)
		//                                           / (1.0 + e * sin_phi)))
		const double sp = std::sin(x.phi);
		const double B_prod = Gd_inv(x.phi) + 0.5*e*std::log((1.0 - e * sp)
		                                                     /(1.0 + e * sp));

		// P = 2.0*np.arctan(np.exp(q)) - 0.5*np.pi
		const real5v P(2.0 * atan(exp(C + B * B_prod)) - 0.5 * PI);

		// L = B * (lambda_ - lambda_c)
		// However, the function of lambda_c rotation is absorbed into the
		// Quaternion
		const real5v L(B * x.lambda);

		/* U = cosP * cos(L)
		 * V = cosP * sin(L)
		 * W = sin(P)
		 */
		const real5v cosP(cos(P));
		_uvw.push_back({cosP * cos(L), cosP * sin(L), sin(P)});
	}
}


LabordeProjectedDataSet::LabordeProjectedDataSet(
                               std::shared_ptr<const DataSet> data,
                               std::shared_ptr<const LabordeCylinder> cylinder)
    : data(data), cyl(cylinder)
{
	if (!cylinder)
		throw std::runtime_error("Nullpointer cylinder in "
		                         "LabordeProjectedDataSet");
	if (!data)
		throw std::runtime_error("Nullpointer data in "
		                         "LabordeProjectedDataSet");
	compute_uvw(data->data, *cylinder);
}

double LabordeProjectedDataSet::w(size_t i) const
{
	return data->w(i);
}

size_t LabordeProjectedDataSet::size() const
{
	return _uvw.size();
}


const LabordeCylinder& LabordeProjectedDataSet::cylinder() const
{
	return *cyl;
}

const vec3_5v& LabordeProjectedDataSet::uvw(size_t i) const
{
	if (i >= _uvw.size())
		throw std::runtime_error("UVW out of bound requested in "
		                         "LabordeProjectedDataSet");
	return _uvw[i];
}

