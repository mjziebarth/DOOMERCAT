/*
 * Cost function.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
 *          Sebastian von Specht
 *
 * Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam,
 *                         Sebastian von Specht
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

#ifndef DOOMERCAT_COST_HPP
#define DOOMERCAT_COST_HPP

namespace doomercat {

class Cost {
/*
 *Result of a cost function evaluation.
 */
public:
	Cost(const LabordeProjectedDataSet& projected_data,
	     unsigned int pnorm, double k0_ap, double sigma_k0);

	operator double() const;

	std::array<double,5> grad() const;

private:
	real5v cost;

};


class CostFunction {
/*
 * A particular configuration of the cost function parameters.
 */
public:
	CostFunction(unsigned int pnorm, double k0_ap, double sigma_k0);

	Cost operator()(const LabordeProjectedDataSet& data) const;

private:
	unsigned int pnorm;
	double k0_ap;
	double sigma_k0;

};

}

#endif
