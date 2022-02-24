/*
 * Cost function based on the Hotine oblique Mercator projection.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
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

#ifndef DOOMERCAT_COST_HOTINE_HPP
#define DOOMERCAT_COST_HOTINE_HPP

#include <../include/dataset.hpp>
#include <../include/hotine.hpp>

namespace doomercat {

class CostFunctionHotine;



class CostHotine {
/*
 *Result of a cost function evaluation.
 */
friend CostFunctionHotine;

public:
	operator double() const;

	std::array<double,4> grad() const;

private:
	CostHotine(const real4v& cost);
	real4v cost;

};


class CostFunctionHotine {
/*
 * A particular configuration of the cost function parameters.
 */
public:
	CostFunctionHotine(unsigned int pnorm, double k0_ap, double sigma_k0,
	                   bool logarithmic);

	CostHotine operator()(const DataSet& data,
	                      const HotineObliqueMercator<real4v>& hom) const;

private:
	unsigned int pnorm;
	double k0_ap;
	double sigma_k0;
	bool logarithmic;

};




} // Namespace

#endif
