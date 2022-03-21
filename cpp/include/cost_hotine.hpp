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

template<typename T>
class CostFunctionHotine;


template<typename T>
class CostHotine {
/*
 *Result of a cost function evaluation.
 */
friend CostFunctionHotine<T>;

public:
	operator double() const;

	std::array<double,4> grad() const;

private:
	CostHotine(const T& cost);
	T cost;

};

template<typename T>
CostHotine<T>::CostHotine(const T& cost) : cost(cost)
{
}


template<typename T>
class CostFunctionHotine {
/*
 * A particular configuration of the cost function parameters.
 */
public:
	CostFunctionHotine(double pnorm, double k0_ap, double sigma_k0,
	                   bool logarithmic, bool parallel=true);

	CostHotine<T> operator()(const DataSet& data,
	                         const HotineObliqueMercator<T>& hom) const;

private:
	double pnorm;
	double k0_ap;
	double sigma_k0;
	bool logarithmic;
	bool parallel = true;

};


template<typename T>
CostFunctionHotine<T>::CostFunctionHotine(double pnorm,
                                          double k0_ap,
                                          double sigma_k0,
                                          bool logarithmic,
                                          bool parallel)
    : pnorm(pnorm), k0_ap(k0_ap), sigma_k0(sigma_k0),
      logarithmic(logarithmic), parallel(parallel)
{
	if (pnorm <= 0)
		throw std::runtime_error("pnorm too small");
}




} // Namespace

#endif
