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

#include <../include/cost_hotine.hpp>
#include <vector>
#include <algorithm>

using doomercat::DataSet;
using doomercat::HotineObliqueMercator;
using doomercat::CostHotine;
using doomercat::CostFunctionHotine;
using doomercat::CostFunctionHotineInf;


template<>
real4v CostFunctionHotine<real4v>::sum(const std::vector<real4v>& x){
	return real4v::sum(x);
}

// Make sure that Kahan summation is not killed by re-association:
#pragma GCC optimize("-fno-associative-math")
template<>
double CostFunctionHotine<double>::sum(const std::vector<double>& x)
{
	/* Copy of the code from autodouble.hpp */
	long double S = 0.0;
	long double comp = 0.0;
	for (const double& xi : x){
		long double add = xi - comp;
		long double res = S + add;
		comp = (res - S) - add;
		S = res;
	}

	return static_cast<double>(S);
}

template<>
CostHotine<real4v>::operator double() const
{
	return cost.value();
}

template<>
CostHotine<double>::operator double() const
{
	return cost;
}


template<>
std::array<double,4> CostHotine<real4v>::grad() const
{
	std::array<double,4> g;
	for (dim_t i=0; i<4; ++i){
		g[i] = cost.derivative(i);
	}
	return g;
}





/***************************************************************************
 *                          Cost with p=infty                              *
 ***************************************************************************/

