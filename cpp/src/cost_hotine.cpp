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


static real4v sum(std::vector<real4v>& x){
	return real4v::sum(x);
}

// Make sure that Kahan summation is not killed by re-association:
#pragma GCC optimize("-fno-associative-math")
static double sum(std::vector<double>& x)
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


template<typename T>
static T compute_cost(const DataSet& data,
                      const HotineObliqueMercator<T> hom,
                      double pnorm, double k0_ap, double sigma_k0,
                      bool logarithmic, bool parallel)
{
	typedef Arithmetic<T> AR;

	/* Compute the distortions: */
	std::vector<T> cost_vec(data.size(), AR::constant(0.0));
	#pragma omp parallel for if(parallel)
	for (size_t i=0; i<data.size(); ++i){
		cost_vec[i] = AR::abs(hom.k(data.lambda(i), data.phi(i)) - 1.0);
	}

	/* Compute the maximum distortion: */
	T distmax(cost_vec[0]);
	for (size_t i=1; i<data.size(); ++i){
		if (cost_vec[i] > distmax)
			distmax = cost_vec[i];
	}

	/* Extract the factor (distmax)**pnorm and calculate the cost: */
	if (pnorm == static_cast<int>(pnorm) && pnorm < 5){
		const int ipnorm = static_cast<int>(pnorm);
		#pragma omp parallel for if(parallel)
		for (size_t i=0; i<data.size(); ++i){
			cost_vec[i] = data.w(i) * AR::pow(cost_vec[i] / distmax, ipnorm);
		}
	} else {
		#pragma omp parallel for if(parallel)
		for (size_t i=0; i<data.size(); ++i){
			cost_vec[i] = data.w(i) * AR::pow(cost_vec[i] / distmax, pnorm);
		}
	}

	T cost(AR::pow(sum(cost_vec), 1.0/pnorm));


	/* Add the k0  prior: */
	if (hom.k0() < k0_ap){
		cost += AR::pow2((hom.k0() - k0_ap)/sigma_k0)
		        * AR::pow(distmax, -pnorm);
	}

	if (logarithmic){
		return AR::log(cost) + AR::log(distmax);
	}

	return cost * distmax;
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


template<>
CostHotine<real4v>
CostFunctionHotine<real4v>::operator()(const DataSet& data,
                              const HotineObliqueMercator<real4v>& hom) const
{
	return CostHotine<real4v>(
	            compute_cost<real4v>(data, hom, pnorm, k0_ap,
	                                 sigma_k0, logarithmic, parallel)
	);
}


template<>
CostHotine<double>
CostFunctionHotine<double>::operator()(const DataSet& data,
                              const HotineObliqueMercator<double>& hom) const
{
	return CostHotine<double>(
	            compute_cost<double>(data, hom, pnorm, k0_ap,
	                                 sigma_k0, logarithmic, parallel)
	);
}





/***************************************************************************
 *                          Cost with p=infty                              *
 ***************************************************************************/
template<typename T>
static T compute_cost_inf(const DataSet& data,
                          const HotineObliqueMercator<T> hom,
                          double k0_ap, double sigma_k0,
                          bool logarithmic, bool parallel)
{
	typedef Arithmetic<T> AR;

	/* Compute the absolute of the distortions: */
	std::vector<T> cost_vec(data.size(), AR::constant(0.0));
	#pragma omp parallel for if(parallel)
	for (size_t i=0; i<data.size(); ++i){
		cost_vec[i] = AR::abs(hom.k(data.lambda(i), data.phi(i)) - 1.0);
	}

	T cost(cost_vec[0]);
	for (size_t i=1; i<data.size(); ++i){
		if (cost_vec[i] > cost)
			cost = cost_vec[i];
	}

	/* Add the k0  prior: */
	if (hom.k0() < k0_ap){
		cost += AR::pow2((hom.k0() - k0_ap)/sigma_k0);
	}

	if (logarithmic)
		return AR::log(cost);

	return cost;
}



template<>
CostHotine<real4v>
CostFunctionHotineInf<real4v>::operator()(const DataSet& data,
                              const HotineObliqueMercator<real4v>& hom) const
{
	return CostHotine<real4v>(
	            compute_cost_inf<real4v>(data, hom, k0_ap, sigma_k0,
	                                     logarithmic, parallel)
	);
}


template<>
CostHotine<double>
CostFunctionHotineInf<double>::operator()(const DataSet& data,
                              const HotineObliqueMercator<double>& hom) const
{
	return CostHotine<double>(
	            compute_cost_inf<double>(data, hom, k0_ap, sigma_k0,
	                                     logarithmic, parallel)
	);
}

