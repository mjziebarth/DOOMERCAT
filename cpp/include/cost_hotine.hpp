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

#include <../include/autodouble.hpp>
#include <../include/dataset.hpp>
#include <../include/hotine.hpp>

#include <algorithm>

namespace doomercat {

template<typename T>
class CostFunctionHotine;

template<typename T>
class CostFunctionHotineInf;


template<typename T>
class CostHotine {
/*
 *Result of a cost function evaluation.
 */
friend CostFunctionHotine<T>;
friend CostFunctionHotineInf<T>;

public:
	typedef Arithmetic<T> AR;
	typedef typename AR::numeric_type numeric_t;

	operator numeric_t() const
	{
		return cost.value();
	}

	std::array<numeric_t,4> grad() const
	{
		std::array<numeric_t,4> g;
		for (dim_t i=0; i<4; ++i){
			g[i] = cost.derivative(i);
		}
		return g;
	}

private:
	CostHotine(const T& cost);
	T cost;

};

/* Declare some specifications: */
template<>
CostHotine<double>::operator double() const;

template<>
CostHotine<long double>::operator long double() const;

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
	                   bool proot, bool logarithmic, bool parallel=true);

	template<typename DS>
	CostHotine<T> operator()(const DS& data,
	                         const HotineObliqueMercator<T>& hom) const
	{
		return compute_cost(data, hom);
	};

private:
	typedef Arithmetic<T> AR;
	typedef typename AR::numeric_type number_t;
	number_t pnorm;
	number_t k0_ap;
	number_t sigma_k0;
	bool logarithmic;
	bool parallel = true;
	bool proot = true;

	static T sum(const std::vector<T>& x)
	{
		return AR::sum(x);
	}

	template<typename DS>
	T compute_cost(const DS& data,
	               const HotineObliqueMercator<T>& hom) const;
};


template<typename T>
CostFunctionHotine<T>::CostFunctionHotine(double pnorm,
                                          double k0_ap,
                                          double sigma_k0,
                                          bool proot,
                                          bool logarithmic,
                                          bool parallel)
    : pnorm(pnorm), k0_ap(k0_ap), sigma_k0(sigma_k0),
      logarithmic(logarithmic), parallel(parallel), proot(proot)
{
	if (pnorm <= 0)
		throw std::runtime_error("pnorm too small");
}


/*
 * Implementation of the cost function:
 */
template<typename T>
template<typename DS>
T CostFunctionHotine<T>::compute_cost(const DS& data,
                                      const HotineObliqueMercator<T>& hom) const
{
	typedef Arithmetic<T> AR;

	/* Compute the distortions: */
	std::vector<T> cost_vec(data.size(), AR::constant(0.0));
	#pragma omp parallel for if(parallel)
	for (size_t i=0; i<data.size(); ++i){
		cost_vec[i] = AR::abs(hom.k(data.lambda(i), data.phi(i))
		                      - data.k_e(i));
	}

	/* Compute the maximum distortion: */
	T distmax(cost_vec[0]);
	for (size_t i=1; i<data.size(); ++i){
		if (cost_vec[i] > distmax)
			distmax = cost_vec[i];
	}

	/* Extract the factor (distmax)**pnorm and calculate the cost: */
	if (pnorm == static_cast<double>(static_cast<int>(pnorm)) && pnorm < 5){
		const int ipnorm = static_cast<int>(pnorm);
		//#pragma omp parallel for if(parallel)
		for (size_t i=0; i<data.size(); ++i){
			const number_t wi = data.w(i);
			cost_vec[i] = wi * AR::pow(cost_vec[i] / distmax, ipnorm);
		}
	} else {
		//#pragma omp parallel for if(parallel)
		for (size_t i=0; i<data.size(); ++i){
			const number_t wi = data.w(i);
			cost_vec[i] = wi * AR::pow(cost_vec[i] / distmax, pnorm);
		}
	}

	if (proot){
		T cost(AR::pow(sum(cost_vec), 1.0/pnorm));


		/* Add the k0  prior: */
		if (hom.k0() < k0_ap){
			cost += AR::pow2((hom.k0() - k0_ap)/sigma_k0)
			        / distmax;
		}

		if (logarithmic){
				return AR::log(cost) + AR::log(distmax);
		}

		return cost * distmax;

	} else {
		T cost(CostFunctionHotine::sum(cost_vec));

		/* Add the k0  prior: */
		if (hom.k0() < k0_ap){
			cost += AR::pow2((hom.k0() - k0_ap)/sigma_k0)
				    * AR::pow(distmax, -pnorm);
		}

		if (logarithmic){
			return AR::log(cost) + pnorm*AR::log(distmax);
		}

		return cost * AR::pow(distmax,pnorm);
	}
}


/******************************************************************************
 *                                                                            *
 *                         Cost function with p=inf.                          *
 *                                                                            *
 ******************************************************************************/

template<typename T>
class CostFunctionHotineInf {
/*
 * A particular configuration of the cost function parameters.
 */
public:
	CostFunctionHotineInf(double k0_ap, double sigma_k0,
	                      bool logarithmic, bool parallel=true);

	template<typename DS>
	CostHotine<T> operator()(const DS& data,
	                         const HotineObliqueMercator<T>& hom) const
	{
		return compute_cost(data, hom);
	};

private:
	double k0_ap;
	double sigma_k0;
	bool logarithmic;
	bool parallel = true;

	template<typename DS>
	T compute_cost(const DS& data,
	               const HotineObliqueMercator<T>& hom) const;
};


template<typename T>
CostFunctionHotineInf<T>::CostFunctionHotineInf(double k0_ap,
                                                double sigma_k0,
                                                bool logarithmic,
                                                bool parallel)
    : k0_ap(k0_ap), sigma_k0(sigma_k0),
      logarithmic(logarithmic), parallel(parallel)
{
}


/*
 * Evaluating the const function for p=inf:
 */
template<typename T>
template<typename DS>
T CostFunctionHotineInf<T>::compute_cost(const DS& data,
                          const HotineObliqueMercator<T>& hom) const
{
	typedef Arithmetic<T> AR;
	typedef typename AR::numeric_type numeric_t;

	/* Compute the absolute of the distortions (here without taking
	 * into consideration the derivatives since we need it only for the
	 * largest distortion): */
	std::vector<numeric_t> cost_vec(data.size(), 0.0);
	if constexpr (std::is_same_v<numeric_t, T>){
		#pragma omp parallel for if(parallel)
		for (size_t i=0; i<data.size(); ++i){
			cost_vec[i] = std::abs(hom.k(data.lambda(i), data.phi(i))
			                       - data.k_e(i));
		}
	} else {
		HotineObliqueMercator<numeric_t> homd(hom);
		#pragma omp parallel for if(parallel)
		for (size_t i=0; i<data.size(); ++i){
			cost_vec[i] = std::abs(homd.k(data.lambda(i), data.phi(i))
			                       - data.k_e(i));
		}
	}

	/* Find the maximum cost: */
	size_t imin=0;
	numeric_t costd(cost_vec[0]);
	for (size_t i=1; i<data.size(); ++i){
		if (cost_vec[i] > costd){
			costd = cost_vec[i];
			imin = i;
		}
	}

	/* Get the minimum cost, possibly with derivatives: */
	T cost;
	if constexpr (std::is_same_v<numeric_t, T>){
		cost = costd;
	} else {
		cost = AR::abs(hom.k(data.lambda(imin), data.phi(imin))
		       - data.k_e(imin));
	}

	/* Add the k0  prior: */
	if (hom.k0() < k0_ap){
		cost += AR::pow2((hom.k0() - k0_ap)/sigma_k0);
	}

	if (logarithmic)
		return AR::log(cost);

	return cost;
}




} // Namespace

#endif
