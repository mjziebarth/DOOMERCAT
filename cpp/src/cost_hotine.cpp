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

typedef Arithmetic<real4v> AR;

static real4v sum(std::vector<real4v>& x){
	return real4v::sum(x);
}


static real4v compute_cost(const DataSet& data,
                           const HotineObliqueMercator<real4v> hom,
                           unsigned int pnorm, double k0_ap, double sigma_k0,
                           bool logarithmic)
{
	/* Compute the weighted, potentiated distortions: */
	std::vector<real4v> cost_vec(data.size(), constant4(0.0));
	#pragma omp parallel for
	for (size_t i=0; i<data.size(); ++i){
		cost_vec[i] = data.w(i)
		        * AR::pow(AR::abs(hom.k(data.lambda(i), data.phi(i)) - 1.0),
		                  static_cast<int>(pnorm));
	}

	real4v cost(sum(cost_vec));


	/* Add the k0  prior: */
	if (hom.k0() < k0_ap){
		cost += AR::pow2((hom.k0() - k0_ap)/sigma_k0);
	}

	if (logarithmic)
		return AR::log(cost);

	return cost;
}




CostHotine::CostHotine(const real4v& cost) : cost(cost)
{
}

CostHotine::operator double() const
{
	return cost.value();
}

std::array<double,4> CostHotine::grad() const
{
	std::array<double,4> g;
	for (int i=0; i<4; ++i){
		g[i] = cost.derivative(i);
	}
	return g;
}



CostFunctionHotine::CostFunctionHotine(unsigned int pnorm, double k0_ap,
                                       double sigma_k0, bool logarithmic)
    : pnorm(pnorm), k0_ap(k0_ap), sigma_k0(sigma_k0),
      logarithmic(logarithmic)
{
	if (pnorm < 2)
		throw std::runtime_error("pnorm too small");
}


CostHotine CostFunctionHotine::operator()(const DataSet& data,
                            const HotineObliqueMercator<real4v>& hom) const
{
	return CostHotine(compute_cost(data, hom, pnorm, k0_ap, sigma_k0,
	                               logarithmic));
}
