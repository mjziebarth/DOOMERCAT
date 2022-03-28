/*
 * Methods suitable for interfacing with Ctypes.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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

#include <../include/optimize.hpp>
#include <../include/ctypesinterface.hpp>
#include <../include/cost_hotine.hpp>
#include <../include/hotine.hpp>
#include <iostream>
#include <memory>
#include <cmath>
#include <cstddef>
#include <vector>


// Bugfix code:
#include <chrono>
#include <thread>

using doomercat::DataSet;
using doomercat::CostFunctionHotine;
using doomercat::CostFunctionHotineInf;
using doomercat::CostHotine;
using doomercat::HotineObliqueMercator;


int compute_cost_hotine_batch(const size_t N, const double* lon,
        const double* lat, const double* w, const size_t M,
        const double* lonc, const double* lat0, const double* alpha,
        const double* k0, double f, double pnorm, double k0_ap,
        double sigma_k0, unsigned short proot,
        unsigned short logarithmic, double* result)
{
	/* Data set: */
	const DataSet data(N, lon, lat, w);

	/* Cost function: */
	if (std::isinf(pnorm)){
		const CostFunctionHotineInf<double> cfun(k0_ap, sigma_k0,
		                                         logarithmic > 0u,
		                                         false);

		/* Compute the Hotine Mercator projections now: */
		#pragma omp parallel for
		for (size_t i=0; i<M; ++i){
			HotineObliqueMercator<double>
			   hom(deg2rad(lonc[i]), deg2rad(lat0[i]),
			       deg2rad(alpha[i]), k0[i], f);

			/* Compute the cost: */
			result[i] = cfun(data, hom);
		}

	} else {
		const CostFunctionHotine<double> cfun(pnorm, k0_ap, sigma_k0,
		                                      proot > 0u, logarithmic > 0u,
		                                      false);

		/* Compute the Hotine Mercator projections now: */
		#pragma omp parallel for
		for (size_t i=0; i<M; ++i){
			HotineObliqueMercator<double>
			   hom(deg2rad(lonc[i]), deg2rad(lat0[i]),
			       deg2rad(alpha[i]), k0[i], f);

			/* Compute the cost: */
			result[i] = cfun(data, hom);
		}

	}

	return 0;
}


int compute_k_hotine(const size_t N, const double* lon,
        const double* lat, const double* w,
        double lonc, double lat0, double alpha, double k0, double f,
        double* result)
{
	/* Compute the Hotine Mercator projections now: */
	const HotineObliqueMercator<double>
	   hom(deg2rad(lonc), deg2rad(lat0), deg2rad(alpha), k0, f);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		/* Compute the cost: */
		result[i] = hom.k(deg2rad(lon[i]), deg2rad(lat[i]));
	}

	return 0;
}


int hotine_project(const size_t N, const double* lon,
        const double* lat, double lonc, double lat0, double alpha,
        double k0, double f, double* result)
{
	/* Compute the Hotine Mercator projections now: */
	const HotineObliqueMercator<double>
	   hom(deg2rad(lonc),deg2rad(lat0), deg2rad(alpha), k0, f);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		/* Compute the cost: */
		HotineObliqueMercator<double>::uv_t uv(hom.uv(deg2rad(lon[i]),
		                                              deg2rad(lat[i])));
		result[2*i]   = uv.u;
		result[2*i+1] = uv.v;
	}

	return 0;
}


template<>
double doomercat::to_double<double>(const double& t)
{
	return t;
}
template<>
double doomercat::to_double<real4v>(const real4v& t)
{
	return t.value();
}


int hotine_bfgs(const size_t N, const double* lon, const double* lat,
                const double* w, double f, double pnorm, double k0_ap,
                double sigma_k0, double lonc_0, double lat_0_0,
                double alpha_0, double k_0_0, unsigned int Nmax,
                unsigned short proot, double epsilon, double* result,
                unsigned int* n_steps)
{
	// Sanity check on weights (probably very much redundant):
	if (w == 0)
		w = nullptr;

	// Init the data set:
	DataSet data(N, lon, lat, w);

	/* Optimize: */
	std::vector<doomercat::hotine_result_t> history;
	if (std::isinf(pnorm)){
		history = bfgs_optimize_hotine_pinf(data, lonc_0, lat_0_0, alpha_0,
		                                    k_0_0, f, k0_ap, sigma_k0, Nmax,
		                                    epsilon);
	} else {
		history = bfgs_optimize_hotine(data, lonc_0, lat_0_0, alpha_0, k_0_0, f,
		                               pnorm, k0_ap, sigma_k0, Nmax,
		                               proot > 0u, epsilon);
	}

	/* Return the results: */
	for (size_t i=0; i<history.size(); ++i){
		result[10*i]   = history[i].cost;
		result[10*i+1] = history[i].lonc;
		result[10*i+2] = history[i].lat_0;
		result[10*i+3] = history[i].alpha;
		result[10*i+4] = history[i].k0;
		result[10*i+5] = history[i].grad_lonc;
		result[10*i+6] = history[i].grad_lat0;
		result[10*i+7] = history[i].grad_alpha;
		result[10*i+8] = history[i].grad_k0;
		result[10*i+9] = history[i].algorithm_state;
	}

	if (n_steps){
		*n_steps = static_cast<unsigned int>(history.size());
	}

	return 0;
}


int hotine_parameters_debug(double lonc, double lat0, double alpha,
                            double k0, double f, double* result)
{
	/* Return parameters E, gamma0, lambda0 for debugging purposes. */
	HotineObliqueMercator<real4v> hom(constant4(deg2rad(lonc)),
	                                  constant4(deg2rad(lat0)),
	                                  constant4(deg2rad(alpha)),
	                                  constant4(k0), f);

	result[0] = hom.E().value();
	result[1] = hom.gamma0().value();
	result[2] = hom.lambda0().value();

	return 0;
}
