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
using doomercat::CostHotine;
using doomercat::HotineObliqueMercator;


int compute_cost_hotine_batch(const size_t N, const double* lon,
        const double* lat, const double* w, const size_t M,
        const double* lonc, const double* lat0, const double* alpha,
        const double* k0, double f, unsigned int pnorm, double k0_ap,
        double sigma_k0, double* result)
{
	/* Data set: */
	const DataSet data(N, lon, lat, w);

	/* Cost function: */
	const CostFunctionHotine cfun(pnorm, k0_ap, sigma_k0, true);

	/* Compute the Hotine Mercator projections now: */
	#pragma omp parallel for
	for (size_t i=0; i<M; ++i){
		HotineObliqueMercator<real4v>
		   hom(constant4(deg2rad(lonc[i])), constant4(deg2rad(lat0[i])),
		       constant4(deg2rad(alpha[i])), constant4(k0[i]), f);

		/* Compute the cost: */
		result[i] = cfun(data, hom);
	}

	return 0;
}


int compute_k_hotine(const size_t N, const double* lon,
        const double* lat, const double* w,
        double lonc, double lat0, double alpha, double k0, double f,
        double* result)
{
	/* Compute the Hotine Mercator projections now: */
	const HotineObliqueMercator<real4v>
	   hom(constant4(deg2rad(lonc)), constant4(deg2rad(lat0)),
	       constant4(deg2rad(alpha)), constant4(k0), f);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		/* Compute the cost: */
		result[i] = hom.k(deg2rad(lon[i]), deg2rad(lat[i])).value();
	}

	return 0;
}


int hotine_bfgs(const size_t N, const double* lon, const double* lat,
                const double* w, double f, unsigned int pnorm, double k0_ap,
                double sigma_k0, double lonc_0, double lat_0_0,
                double alpha_0, double k_0_0, const size_t Nmax,
                double* result, unsigned int* n_steps)
{
	// Sanity check on weights (probably very much redundant):
	if (w == 0)
		w = nullptr;

	// Init the data set:
	DataSet data(N, lon, lat, w);

	/* Optimize: */
	std::vector<doomercat::hotine_result_t> history
	   = bfgs_optimize_hotine(data, lonc_0, lat_0_0, alpha_0, k_0_0, f,
	                          pnorm, k0_ap, sigma_k0, Nmax, false);

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

	if (n_steps)
		*n_steps = history.size();

	return 0;
}
