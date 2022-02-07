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

#include <../include/labordecylinder.hpp>
#include <../include/cost.hpp>
#include <../include/optimize.hpp>
#include <iostream>
#include <memory>
#include <cmath>
#include <cstddef>
#include <vector>


// Bugfix code:
#include <chrono>
#include <thread>

extern "C" {
	int compute_cost(const size_t N, const double* lon,const double* lat,
	                 const double* w, double lon_cyl, double lat_cyl, double k0,
	                 double f, unsigned int pnorm, double k0_ap,
	                 double sigma_k0, double* result);

	int compute_cost_and_derivatives(const size_t N, const double* lon,
	                 const double* lat, const double* w, double lon_cyl,
	                 double lat_cyl, double k0, double f,  unsigned int pnorm,
	                 double k0_ap, double sigma_k0, double* result);

	int perform_billo_gradient_descent(const size_t N, const double* lon,
	                 const double* lat, double f,
	                 unsigned int pnorm, double k0_ap, double sigma_k0,
	                 double lon0, double lat0, const size_t Nmax,
	                 double* result);

	int perform_bfgs(const size_t N, const double* lon, const double* lat,
	                 const double* w, double f, unsigned int pnorm,
	                 double k0_ap, double sigma_k0, double lon0, double lat0,
	                 const size_t Nmax, double* result, unsigned int* n_steps);
}

using doomercat::DataSet;
using doomercat::LabordeCylinder;
using doomercat::LabordeProjectedDataSet;
using doomercat::CostFunction;
using doomercat::Cost;
using doomercat::result_t;
using doomercat::billo_gradient_descent;


int compute_cost(const size_t N, const double* lon, const double* lat,
                 const double* w, double lon_cyl, double lat_cyl, double k0,
                 double f, unsigned int pnorm, double k0_ap, double sigma_k0,
                 double* result)
{
	/* Compute the cost for a data set given a cylinder. */
	std::shared_ptr<LabordeCylinder>
	   cyl(LabordeCylinder::from_axis_lon_lat(lon_cyl, lat_cyl, k0, f));

	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, w);

	// Project and compute cost function:
	LabordeProjectedDataSet pd(data, cyl);

	*result = CostFunction(pnorm,k0_ap,sigma_k0)(pd);

	return 0;
}


int compute_cost_and_derivatives(const size_t N, const double* lon,
        const double* lat, const double* w, double lon_cyl, double lat_cyl,
        double k0, double f, unsigned int pnorm, double k0_ap, double sigma_k0,
        double* result)
{
	/* Compute the cost for a data set given a cylinder. */
	std::shared_ptr<LabordeCylinder>
	   cyl(LabordeCylinder::from_axis_lon_lat(lon_cyl, lat_cyl, k0, f));

	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, w);

	// Project and compute cost function:
	LabordeProjectedDataSet pd(data, cyl);

	Cost cost(CostFunction(pnorm,k0_ap,sigma_k0)(pd));

	result[0] = cost;
	std::array<double,5> grad = cost.grad();
	for (int i=0; i<5; ++i)
		result[i+1] = grad[i];

	return 0;
}

int perform_billo_gradient_descent(const size_t N, const double* lon,
        const double* lat, double f,
        unsigned int pnorm, double k0_ap, double sigma_k0,
        double lon0, double lat0, const size_t Nmax, double* result)
{
	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, nullptr);

	/* Initial cylinder: */
	std::shared_ptr<LabordeCylinder> cyl0
	   = LabordeCylinder::from_axis_lon_lat(lon0, lat0, 1.0, f);

	/* Cost function: */
	const CostFunction cost_fun(pnorm, k0_ap, sigma_k0);

	/* Optimize: */
	std::cout << "optimizing!\n" << std::flush;
	std::vector<doomercat::result_t> history
	   = billo_gradient_descent(data, cyl0, cost_fun, Nmax);

	/* Return the results: */
	for (size_t i=0; i<Nmax; ++i){
		result[4*i]   = history[i].cost;
		vec3_5v ax = history[i].cylinder.axis();
		result[4*i+1] = rad2deg(std::atan2(ax[1].value(), ax[0].value()));
		result[4*i+2] = rad2deg(std::asin(std::min(std::max(ax[2].value(),
		                                                    -1.0), 1.0)));
		result[4*i+3] = history[i].cylinder.k0().value();
	}

	return 0;
}

int perform_bfgs(const size_t N, const double* lon, const double* lat,
                 const double* w, double f, unsigned int pnorm, double k0_ap,
                 double sigma_k0, double lon0, double lat0, const size_t Nmax,
                 double* result, unsigned int* n_steps)
{
	// Sanity check on weights (probably very much redundant):
	if (w == 0)
		w = nullptr;

	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, w);

	/* Initial cylinder: */
	std::shared_ptr<LabordeCylinder> cyl0
	   = LabordeCylinder::from_axis_lon_lat(lon0, lat0, 1.0, f);

	/* Cost function: */
	const CostFunction cost_fun(pnorm, k0_ap, sigma_k0);

	/* Optimize: */
	std::vector<doomercat::result_t> history
	   = bfgs_optimize(data, cyl0, cost_fun, Nmax);

	/* Return the results: */
	for (size_t i=0; i<history.size(); ++i){
		result[6*i]   = history[i].cost;
		result[6*i+1] = history[i].cylinder.lat_0().value();
		result[6*i+2] = history[i].cylinder.azimuth().value();
		result[6*i+3] = history[i].cylinder.k0().value();
		// For debugging, save lon & lat at which the cylinder axis strike
		// the Laborde sphere:
		vec3_5v ax = history[i].cylinder.axis();
		result[6*i+4] = rad2deg(std::atan2(ax[1].value(), ax[0].value()));
		result[6*i+5] = rad2deg(std::asin(std::min(std::max(ax[2].value(),
		                                                    -1.0), 1.0)));
	}

	if (n_steps)
		*n_steps = history.size();

	return 0;
}
