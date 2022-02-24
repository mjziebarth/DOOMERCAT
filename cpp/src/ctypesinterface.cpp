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
#include <../include/ctypesinterface.hpp>
#include <iostream>
#include <memory>
#include <cmath>
#include <cstddef>
#include <vector>


// Bugfix code:
#include <chrono>
#include <thread>

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
        double lonc, double k0, double f, unsigned int pnorm, double k0_ap,
        double sigma_k0, double* result)
{
	/* Compute the cost for a data set given a cylinder. */
	std::shared_ptr<LabordeCylinder>
	   cyl(LabordeCylinder::from_axis_and_central(lon_cyl, lat_cyl, lonc, k0,
	                                              f));

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



int perform_adam(const size_t N, const double* lon, const double* lat,
                 const double* w, double f, unsigned int pnorm, double k0_ap,
                 double sigma_k0, double lon_cyl0, double lat_cyl0,
                 double lonc0, double k00, const size_t Nmax, double* result,
                 unsigned int* n_steps, double* final_cylinder)
{
	// Sanity check on weights (probably very much redundant):
	if (w == 0)
		w = nullptr;

	std::cout << "Parameters passed to BFGS:\n";
	std::cout.precision(17);
	std::cout << "  N =        " << N << "\n";
	std::cout << "  f =        " << f << "\n";
	std::cout << "  pnorm =    " << pnorm << "\n";
	std::cout << "  k0_ap =    " << k0_ap << "\n";
	std::cout << "  sigma =    " << sigma_k0 << "\n";
	std::cout << "  lon_cyl0 = " << lon_cyl0 << "\n";
	std::cout << "  lat_cyl0 = " << lat_cyl0 << "\n";
	std::cout << "  lonc0 =    " << lonc0 << "\n";
	std::cout << "  k00 =      " << k00 << "\n";
	std::cout << "  Nmax =     " << Nmax << "\n";
	std::cout << "  First 7 coordinates:\n  [";
	for (size_t i=0; i<std::min<size_t>(7,Nmax); ++i){
		std::cout << "(" << lon[i] << "," << lat[i] << "), ";
	}
	std::cout << "\n" << std::flush;

	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, w);

	/* Initial cylinder: */
	std::shared_ptr<LabordeCylinder> cyl0
	   = LabordeCylinder::from_axis_and_central(lon_cyl0, lat_cyl0, lonc0,
	                                            k00, f);

	/* Compute the correction factor in k0 at the starting central location: */
	const double k0_corr = cyl0->k0_correction();
	std::cout << "k0_corr: " << k0_corr << "\n";

	/* Cost function: */
	const CostFunction cost_fun(pnorm, k0_ap, sigma_k0);
	std::cout << "k0_ap: " << k0_ap << "\n" << std::flush;

	/* Optimize: */
	std::vector<doomercat::result_t> history
	   = adam_optimize(data, cyl0, cost_fun, Nmax);

	/* Return the results: */
	for (size_t i=0; i<history.size(); ++i){
		result[7*i]   = history[i].cost;
		result[7*i+1] = history[i].cylinder.lonc().value();
		result[7*i+2] = history[i].cylinder.lat_0().value();
		result[7*i+3] = history[i].cylinder.azimuth().value();
		// Correct k0 by the relative ellipsoid radius at the initial coordinate:
		result[7*i+4] = history[i].cylinder.k0().value() * k0_corr;
		// For debugging, save lon & lat at which the cylinder axis strike
		// the Laborde sphere:
		vec3_5v ax = history[i].cylinder.axis();
		result[7*i+5] = rad2deg(std::atan2(ax[1].value(), ax[0].value()));
		result[7*i+6] = rad2deg(std::asin(std::min(std::max(ax[2].value(),
		                                                    -1.0), 1.0)));
	}

	std::cout << "k0 last: " << history.back().cylinder.k0().value()
	          << "\n" << std::flush;


	/* An exact representation of the final cylinder in terms of
	 * the rotation quaternion, k0, and f: */
	if (final_cylinder){
		std::array<double,6> cyl_last
		   = history.back().cylinder.representation();
		for (int i=0; i<6; ++i){
			final_cylinder[i] = cyl_last[i];
		}
	}

	if (n_steps)
		*n_steps = history.size();

	return 0;
}





int perform_bfgs(const size_t N, const double* lon, const double* lat,
                 const double* w, double f, unsigned int pnorm, double k0_ap,
                 double sigma_k0, double lon_cyl0, double lat_cyl0,
                 double lonc0, double k00, const size_t Nmax, double* result,
                 unsigned int* n_steps, double* final_cylinder)
{
	// Sanity check on weights (probably very much redundant):
	if (w == 0)
		w = nullptr;

	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, w);

	/* Initial cylinder: */
	std::shared_ptr<LabordeCylinder> cyl0
	   = LabordeCylinder::from_axis_and_central(lon_cyl0, lat_cyl0, lonc0,
	                                            k00, f);

	/* Compute the correction factor in k0 at the starting central location: */
	const double k0_corr = cyl0->k0_correction();
	std::cout << "k0_corr: " << k0_corr << "\n";

	/* Cost function: */
	const CostFunction cost_fun(pnorm, k0_ap, sigma_k0);
	std::cout << "k0_ap: " << k0_ap << "\n" << std::flush;

	/* Optimize: */
	std::vector<doomercat::result_t> history
	   = bfgs_optimize(data, cyl0, cost_fun, Nmax);

	/* Return the results: */
	for (size_t i=0; i<history.size(); ++i){
		result[7*i]   = history[i].cost;
		result[7*i+1] = history[i].cylinder.lonc().value();
		result[7*i+2] = history[i].cylinder.lat_0().value();
		result[7*i+3] = history[i].cylinder.azimuth().value();
		// Correct k0 by the relative ellipsoid radius at the initial coordinate:
		result[7*i+4] = history[i].cylinder.k0().value() * k0_corr;
		// For debugging, save lon & lat at which the cylinder axis strike
		// the Laborde sphere:
		vec3_5v ax = history[i].cylinder.axis();
		result[7*i+5] = rad2deg(std::atan2(ax[1].value(), ax[0].value()));
		result[7*i+6] = rad2deg(std::asin(std::min(std::max(ax[2].value(),
		                                                    -1.0), 1.0)));
	}

	std::cout << "k0 last: " << history.back().cylinder.k0().value()
	          << "\n" << std::flush;


	/* An exact representation of the final cylinder in terms of
	 * the rotation quaternion, k0, and f: */
	if (final_cylinder){
		std::array<double,6> cyl_last
		   = history.back().cylinder.representation();
		for (int i=0; i<6; ++i){
			final_cylinder[i] = cyl_last[i];
		}
	}

	if (n_steps)
		*n_steps = history.size();

	return 0;
}



int laborde_project(size_t N, const double* lon, const double* lat,
                    double qr, double qi, double qj, double qk, double k0,
                    double f, double a, double* xy)
{
	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, nullptr);

	// Init the cylinder:
	std::shared_ptr<LabordeCylinder> cyl
	   = std::make_shared<LabordeCylinder>(qr, qi, qj, qk, k0, f);

	// Projected data set:
	LabordeProjectedDataSet pd(data, cyl);

	// Project:
	std::vector<coordinate_t> vec_xy(pd.projected(a));
	for (size_t i=0; i<N; ++i){
		xy[2*i] = vec_xy[i].x;
		xy[2*i+1] = vec_xy[i].y;
	}

	return 0;
}

int compute_cost_k0_iter(const size_t N, const double* lon,
        const double* lat, const double* w, const size_t M,
        const double* k0, double qr, double qi, double qj, double qk,
        double f,
        unsigned int pnorm, double k0_ap, double sigma_k0,
        double* cost_result)
{
	// Init the data set:
	std::shared_ptr<DataSet> data
		= std::make_shared<DataSet>(N, lon, lat, nullptr);

	for (size_t i=0; i<M; ++i){
		/* Compute the cost for a data set given a cylinder. */
		std::shared_ptr<LabordeCylinder> cyl
		  = std::make_shared<LabordeCylinder>(qr, qi, qj, qk, k0[i], f);

		// Project and compute cost function:
		LabordeProjectedDataSet pd(data, cyl);

		cost_result[i] = CostFunction(pnorm, k0_ap, sigma_k0)(pd);
	}

	return 0;
}
