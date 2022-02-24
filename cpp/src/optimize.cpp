/*
 * Optimization routines.
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
#include <../include/bfgs.hpp>

#include <vector>
#include <limits>
#include <cmath>


// Bugfix code:
#include <chrono>
#include <thread>
#include <iostream>

using doomercat::DataSet;
using doomercat::return_state_t;
using doomercat::CostFunctionHotine;
using doomercat::CostHotine;
using doomercat::hotine_result_t;
using doomercat::HotineObliqueMercator;


constexpr double EPSILON = 1e-8;

std::vector<hotine_result_t>
doomercat::bfgs_optimize_hotine(const DataSet& data, const double lonc0,
                         const double lat_00, const double alpha0,
                         const double k00, const double f,
                         const unsigned int pnorm, const double k0_ap,
                         const double sigma_k0,
                         const size_t Nmax, bool transform_k0)
{
	/* Defining the linear algebra implementation: */
	typedef linalg_t<5,double> lina_t;
	typedef HotineObliqueMercator<real4v> hom_t;

	std::cout << "init lambdas.\n" << std::flush;

	/* Variable transformation in k0: */
	auto z_to_k0 = (transform_k0) ?
	[](double z) -> double {
		return 1.0 / (1.0 + std::exp(z));
	} :
	[](double z) -> double {
		return z;
	}
	;

	auto d_k0_d_z = (transform_k0) ?
	[](double z) -> double {
		const double ez = std::exp(z);
		const double k0 = 1.0 / (1.0 + ez);
		return - k0 * k0 * ez;
	} :
	[](double z) -> double {
		return 1.0;
	};

	auto k0_to_z = (transform_k0) ?
	[](double k0) -> double {
		return std::log(1.0 / k0 - 1.0);
	} :
	[](double k0) -> double {
		return k0;
	};


	/*
	 * Variable transformation in lonc:
	 */
	auto xy_to_lambda_c = [](double x, double y) -> double {
		return std::atan2(y,x);
	};
	auto d_lambda_c_d_x = [](double x, double y) -> double {
		return -y / (x*x + y*y);
	};
	auto d_lambda_c_d_y = [](double x, double y) -> double {
		return x / (x*x + y*y);
	};
	auto lambda_c_to_xy = [](double lambda_c) -> std::array<double,2>
	{
		return {std::cos(lambda_c), std::sin(lambda_c)};
	};


	/* Initial config: */
	CostFunctionHotine cost_function(pnorm, k0_ap, sigma_k0, true);
	CostHotine cost(cost_function(data, hom_t(variable4<0>(deg2rad(lonc0)),
	                                          variable4<1>(deg2rad(lat_00)),
	                                          variable4<2>(deg2rad(alpha0)),
	                                          variable4<3>(k00), f)));

	std::cout << "cached cost\n" << std::flush;


	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	std::array<double,5> last_x;
	auto check_cached_cost = [&](const std::array<double,5>& x) {
		bool x_new = false;
		for (int i=0; i<4; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			const double lambda_c = xy_to_lambda_c(x[0],x[1]);
			cost = cost_function(data, hom_t(variable4<0>(lambda_c),
			                                 variable4<1>(x[2]),
 			                                 variable4<2>(x[3]),
 			                                 variable4<3>(z_to_k0(x[4])),
 			                                 f));
			last_x = x;
		}
	};

	auto in_boundary = [&](const std::array<double,5>& x) -> bool {
		return (x[2] >= -0.5*PI && x[2] <= 0.5*PI && x[3] >= -0.5*PI &&
			    x[3] <= 0.5*PI && z_to_k0(x[4]) > 0.0 &&
			    z_to_k0(x[4]) < 1.01);
	};

	auto cost_lambda = [&](const std::array<double,5>& x) -> double {
		/* Check if we can used cached computation: */
		check_cached_cost(x);
		return cost;
	};

	auto gradient_lambda_redux = [&](const std::array<double,5>& x)
	   -> std::array<double,5>
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);

		/* Compute the gradient and apply chain rule for k0 as
		 * well as x&y: */
		std::array<double,4> grad = cost.grad();
		std::array<double,5> res;
		res[0] = grad[0] * d_lambda_c_d_x(x[0],x[1]);
		res[1] = grad[0] * d_lambda_c_d_y(x[0],x[1]);
		res[2] = grad[1];
		res[3] = grad[2];
		res[4] = grad[3] * d_k0_d_z(x[4]);

		/* Return the gradient: */
		return res;
	};

	std::cout << "create x0\n" << std::flush;

	/* Initial config and optimization: */
	std::array<double,5> x0({std::cos(deg2rad(lonc0)),
	                         std::sin(deg2rad(lonc0)),
	                         deg2rad(lat_00),
	                         deg2rad(alpha0),
	                         std::max(k0_to_z(k00), -15.0)});

	std::cout << "BFGS\n" << std::flush;
	BFGS_result_t<lina_t> y
	   = fallback_gradient_BFGS<5,lina_t>(x0, cost_lambda,
	                                      gradient_lambda_redux,
	                                      Nmax, EPSILON,
	                                      in_boundary);

	std::cout << "done 1!\n" << std::flush;

	/*
	 * Optimize non-logarithmic cost:
	 */
	cost_function = CostFunctionHotine(pnorm, k0_ap, sigma_k0, false);
	// Poor man's cache invalidation.
	for (int i=0; i<5; ++i){
		last_x[i] *= 2;
	}
	x0 = y.history.back().parameters;
	check_cached_cost(x0);
	BFGS_result_t<lina_t> y2
	   = fallback_gradient_BFGS<5,lina_t>(x0, cost_lambda,
	                                      gradient_lambda_redux,
	                                      Nmax-y.history.size(),
	                                      EPSILON,
	                                      in_boundary);



	std::cout << "done 2!\n" << std::flush;



	std::vector<hotine_result_t> res;
	res.reserve(y.history.size() + y2.history.size());
	auto append_history = [&](const BFGS_result_t<lina_t>& Y) {
		for (auto z : Y.history){
			return_state_t state = return_state_t::ERROR;
			switch (y.exit_code){
				case CONVERGED:
					state = return_state_t::CONVERGED;
					break;
				case MAX_ITERATIONS:
					state = return_state_t::MAX_ITERATIONS;
					break;
				case RHO_DIVERGENCE:
				case LINESEARCH_FAIL:
				case COST_DIVERGENCE:
					state = return_state_t::ERROR;
			}

			const double lambda_c = xy_to_lambda_c(z.parameters[0],
				                                   z.parameters[1]);
			/* We want to save the gradient in terms of lambda_c
			 * but we have it expressed in terms of x and y:
			 *   res[0] = grad[0] * d_lambda_c_d_x(x[0],x[1]);
			 *   res[1] = grad[0] * d_lambda_c_d_y(x[0],x[1]);
			 * Compute the chain rule for both and take the one with the
			 * larger magnitude to get a more stable estimate of grad[0]: */
			const double dlcdx = d_lambda_c_d_x(z.parameters[0],
			                                    z.parameters[1]);
			const double dlcdy = d_lambda_c_d_y(z.parameters[0],
			                                    z.parameters[1]);
			double grad0;
			if (std::abs(dlcdx) > std::abs(dlcdy)){
				grad0 = z.grad[0] / dlcdx;
			} else {
				grad0 = z.grad[1] / dlcdy;
			}

			res.push_back({z.cost,
				           rad2deg(lambda_c),
				           rad2deg(z.parameters[2]),
				           rad2deg(z.parameters[3]),
				           z_to_k0(z.parameters[4]),
				           deg2rad(grad0),
				           deg2rad(z.grad[2]),
				           deg2rad(z.grad[3]),
				           z.grad[4] / d_k0_d_z(z.parameters[4]),
				           state,
				           static_cast<unsigned int>(z.mode)});
		}
	};

	append_history(y);
	append_history(y2);

	std::cout << "returning result.\n" << std::flush;

	return res;
}
