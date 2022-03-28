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
using doomercat::CostFunctionHotineInf;
using doomercat::CostHotine;
using doomercat::hotine_result_t;
using doomercat::HotineObliqueMercator;



std::vector<hotine_result_t>
doomercat::bfgs_optimize_hotine(const DataSet& data, const double lonc0,
                         const double lat_00, const double alpha0,
                         const double k00, const double f,
                         const double pnorm, const double k0_ap,
                         const double sigma_k0,
                         const size_t Nmax, const bool proot,
                         const double epsilon)
{
	/* Number of parameters used in the optimization: */
	constexpr size_t P = 4;

	/* Defining the linear algebra implementation: */
	typedef linalg_t<P,double> lina_t;
	typedef HotineObliqueMercator<real4v> hom_t;


	/* Initial config: */
	CostFunctionHotine<real4v> cost_function(pnorm, k0_ap, sigma_k0, proot,
	                                         true, true);
	CostHotine<real4v> cost(cost_function(data,
	                             hom_t(variable4<0>(deg2rad(lonc0)),
	                                   variable4<1>(deg2rad(lat_00)),
	                                   variable4<2>(deg2rad(alpha0)),
	                                   variable4<3>(k00), f)));


	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	std::array<double,P> last_x;
	auto check_cached_cost = [&](const std::array<double,P>& x) {
		bool x_new = false;
		for (int i=0; i<P; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			const double lambda_c = std::fmod(x[0] + PI, 2*PI) - PI;
			const double alpha    = std::fmod(x[2] + PI/2, PI) - PI/2;
			cost = cost_function(data, hom_t(variable4<0>(lambda_c),
			                                 variable4<1>(x[1]),
			                                 variable4<2>(alpha),
			                                 variable4<3>(x[3]),
			                                 f));
			last_x = x;
		}
	};

	auto in_boundary = [&](const std::array<double,P>& x) -> bool {
		return (x[1] >= -0.5*PI && x[1] <= 0.5*PI
		        && x[3] > 0.0 && x[3] < 1.01);
	};

	std::cout.precision(16);

	auto propose_jump = [&](const std::array<double,P>& x,
	                        const std::array<double,P>& grad)
	      -> std::unique_ptr<std::array<double,P>>
	{
		if (   (x[1] > 89.0 * PI/180.0  && grad[1] < 0)
		    || (x[1] < -89.0 * PI/180.0 && grad[1] > 0))
		{
			double grad_scale = 2 * (90.0 - std::abs(x[1]))
			                    / std::abs(grad[1]);
			double lonc  = x[0] - grad_scale * grad[0] + PI;
			double lat0  = x[1] - grad_scale * grad[1];
			double alpha = x[2] - grad_scale * grad[2];
			double k0    = x[3] - grad_scale * grad[3];
			lonc = x[0] + PI;
			lat0 = x[1];
			alpha = x[2];
			std::unique_ptr<std::array<double,P>> res
			   = std::make_unique<std::array<double,P>>();
			(*res)[0] = lonc;
			(*res)[1] = lat0;
			(*res)[2] = alpha;
			(*res)[3] = x[3];
			return res;
		}
		return std::unique_ptr<std::array<double,P>>();
	};

	auto cost_lambda = [&](const std::array<double,P>& x) -> double {
		/* Check if we can used cached computation: */
		check_cached_cost(x);
		return cost;
	};

	auto gradient_lambda_redux = [&](const std::array<double,P>& x)
	   -> std::array<double,P>
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);

		/* Compute the gradient: */
		std::array<double,4> grad = cost.grad();

		/* Return the gradient: */
		return grad;
	};

	/* Initial config and optimization: */
	std::array<double,P> x0({deg2rad(lonc0),
	                         deg2rad(lat_00),
	                         deg2rad(alpha0),
	                         k00,
	                        });

	BFGS_result_t<lina_t> y, y2;
	y = fallback_gradient_BFGS<P,lina_t>(x0, cost_lambda,
	                                     gradient_lambda_redux,
	                                     Nmax, epsilon,
	                                     in_boundary,
	                                     propose_jump);
	// Poor man's cache invalidation.
	for (int i=0; i<P; ++i){
		last_x[i] *= 2;
	}
	x0 = y.history.back().parameters;

	/*
	 * Optimize non-logarithmic cost:
	 */
	constexpr bool LINEAR_OPTIMIZATION = false;
	if (LINEAR_OPTIMIZATION){
		cost_function = CostFunctionHotine<real4v>(pnorm, k0_ap, sigma_k0,
		                                           proot, false, true);

		y2 = fallback_gradient_BFGS<P,lina_t>(x0, cost_lambda,
		                                      gradient_lambda_redux,
		                                      Nmax-y.history.size(),
		                                      epsilon,
		                                      in_boundary,
		                                      propose_jump);
	}



	std::vector<hotine_result_t> res;
	res.reserve(y.history.size() + y2.history.size());
	auto append_history = [&](const BFGS_result_t<lina_t>& Y, bool cost_log) {
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
			const double lambda_c = std::fmod(z.parameters[0]+PI, 2*PI) - PI;

			/* Same for alpha: */
			const double alpha = std::fmod(z.parameters[2]+PI/2, PI) - PI/2;

			res.push_back({(cost_log) ? std::exp(z.cost) : z.cost,
				           rad2deg(lambda_c),
				           rad2deg(z.parameters[1]),
				           rad2deg(alpha),
				           z.parameters[3],
				           deg2rad(z.grad[0]),
				           deg2rad(z.grad[1]),
				           deg2rad(z.grad[2]),
				           z.grad[3],
				           state,
				           static_cast<unsigned int>(z.mode)});
		}
	};

	append_history(y, true);
	append_history(y2, false);

	return res;
}



std::vector<hotine_result_t>
doomercat::bfgs_optimize_hotine_pinf(const DataSet& data, const double lonc0,
                         const double lat_00, const double alpha0,
                         const double k00, const double f,
                         const double k0_ap, const double sigma_k0,
                         const size_t Nmax, const double epsilon)
{
	/* Number of parameters used in the optimization: */
	constexpr size_t P = 4;

	/* Defining the linear algebra implementation: */
	typedef linalg_t<P,double> lina_t;
	typedef HotineObliqueMercator<real4v> hom_t;


	/* Initial config: */
	CostFunctionHotineInf<real4v> cost_function(k0_ap, sigma_k0, true, true);
	CostHotine<real4v> cost(cost_function(data,
	                             hom_t(variable4<0>(deg2rad(lonc0)),
	                                   variable4<1>(deg2rad(lat_00)),
	                                   variable4<2>(deg2rad(alpha0)),
	                                   variable4<3>(k00), f)));


	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	std::array<double,P> last_x;
	auto check_cached_cost = [&](const std::array<double,P>& x) {
		bool x_new = false;
		for (int i=0; i<P; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			const double lambda_c = std::fmod(x[0] + PI, 2*PI) - PI;
			const double alpha    = std::fmod(x[2] + PI/2, PI) - PI/2;
			cost = cost_function(data, hom_t(variable4<0>(lambda_c),
			                                 variable4<1>(x[1]),
			                                 variable4<2>(alpha),
			                                 variable4<3>(x[3]),
			                                 f));
			last_x = x;
		}
	};

	auto in_boundary = [&](const std::array<double,P>& x) -> bool {
		return (x[1] >= -0.5*PI && x[1] <= 0.5*PI
		        && x[3] > 0.0 && x[3] < 1.01);
	};

	std::cout.precision(16);

	auto propose_jump = [&](const std::array<double,P>& x,
	                        const std::array<double,P>& grad)
	      -> std::unique_ptr<std::array<double,P>>
	{
		if (   (x[1] > 89.0 * PI/180.0  && grad[1] < 0)
		    || (x[1] < -89.0 * PI/180.0 && grad[1] > 0))
		{
			double grad_scale = 2 * (90.0 - std::abs(x[1]))
			                    / std::abs(grad[1]);
			double lonc  = x[0] - grad_scale * grad[0] + PI;
			double lat0  = x[1] - grad_scale * grad[1];
			double alpha = x[2] - grad_scale * grad[2];
			double k0    = x[3] - grad_scale * grad[3];
			lonc = x[0] + PI;
			lat0 = x[1];
			alpha = x[2];
			std::unique_ptr<std::array<double,P>> res
			   = std::make_unique<std::array<double,P>>();
			(*res)[0] = lonc;
			(*res)[1] = lat0;
			(*res)[2] = alpha;
			(*res)[3] = x[3];
			return res;
		}
		return std::unique_ptr<std::array<double,P>>();
	};

	auto cost_lambda = [&](const std::array<double,P>& x) -> double {
		/* Check if we can used cached computation: */
		check_cached_cost(x);
		return cost;
	};

	auto gradient_lambda_redux = [&](const std::array<double,P>& x)
	   -> std::array<double,P>
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);

		/* Compute the gradient: */
		std::array<double,4> grad = cost.grad();

		/* Return the gradient: */
		return grad;
	};

	/* Initial config and optimization: */
	std::array<double,P> x0({deg2rad(lonc0),
	                         deg2rad(lat_00),
	                         deg2rad(alpha0),
	                         k00,
	                        });

	BFGS_result_t<lina_t> y, y2;
	y = fallback_gradient_BFGS<P,lina_t>(x0, cost_lambda,
	                                     gradient_lambda_redux,
	                                     Nmax, epsilon,
	                                     in_boundary,
	                                     propose_jump);
	// Poor man's cache invalidation.
	for (int i=0; i<P; ++i){
		last_x[i] *= 2;
	}
	x0 = y.history.back().parameters;

	/*
	 * Optimize non-logarithmic cost:
	 */
	cost_function = CostFunctionHotineInf<real4v>(k0_ap, sigma_k0, false, true);

	y2 = fallback_gradient_BFGS<P,lina_t>(x0, cost_lambda,
	                                      gradient_lambda_redux,
	                                      Nmax-y.history.size(),
	                                      epsilon,
	                                      in_boundary,
	                                      propose_jump);



	std::vector<hotine_result_t> res;
	res.reserve(y.history.size() + y2.history.size());
	auto append_history = [&](const BFGS_result_t<lina_t>& Y, bool cost_log) {
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
			const double lambda_c = std::fmod(z.parameters[0]+PI, 2*PI) - PI;

			/* Same for alpha: */
			const double alpha = std::fmod(z.parameters[2]+PI/2, PI) - PI/2;

			res.push_back({(cost_log) ? std::exp(z.cost) : z.cost,
				           rad2deg(lambda_c),
				           rad2deg(z.parameters[1]),
				           rad2deg(alpha),
				           z.parameters[3],
				           deg2rad(z.grad[0]),
				           deg2rad(z.grad[1]),
				           deg2rad(z.grad[2]),
				           z.grad[3],
				           state,
				           static_cast<unsigned int>(z.mode)});
		}
	};

	append_history(y, true);
	append_history(y2, false);

	return res;
}
