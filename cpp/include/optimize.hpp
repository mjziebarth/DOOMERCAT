/*
 * Optimization routines.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
 *               2024 Technische Universität München
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

#include <memory>
#include <variant>
#include <optional>
#include <../include/cost_hotine.hpp>
#include <../include/hotine.hpp>
#include <../include/linalg.hpp>
#include <../include/bfgs2.hpp>
#include <../include/truong2020.hpp>
#include <../include/parameters.hpp>

#ifndef DOOMERCAT_OPTIMIZE_H
#define DOOMERCAT_OPTIMIZE_H

namespace doomercat {

enum class return_state_t {
	CONVERGED=0,
	MAX_ITERATIONS=1,
	ERROR=2
};

struct hotine_result_t {
	double cost;
	double lonc;
	double lat_0;
	double alpha;
	double k0;
	double grad_lonc;
	double grad_lat0;
	double grad_alpha;
	double grad_k0;
	return_state_t state;
	unsigned int algorithm_state;
	double step;
};


/*
 * This wrapper class provides the 'real_t' typedef and the 'ndim' constant.
 * Besides that, it ensures that all assignments from std::array can be
 * performed.
 */
template<typename real, size_t _ndim>
struct ArrayWrapper : public linalg_t<_ndim, real>::point_t
{
public:
	typedef real real_t;
	constexpr static size_t ndim = _ndim;

	typedef typename linalg_t<_ndim, real>::point_t base_t;

	static_assert(std::is_same<base_t, std::array<real,_ndim>>::value,
	              "linalg_t::point_t must be a std::array.");

	ArrayWrapper(const real& lonc, const real& lat_0, const real& alpha,
	             const real& k0)
	{
		base_t::operator[](0) = lonc;
		base_t::operator[](1) = lat_0;
		base_t::operator[](2) = alpha;
		base_t::operator[](3) = k0;
	}

	ArrayWrapper(const base_t& other)
	   : base_t(other)
	{}

	ArrayWrapper& operator=(const base_t& other)
	{
		base_t::operator=(other);
		return *this;
	}
};

/*
 * BFGS v2: dampened update.
 */
template<typename real_t = double, typename DS = void>
std::vector<hotine_result_t>
damped_BFGS_optimize_hotine(
    const DS& data,
    const double lonc0,
    const double lat_00,
    const double alpha0,
    const double k00,
    const double f,
    const double pnorm,
    const double k0_ap,
    const double sigma_k0,
    const size_t Nmax,
    const double epsilon,
    std::optional<size_t>& function_evaluations
)
{

	/* Number of parameters used in the optimization: */
	constexpr size_t P = 4;
	static_assert(P < 256, "If P >= 256, the iteration types need to be "
	                       "increased.");
	typedef uint_fast8_t p_iter_t;

	/* Defining the linear algebra implementation: */
	typedef linalg_t<P, real_t> lina_t;
	typedef autodouble<4, real_t> real4v;
	typedef HotineObliqueMercator<real4v> hom_t;

	typedef ArrayWrapper<real_t, P> point_t;

	/*
	 * The cost function: a variant for finite and infinite
	 * pnorm.
	 */
	typedef std::variant<CostFunctionHotine<real4v>,
	                     CostFunctionHotineInf<real4v>>
	    costfun_t;
	auto generate_cost_function = [=]() -> costfun_t
	{
		if (std::isinf(pnorm)){
		    return CostFunctionHotineInf<real4v>(
		            k0_ap, sigma_k0, true, true
		    );
		} else {
		    return CostFunctionHotine<real4v>(
		            pnorm, k0_ap, sigma_k0, true, true
		    );
		}
	};
	costfun_t cost_function = generate_cost_function();

	/* Initial cost: */
	CostHotine<real4v> cost(
		std::visit(
			[lonc0, lat_00, alpha0, k00, f, &data]
			(auto&& cfun) -> CostHotine<real4v>
			{
				HotineParameters<real4v> params(
				    variable4<0>(deg2rad(static_cast<real_t>(lonc0))),
				    variable4<1>(deg2rad(static_cast<real_t>(lat_00))),
				    variable4<2>(deg2rad(static_cast<real_t>(alpha0))),
				    variable4<3>(static_cast<real_t>(k00))
				);
				return cfun(data, hom_t(params, f));
			},
			cost_function
		)
	);

	size_t fun_evals = 0;


	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	point_t last_x = {0.0, 0.0, 0.0, -1.0};
	auto check_cached_cost
	    = [&last_x, &cost_function, &cost, &data, &fun_evals, f]
	    (const point_t& x)
	{
		bool x_new = false;
		for (p_iter_t i=0; i<P; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			cost = std::visit(
				[x,f,&data](auto&& cfun) -> CostHotine<real4v>
				{
					HotineParameters<real4v> params(
					    variable4<0>(x[0]),
					    variable4<1>(x[1]),
					    variable4<2>(x[2]),
					    variable4<3>(x[3])
					);
					return cfun(data, hom_t(params, f));
				},
				cost_function
			);
			++fun_evals;
			last_x = x;
		}
	};

	auto in_boundary
		= [&](const point_t& x) -> bool
	{
		return x[3] > 0.0 && x[3] < 1.01;
	};

	auto cost_lambda
		= [&](const point_t& x) -> double
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);
		return cost;
	};

	auto gradient_lambda_redux
		= [&](const point_t& x)
	      -> std::array<real_t,P>
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);

		/* Compute the gradient: */
		std::array<real_t,4> grad = cost.grad();

		/* Return the gradient: */
		return grad;
	};

	/* Initial config and optimization: */
	point_t x0(
	    deg2rad(lonc0),
	    deg2rad(lat_00),
	    deg2rad(alpha0),
	    k00
	);

	/* Parameters: */
	constexpr bool last_redux = false;

	BFGS_result_t<lina_t> y, y2;
	y = dampened_BFGS<last_redux, real_t>(
	            x0,
	            cost_lambda,
	            gradient_lambda_redux,
	            Nmax,
	            epsilon,
	            in_boundary
	);


	// Poor man's cache invalidation.
	{
		point_t last_x_ = last_x;
		for (p_iter_t i=0; i<P; ++i){
			last_x_[i] += 1;
			last_x_[i] *= 2;
		}
		last_x = last_x_;
	}
	x0 = y.history.back().parameters;

	/*
	 * Optimize non-logarithmic cost:
	 */
	if (std::isinf(pnorm)){
		cost_function
			.template emplace<CostFunctionHotineInf<real4v>>(
				k0_ap, sigma_k0, false, true
		);
	} else {
		cost_function
			.template emplace<CostFunctionHotine<real4v>>(
				pnorm, k0_ap, sigma_k0, false, true
		);
	}


	y2 = dampened_BFGS<last_redux, real_t>(
	            x0,
	            cost_lambda,
	            gradient_lambda_redux,
	            Nmax-y.history.size(),
	            epsilon,
	            in_boundary
	);

	/*
	 * Copy the history:
	 */
	std::vector<hotine_result_t> res;
	res.reserve(y.history.size() + y2.history.size());
	auto append_history
	= [&](const BFGS_result_t<lina_t>& Y, bool cost_log)
	{
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

			HotineParameters<real_t> params(z.parameters);
			res.push_back({(cost_log) ? static_cast<double>(std::exp(z.cost))
			                          : static_cast<double>(z.cost),
			               static_cast<double>(rad2deg(params[0])),
			               static_cast<double>(rad2deg(params[1])),
			               static_cast<double>(rad2deg(params[2])),
			               static_cast<double>(params[3]),
			               static_cast<double>(deg2rad(z.grad[0])),
			               static_cast<double>(deg2rad(z.grad[1])),
			               static_cast<double>(deg2rad(z.grad[2])),
			               static_cast<double>(z.grad[3]),
			               state,
			               static_cast<unsigned int>(-1),
			               z.step});
		}
	};

	append_history(y, true);
	append_history(y2, false);


	/* If requested, list the total number of function evaluations: */
	if (function_evaluations)
		*function_evaluations = fun_evals;

	return res;
}


/*
 * Backtracking gradient descent.
 */
template<typename DS>
std::vector<hotine_result_t>
backtrack_GD_optimize_hotine(
    const DS& data, const double lonc0,
    const double lat_00, const double alpha0,
    const double k00, const double f,
    const double pnorm,
    const double k0_ap, const double sigma_k0,
    const size_t Nmax, const double epsilon,
    std::optional<size_t>& function_evaluations
)
{
	/* Number of parameters used in the optimization: */
	constexpr size_t P = 4;
	static_assert(P < 256, "If P >= 256, the iteration types need to be "
	                       "increased.");
	typedef uint_fast8_t p_iter_t;

	/* Defining the linear algebra implementation: */
	typedef double real_t;
	typedef linalg_t<P, real_t> lina_t;
	typedef autodouble<4, real_t> real4v;
	typedef HotineObliqueMercator<real4v> hom_t;


	/*
	 * The cost function: a variant for finite and infinite
	 * pnorm.
	 */
	typedef std::variant<CostFunctionHotine<real4v>,
	                     CostFunctionHotineInf<real4v>>
	    costfun_t;
	auto generate_cost_function = [=]() -> costfun_t
	{
		if (std::isinf(pnorm)){
		    return CostFunctionHotineInf<real4v>(
		            k0_ap, sigma_k0, true, true
		    );
		} else {
		    return CostFunctionHotine<real4v>(
		            pnorm, k0_ap, sigma_k0, true, true
		    );
		}
	};
	costfun_t cost_function = generate_cost_function();

	/* Initial cost: */
	CostHotine<real4v> cost(
		std::visit(
			[lonc0, lat_00, alpha0, k00, f, &data]
			(auto&& cfun) -> CostHotine<real4v>
			{
			return cfun(data,
			         hom_t(variable4<0>(deg2rad(lonc0)),
			               variable4<1>(deg2rad(lat_00)),
			               variable4<2>(deg2rad(alpha0)),
			               variable4<3>(k00), f)
				);
			},
			cost_function
		)
	);


	size_t fun_evals = 0;

	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	HotineParameters<real_t> last_x = HotineParameters<real_t>::invalid();
	auto check_cached_cost
	    = [&last_x, &cost_function, &cost, &data, &fun_evals, f]
	    (const HotineParameters<real_t>& x)
	{
		bool x_new = false;
		for (p_iter_t i=0; i<P; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			cost = std::visit(
				[x,f,&data,&fun_evals](auto&& cfun) -> CostHotine<real4v>
				{
					return cfun(data,
					    hom_t(variable4<0>(x[0]),
					          variable4<1>(x[1]),
					          variable4<2>(x[2]),
					          variable4<3>(x[3]),
					          f)
					);
				},
				cost_function
			);
			++fun_evals;
			last_x = x;
		}
	};

	auto in_boundary
		= [&](const HotineParameters<real_t>& x) -> bool
	{
		return x[3] > 0.0 && x[3] < 1.01 && !std::isnan(x[0])
		    && !std::isnan(x[1]) && !std::isnan(x[2])
		    && !std::isinf(x[0]) && !std::isinf(x[1]) && !std::isinf(x[2]);
	};

	auto cost_lambda
		= [&](const HotineParameters<real_t>& x) -> double
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);
		return cost;
	};

	auto gradient_lambda_redux
		= [&](const HotineParameters<real_t>& x)
	      -> std::array<real_t,P>
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);

		/* Compute the gradient: */
		std::array<real_t,4> grad = cost.grad();

		/* Return the gradient: */
		return grad;
	};

	/* Initial config and optimization: */
	HotineParameters<real_t>
	    x0(deg2rad(lonc0), deg2rad(lat_00),
	       deg2rad(alpha0), k00);

	/* Parameters: */
	constexpr double delta0 = 10.0;
	constexpr double alpha = 0.02;
	constexpr double beta = 0.5;
	constexpr size_t Nmax_linesearch = 65;

	GD_result_t<lina_t> y, y2;
	y = two_way_backtracking_gradient_descent(
	        x0, cost_lambda,
	        gradient_lambda_redux,
	        delta0, alpha, beta,
	        Nmax, Nmax_linesearch, epsilon,
	        in_boundary
	);

	// Poor man's cache invalidation.
	{
		std::array<real_t, P> last_x_ = last_x;
		for (p_iter_t i=0; i<P; ++i){
			last_x_[i] += 1;
			last_x_[i] *= 2;
		}
		last_x = HotineParameters<real_t>(last_x_);
	}
	x0 = y.history.back().parameters;

	/*
	 * Optimize non-logarithmic cost:
	 */
	if (std::isinf(pnorm)){
		cost_function
			.emplace<CostFunctionHotineInf<real4v>>(
				k0_ap, sigma_k0, false, true
		);
	} else {
		cost_function
			.emplace<CostFunctionHotine<real4v>>(
				pnorm, k0_ap, sigma_k0, false, true
		);
	}

	y2 = two_way_backtracking_gradient_descent(
			x0, cost_lambda,
	        gradient_lambda_redux,
			delta0, alpha, beta,
	        Nmax-y.history.size(),
			Nmax_linesearch, epsilon,
	        in_boundary
	);

	/*
	 * Copy the history:
	 */
	std::vector<hotine_result_t> res;
	res.reserve(y.history.size() + y2.history.size());
	auto append_history
	= [&](const GD_result_t<lina_t>& Y, bool cost_log)
	{
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

			HotineParameters<real_t> params(z.parameters);
			res.push_back({(cost_log) ? std::exp(z.cost) : z.cost,
				           rad2deg(params[0]),
				           rad2deg(params[1]),
				           rad2deg(params[2]),
				           params[3],
				           deg2rad(z.grad[0]),
				           deg2rad(z.grad[1]),
				           deg2rad(z.grad[2]),
				           z.grad[3],
				           state,
				           static_cast<unsigned int>(-1),
				           z.step});
		}
	};

	append_history(y, true);
	append_history(y2, false);

	/* If requested, list the total number of function evaluations: */
	if (function_evaluations)
		*function_evaluations = fun_evals;

	return res;
}



}
#endif
