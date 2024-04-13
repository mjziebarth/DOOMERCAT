/*
 * Implementation of the inexact backtracking gradient descent method.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2024 Technische Universität München,
 *               2022 Deutsches GeoForschungsZentrum Potsdam
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

#include <../include/linalg.hpp>
#include <../include/exitcode.hpp>

#ifndef DOOMERCAT_TRUONG2020_H
#define DOOMERCAT_TRUONG2020_H


/*
 *
 * Return types and history bookkeeping:
 *
 */

template<typename lina>
struct GD_step_t {
	double cost;
	typename lina::point_t parameters;
	typename lina::grad_t grad;
	double step;

	GD_step_t(double cost, const typename lina::point_t& parameters,
	          double step)
	   : cost(cost), parameters(parameters), step(step)
	{
		for (size_t i=0; i<grad.size(); ++i){
			grad[i] = 0.0;
		}
	};

	GD_step_t(
        double cost,
	    const typename lina::point_t& parameters,
	    const typename lina::grad_t& gradient,
	    double step
    ) : cost(cost), parameters(parameters),
	    grad(gradient), step(step)
	{};

	GD_step_t(GD_step_t&& other) = default;
	GD_step_t(const GD_step_t& other) = default;
};

template<typename lina>
struct GD_result_t {
	exit_code_t exit_code;
	std::vector<GD_step_t<lina>> history;
};

/*
 * Implements the two-way backtracking line search by Truong & Nguyen (2020).
 */

template<typename parameters_t,
         typename costfun_t, typename gradfun_t,
         typename boundsfun_t>
GD_result_t<
    linalg_t<
        parameters_t::ndim,
        typename parameters_t::real_t
	>
>
two_way_backtracking_gradient_descent(
    const parameters_t& x0,
    costfun_t costfun, gradfun_t gradient,
    const double delta0,
    const double alpha, const double beta,
    const size_t Nmax, const size_t Nmax_linesearch,
	const double epsilon, boundsfun_t in_bounds
)
{
    /* Typedefs for brevity: */
	constexpr size_t d = parameters_t::ndim;
	typedef typename parameters_t::real_t real_t;
	typedef linalg_t<d, real_t> lina;
	typedef typename lina::column_vectord_t vd_t;
	typedef typename lina::point_t point_t;
	typedef typename lina::grad_t grad_t;

	/*
	 * This function implements the algorithm described in
	 * section 2.1 of Truong & Nguyen (2021) with the
	 * two-way backtracking modification of section ???
	 *
	 * We use the gradient estimate, vk = grad(cost), hence
	 * we can set A1 = A2 = mu = 1 (Truong & Nguyen, 2021) and
	 * disregard these
	 */


	/* Init the starting point from the value given: */
	vd_t xk;
	lina::init_column_vectord(xk, x0);

	/* The trajectory in optimization space; and early exit: */
	GD_result_t<lina> result;
	result.exit_code = MAX_ITERATIONS;
	if (Nmax == 0)
		return result;

	if (Nmax_linesearch == 0)
		throw std::runtime_error("No line search possible.");

	/* The loop: */
	point_t P(x0);
	vd_t grad_fk;
	if (!in_bounds(x0))
		throw std::runtime_error("Start point out of bounds.");
	real_t cost0 = costfun(x0);
	double last_delta = delta0;
	vd_t xkp1;
	for (size_t i=0; i<Nmax-1; ++i){
		/* At the start of this loop, the current position is
		 * stored in the 'row_vectord_t' xk.
		 *
		 * Transfer to all relevant types and
		 * compute the gradient for this step: */
		lina::fill_array(P, xk);
		grad_t G(gradient(P));
		lina::init_column_vectord(grad_fk, G);

		/* Check the gradient for nan: */
		for (size_t j=0; j<d; ++j){
			if (std::isnan(grad_fk[j])){
				#ifdef DEBUG
					std::cout << "EXITING BECAUSE grad_fk IS NAN!\n"
					          << std::flush;
				#endif
				result.exit_code = COST_DIVERGENCE;
				break;
			}
		}

		/* History for debugging purposes: */
		result.history.emplace_back(
			cost0, P, G, last_delta
		);

		/* Line search: */
		real_t cost = cost0;
		bool armijo = false;
		double sigma = last_delta;
		const real_t grad_nrm_2 = lina::dot(grad_fk, grad_fk);
		for (size_t j=0; j<Nmax_linesearch; ++j)
		{
			/* Propose a new point: */
			xkp1 = xk - sigma * grad_fk;

			/* Sanity: */
			if (xkp1 == xk)
				break;

			/* Compute the cost: */
			point_t Pp1;
			lina::fill_array(Pp1, xkp1);
			if (in_bounds(Pp1)){
				cost = costfun(Pp1);
			} else {
				cost = std::numeric_limits<real_t>::infinity();
			}

			/* Check the Armijo condition: */
			armijo = cost - cost0 < - alpha * sigma * grad_nrm_2;
			if (armijo){
				/* Here goes the two-way backtracking modification
				 * from section 3.3 of Truong & Nguyen (2021).
				 * Note: It is not clear from the paper's PDF
				 * how exactly point 3.2.1 is to be interpreted.
				 * The way it is currently worded implies that
				 * delta is always to be reset to delta0 * beta
				 * (by iterative multiplication...). This does not
				 * seem correct. Maybe the Armijo-condition should
				 * be checked in each iteration that increases
				 * sigma? No clue. Here, we let sigma increase
				 * one beta at a time. */
				if (j == 0 && sigma <= beta * delta0){
					sigma /= beta;
				}
				break;
			}

			/* Next in the reducing series: */
			sigma *= beta;
		}
		if (!armijo){
			result.exit_code = LINESEARCH_FAIL;
			return result;
		}

		/* Besides the new point xkp1, this defines delta_n: */
		last_delta = sigma;

		/* Accept the new point: */
		xk = xkp1;
		cost0 = cost;

		/* Exit criterion: */
		// TODO
	}

	/* Save final value to the history: */
	lina::fill_array(P, xk);
	grad_t G(gradient(P));
	lina::init_column_vectord(grad_fk, G);

	/* History for debugging purposes: */
	result.history.emplace_back(
		cost0, P, G, last_delta
	);

	return result;
}


#endif