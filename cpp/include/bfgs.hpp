/*
 * Implementation of the BFGS method.
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

#include <array>
#include <cstddef>
#include <cmath>
#include <vector>
#include <functional>
#include <tuple>
#include <limits>
#include <memory>

// Debug:
#ifdef DEBUG
#include <iostream>
#endif

#include <../include/linalg.hpp>

#ifndef DOOMERCAT_BFGS_H
#define DOOMERCAT_BFGS_H

enum BFGS_exit_code_t {
	MAX_ITERATIONS, CONVERGED, RHO_DIVERGENCE, LINESEARCH_FAIL, COST_DIVERGENCE
};

enum class BFGS_mode_t {
	BFGS=1, FALLBACK_GRADIENT=2, FINISHED=0
};

template<typename lina>
struct BFGS_step_t {
	double cost;
	BFGS_mode_t mode;
	typename lina::point_t parameters;
	typename lina::grad_t grad;
	double step;

	BFGS_step_t(double cost, BFGS_mode_t mode,
	            const typename lina::point_t& parameters,
	            double step)
	   : cost(cost), mode(mode), parameters(parameters), step(step)
	{
		for (int i=0; i<grad.size(); ++i){
			grad[i] = 0.0;
		}
	};

	BFGS_step_t(double cost, BFGS_mode_t mode,
	            const typename lina::point_t& parameters,
	            const typename lina::grad_t& gradient,
	            double step)
	   : cost(cost), mode(mode), parameters(parameters),
	     grad(gradient), step(step)
	{};

	BFGS_step_t(BFGS_step_t&& other) = default;
	BFGS_step_t(const BFGS_step_t& other) = default;
};

template<typename lina>
struct BFGS_result_t {
	BFGS_exit_code_t exit_code;
	std::vector<BFGS_step_t<lina>> history;
};


template<size_t d, typename lina>
BFGS_result_t<lina>
    BFGS(const typename lina::point_t& x0,
         std::function<typename lina::real_t (const typename lina::point_t&)>
              cost,
         std::function<typename lina::grad_t (const typename lina::point_t&)>
              gradient,
         const size_t Nmax, const double epsilon
    )
{
	typedef typename lina::point_t point_t;
	typedef typename lina::matrix_dxd_t Mdxd_t;
	typedef typename lina::column_vectord_t vd_t;

	/* Follows Nocedal & Wright: Numerical Optimization */
	Mdxd_t Hk(1e-3 * lina::identity());
	Hk(d-1,d-1) *= 1e-5;
	const Mdxd_t I(lina::identity());

	/* Configuration of the Wolfe condition: */
	constexpr double c1 = 1e-4;
	constexpr double c2 = 0.9;

	/* Init the starting point from the value given: */
	vd_t xk;
	lina::init_column_vectord(xk, x0);

	/* The trajectory in optimization space; and early exit: */
	BFGS_result_t<lina> result;
	result.exit_code = MAX_ITERATIONS;
	if (Nmax == 0)
		return result;

	auto convert_gradient = [](const vd_t& grad) -> typename lina::grad_t {
		typename lina::grad_t dest;
		lina::fill_array(dest, grad);
		return dest;
	};

	/* The loop: */
	point_t P(x0);
	vd_t grad_fk;
	lina::init_column_vectord(grad_fk, gradient(x0));
	double cost0 = cost(x0);
	size_t Hk_age = 0;
	for (size_t i=0; i<Nmax-1; ++i){
		// From xk, compute the point:
		lina::fill_array(P, xk);

		// Keep track:
		result.history.emplace_back(cost0, BFGS_mode_t::BFGS, P,
		                            convert_gradient(grad_fk), 0.0);

		#ifdef DEBUG
			std::cout << "gradient[" << i << "] = (";
			for (size_t j=0; j<d; ++j)
				std::cout << grad_fk[j] << ",";
			std::cout << ")\n";
		#endif

		// Check finiteness of cost function:
		if (std::isnan(cost0) || std::isinf(cost0)){
			/* This might happen due to bad numerical problems. Exit
			 * before this influences anything else. */
			result.exit_code = COST_DIVERGENCE;
			break;
		}


		// Compute the search direction:
		vd_t pk = (Hk_age > 3) ? - (Hk * grad_fk) : -1.0 * grad_fk;

		// Compute the new point and gradient (Wolfe search, (3.6a & 3.6b).
		// First, decide how deep we can reduce alpha until we do not change
		// xk any more:
		constexpr double ALPHA_REDUX = 0.5;
		const double alpha_min = 1e-10 * lina::norm(xk) / lina::norm(pk);
		const size_t jmax_down
		   = std::max(
		        std::min(static_cast<size_t>(
		                    std::floor(std::log(alpha_min)
		                                      / std::log(ALPHA_REDUX))),
		                 static_cast<size_t>(100)),
		        static_cast<size_t>(10));
		double alpha = 1.0;
		vd_t xkp1(xk + alpha * pk);
		point_t Pp1;
		lina::fill_array(Pp1, xkp1);
		vd_t grad_fkp1;
		lina::init_column_vectord(grad_fkp1, gradient(Pp1));
		double cost1 = cost(Pp1);

		const double grad_fk_dot_pk = lina::dot(grad_fk, pk);
		bool wolfe_success = false;
		bool wolfe_1 = false;
		bool wolfe_2 = false;
		for (size_t j=0; j<jmax_down; ++j)
		{
			wolfe_1 = cost1 <= cost0 + c1 * alpha * grad_fk_dot_pk; // Wolfe 1, 3.6a
			wolfe_2 = lina::dot(grad_fkp1,pk) >= c2 * grad_fk_dot_pk; // Wolfe 2, 3.6b
			if (wolfe_1 && (wolfe_2 || Hk_age <= 3))
			{
				wolfe_success = true;
				break;
			}
			/* Reduce alpha and compute the new point: */
			alpha *= ALPHA_REDUX;
			xkp1 = xk + alpha * pk;
			lina::fill_array(Pp1, xkp1);
			lina::init_column_vectord(grad_fkp1, gradient(Pp1));
			cost1 = cost(Pp1);
		}

		if (!wolfe_success){
			if (wolfe_1){
				/* The sufficient reduction was not fulfilled. But maybe
				 * the Hessian changed a lot, so we might gain something
				 * from proceeding for another small step: */
				result.exit_code = MAX_ITERATIONS;
			} else {
				/* No sufficient reduction possible. */
				result.exit_code = LINESEARCH_FAIL;

				/* Switch back to gradient descent: */
				Hk = lina::identity();
				Hk_age = 0;
				continue;
			}
			#ifdef DEBUG
			std::cout << "Wolfe failed. W1=" << wolfe_1 << ", W2="
				      << wolfe_2 << "\n";
			#endif
		} else {
			result.exit_code = MAX_ITERATIONS;
		}


		#ifdef DEBUG
			/* Print some status information about the curren parameters: */
			std::cout << "   -> grad_fk_dot_pk = " << grad_fk_dot_pk << "\n";
			std::cout << "   -> param = (";
			for (size_t j=0; j<d; ++j)
				std::cout << xkp1[j] << ",";
			std::cout << ")\n";
			std::cout << "   -> alpha =  " << alpha << "\n";
			std::cout << "   -> Hk = (";
			for (size_t j=0; j<d; ++j){
				for (size_t k=0; k<d; ++k){
					std::cout << Hk(j,k) << ",\t";
				}
				std::cout << "\n            ";
			}
			std::cout << ")\n";
		#endif

		// Compute sk and yk:
		vd_t sk = xkp1 - xk;
		vd_t yk = grad_fkp1 - grad_fk;

		// Advance step:
		xk = xkp1;
		grad_fk = grad_fkp1;
		cost0 = cost1;

		// Compute next inverse Hessian approximation (6.17):
		const double rhok = 1.0 / lina::dot(yk,sk);
		if (std::isnan(rhok) || std::isinf(rhok)){
			/* This might happen due to bad numerical problems. Exit
			 * before this influences anything else. */
			result.exit_code = RHO_DIVERGENCE;
			#ifdef DEBUG
			std::cout << "RHO_DIVERGENCE\n" << std::flush;
			#endif
			break;
		}
		Hk = (I - rhok * sk * lina::transpose(yk)) * Hk
		             * (I - rhok * yk * lina::transpose(sk))
		     + rhok * sk * lina::transpose(sk);
		++Hk_age;

		// Early exit:
		if (lina::norm(grad_fk) < epsilon){
			result.exit_code = CONVERGED;
			break;
		}
	}

	#ifdef DEBUG
		std::cout << "exit loop.\n" << std::flush;
	#endif

	/* Last point: */
	lina::fill_array(P, xk);
	result.history.emplace_back(cost0, BFGS_mode_t::FINISHED, P,
	                            convert_gradient(grad_fk), 0.0);

	return result;
}




template<typename point_t>
bool no_boundary(const point_t&)
{
	return true;
}


template<typename point_t, typename grad_t>
std::unique_ptr<point_t> no_jumps(const point_t&, const grad_t&)
{
	return std::unique_ptr<point_t>();
}

/*
 * The "preconditioner" template argument of the fallback_gradient_BFGS
 * method can be used to precondition the gradient before it is used to
 * compute the step.
 * It should return *true* if the step size has to be recomputed after
 * the preconditioning check (i.e. if a switch between preconditioning
 * strategies has been performed and the step size might not be accurate
 * anymore), or *false* if the step size can be evolved as before.
 */
template<typename lina>
bool no_preconditioning(typename lina::grad_t& g)
{
	return false;
}


template<size_t d, typename lina, bool last_redux=true>
BFGS_result_t<lina>
fallback_gradient_BFGS
    (const typename lina::point_t& x0,
     std::function<typename lina::real_t (const typename lina::point_t&)>
         cost,
     std::function<typename lina::grad_t (const typename lina::point_t&)>
          gradient,
     const size_t Nmax, const double epsilon,
     std::function<bool (const typename lina::point_t&)>
          in_bounds = no_boundary<typename lina::point_t>,
     std::function<std::unique_ptr<typename lina::point_t>
                   (const typename lina::point_t&,
                    const typename lina::grad_t&)>
          propose_jump = no_jumps<typename lina::point_t>,
     std::function<bool (typename lina::grad_t&)>
          preconditioner = no_preconditioning<lina>
    )
{
	typedef typename lina::point_t point_t;
	typedef typename lina::matrix_dxd_t Mdxd_t;
	typedef typename lina::column_vectord_t vd_t;

	/* Follows Nocedal & Wright: Numerical Optimization */
	Mdxd_t Hk(1e-3 * lina::identity());
	if (last_redux)
		Hk(d-1,d-1) *= 1e-5;
	const Mdxd_t I(lina::identity());

	/* Configuration of the Wolfe condition: */
	constexpr double c1 = 1e-4;
	constexpr double c2 = 0.9;

	/* Configuration of the fallback gradient descent: */
	constexpr size_t N_GRADIENT_STEPS = 40;
	constexpr double INITIAL_STEP = 1e-5;

	/* Init the starting point from the value given: */
	vd_t xk;
	lina::init_column_vectord(xk, x0);

	/* The trajectory in optimization space; and early exit: */
	BFGS_result_t<lina> result;
	result.exit_code = MAX_ITERATIONS;
	if (Nmax == 0)
		return result;

	/* Functions to take into consideration the boundary: */
	auto evaluate_cost = [&](const point_t& p) -> typename lina::real_t {
		if (!in_bounds(p))
			return std::numeric_limits<typename lina::real_t>::infinity();
		return cost(p);
	};

	auto evaluate_gradient = [&](const point_t& p) -> typename lina::grad_t {
		if (!in_bounds(p))
			return typename lina::grad_t();
		return gradient(p);
	};

	auto convert_gradient = [](const vd_t& grad) -> typename lina::grad_t {
		typename lina::grad_t dest;
		lina::fill_array(dest, grad);
		return dest;
	};

	/* The loop: */
	point_t P(x0);
	vd_t grad_fk;
	typename lina::grad_t G(gradient(x0));
	lina::init_column_vectord(grad_fk, G);
	double cost0 = cost(x0);
	size_t Hk_age = 0;
	double step = INITIAL_STEP;
	unsigned int gradient_steps = 0;
	unsigned int update_steps = 0;
	int jump_block = 0;
	std::unique_ptr<point_t> proposed;
	vd_t xkp1;
	for (size_t i=0; i<Nmax-1; ++i){
		/* From xk, compute the point: */
		lina::fill_array(P, xk);
		lina::fill_array(G, grad_fk);

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

		#ifdef DEBUG
			std::cout << "params[" << i << "]   = (";
			for (size_t j=0; j<d; ++j)
				std::cout << xk[j] << ",";
			std::cout << ")\n";
			std::cout << "gradient[" << i << "] = (";
			for (size_t j=0; j<d; ++j)
				std::cout << grad_fk[j] << ",";
			std::cout << ")\n";
			std::cout << "step[" << i << "] = " << step << "\n";
		#endif

		/* Keep track: */
		result.history.emplace_back(cost0, (gradient_steps > 0)
		                                ? BFGS_mode_t::FALLBACK_GRADIENT
		                                : BFGS_mode_t::BFGS,
		                            P, G, step);

		/* Check for proposed step: */
		proposed = propose_jump(P, G);
		if (proposed){
			double costprop = cost(*proposed);
			#ifdef DEBUG
				std::cout << "   proposed cost: " << costprop << "\n";
				std::cout << "   cost0:         " << cost0 << "\n";
			#endif
			if (costprop <= cost0 && jump_block == 0){
				/* Accept the jump, reset Hessian curvature info: */
				P = *proposed;
				lina::init_column_vectord(xk, P);
				cost0 = costprop;
				G = gradient(P);
				lina::init_column_vectord(grad_fk, G);
				Hk = lina::identity();
				Hk_age = 0;
				/* Block jumping for 10 steps, giving the gradient
				 * some time to leave the jump region: */
				jump_block = 10;
				#ifdef DEBUG
					std::cout << "   jumped!\n";
				#endif
				continue;
			} else {
				#ifdef DEBUG
					std::cout << "   did not jump.\n";
				#endif
			}
		}

		if (jump_block > 0)
			--jump_block;

		/* Check finiteness of cost function: */
		if (std::isnan(cost0) || std::isinf(cost0)){
			/* This might happen due to bad numerical problems. Exit
			 * before this influences anything else. */
			result.exit_code = COST_DIVERGENCE;
			break;
		}

		if (gradient_steps > 0){
			/* Perform gradient descent.
			 * Preconditioning: */
			vd_t grad_pre;
			lina::fill_array(G, grad_fk);
			bool step_reset = preconditioner(G);
			lina::init_column_vectord(grad_pre, G);

			if (step_reset)
				step = INITIAL_STEP;

			xkp1 = xk - step * grad_pre;
			point_t Pp1;
			lina::fill_array(Pp1, xkp1);
			bool accept_step = in_bounds(Pp1);
			const double gpre_norm = grad_pre.norm();;
			#ifdef DEBUG
			std::cout << "grad_pre[" << i << "] = (";
			for (size_t j=0; j<d; ++j)
				std::cout << grad_pre[j] << ",";
			std::cout << ")\n";
			#endif


			if (accept_step){
				const double cost1 = cost(Pp1);
				const double delta_cost_expected
				       = - step * gpre_norm * gpre_norm;
				if (cost1 < cost0 + c1 * delta_cost_expected){
					// Accept step.
					// Inverse Hessian update:
					vd_t grad_fkp1;
					lina::init_column_vectord(grad_fkp1, gradient(Pp1));
					if (lina::dot(grad_fkp1,grad_pre)
					    >= c2 * gpre_norm * gpre_norm)
					{
						vd_t sk = xkp1 - xk;
						vd_t yk = grad_fkp1 - grad_fk;
						const double rhok = 1.0 / lina::dot(yk,sk);
						if (!std::isnan(rhok) && !std::isinf(rhok)){
							Hk = (I - rhok * sk * lina::transpose(yk)) * Hk
							            * (I - rhok * yk * lina::transpose(sk))
							     + rhok * sk * lina::transpose(sk);
							++Hk_age;
						}
					}

					// Advance the gradient step:
					xk = xkp1;
					grad_fk = grad_fkp1;
					if (cost1 < cost0 + c1 * 1.1 * delta_cost_expected)
						step *= 1.1;
					cost0 = cost1;
					--gradient_steps;
					result.exit_code = MAX_ITERATIONS;
					accept_step = true;
				} else {
					accept_step = false;
					#ifdef DEBUG
					std::cout << "   do not accept step with \n"
					             "      cost0 = " << cost0 << "\n"
					             "      cost1 = " << cost1 << "\n";
					#endif
				}
			}
			if (!accept_step){
				// Do not accept step.
				step *= 0.125;

				if (xk - step * grad_fk == xk && xk - step * grad_pre == xk){
					/* Converged to the limit of gradient information. */
					#ifdef DEBUG
						std::cout << "EXIT GRADIENT: STEP SMALLER THAN "
						             "PRECISION.\n" << std::flush;
					#endif
					result.exit_code = CONVERGED;
					break;
				}

				// Make sure that we do not end up in an infinite loop here:
				const double xknorm = lina::dot(grad_pre,xk) / grad_pre.norm();
				if (step*gpre_norm*gpre_norm < std::max(1e-12*xknorm, 1e-40)){
					if (gradient_steps == N_GRADIENT_STEPS){
						// Just entered the gradient loop and already we fail
						// to perform any step - likely we are caught in an
						// infinite loop.
						// Here, we accept defeat and return with
						// LINE_SEARCH_FAIL set.
						#ifdef DEBUG
						std::cout << "EXIT GRADIENT: INFINITE LOOP.\n"
						                 "final cost:     "
						          << cost0 << "\n" << std::flush;
						if (in_bounds(Pp1)){
							double cost1 = cost(Pp1);
							const double gfk_gfp = std::abs(lina::dot(grad_fk, grad_pre));
							const double delta_cost_expected
							       = - step * gfk_gfp;
							std::cout << "proposed cost:  " << cost1 << "\n";
							std::cout << "      compare:  "
							          << cost0 + c1 * delta_cost_expected
							          << "\n";
							std::cout << "      c1:       " << c1 << "\n";
							std::cout << "delta_cost_exp. "
							          << delta_cost_expected << "\n";
							std::cout << "      step:     " << step << "\n"
							          << std::flush;
							std::cout << "grad_pre[" << i << "] = (";
							for (size_t j=0; j<d; ++j)
								std::cout << grad_pre[j] << ",";
							std::cout << ")\n";
							std::cout << "delta[" << i << "] = (";
							for (size_t j=0; j<d; ++j)
								std::cout << (xkp1-xk)[j] << ",";
							std::cout << ")\n";
						}
						#endif
						result.exit_code = LINESEARCH_FAIL;
						break;
					}
					--gradient_steps;
				}
				continue;
			}
		} else {
			/* BFGS: */

			/* Compute the search direction: */
			vd_t pk = (Hk_age > 3) ? - (Hk * grad_fk) : -1.0 * grad_fk;

			/* Compute the new point and gradient (Wolfe search, (3.6a & 3.6b).
			 * First, decide how deep we can reduce alpha until we do not change
			 * xk any more:
			 */
			constexpr double ALPHA_REDUX = 0.5;
			const double alpha_min = 1e-10 * lina::norm(xk) / lina::norm(pk);
			const size_t jmax_down
			   = std::max(
				    std::min(static_cast<size_t>(
				                std::floor(std::log(alpha_min)
				                                  / std::log(ALPHA_REDUX))),
				             static_cast<size_t>(100)),
				    static_cast<size_t>(10));
			double alpha = 1.0;
			vd_t xkp1(xk + alpha * pk);
			point_t Pp1;
			lina::fill_array(Pp1, xkp1);
			vd_t grad_fkp1;
			lina::init_column_vectord(grad_fkp1, evaluate_gradient(Pp1));
			double cost1 = evaluate_cost(Pp1);

			const double grad_fk_dot_pk = lina::dot(grad_fk, pk);
			bool wolfe_success = false;
			bool wolfe_1 = false;
			bool wolfe_2 = false;
			for (size_t j=0; j<jmax_down; ++j)
			{
				wolfe_1 = cost1 <= cost0 + c1 * alpha * grad_fk_dot_pk; // Wolfe 1, 3.6a
				wolfe_2 = lina::dot(grad_fkp1,pk) >= c2 * grad_fk_dot_pk; // Wolfe 2, 3.6b
				if (in_bounds(Pp1) && wolfe_1 && (wolfe_2 || Hk_age <= 3))
				{
					wolfe_success = true;
					break;
				}
				/* Reduce alpha and compute the new point: */
				alpha *= ALPHA_REDUX;
				xkp1 = xk + alpha * pk;
				lina::fill_array(Pp1, xkp1);
				lina::init_column_vectord(grad_fkp1, evaluate_gradient(Pp1));
				cost1 = evaluate_cost(Pp1);
			}

			if (!wolfe_success){
				if (wolfe_1){
					/* The sufficient reduction was not fulfilled. But maybe
					 * the Hessian changed a lot, so we might gain something
					 * from proceeding for another small step: */
					result.exit_code = MAX_ITERATIONS;

					/* But also make sure not to be forever stuck in this
					 * loop: */
					++update_steps;
					if (update_steps == 4){
						update_steps = 0;

						/* Switch back to gradient descent: */
						Hk = lina::identity();
						Hk_age = 0;
						gradient_steps = N_GRADIENT_STEPS;
						continue;
					}
				} else {
					/* No sufficient reduction possible. */
					result.exit_code = LINESEARCH_FAIL;

					/* Switch back to gradient descent: */
					Hk = lina::identity();
					Hk_age = 0;
					gradient_steps = N_GRADIENT_STEPS;
					continue;
				}
				#ifdef DEBUG
				std::cout << "Wolfe failed. W1=" << wolfe_1 << ", W2="
						  << wolfe_2 << "\n";
				#endif
			} else {
				result.exit_code = MAX_ITERATIONS;
			}


			/* Compute sk and yk: */
			vd_t sk = xkp1 - xk;
			vd_t yk = grad_fkp1 - grad_fk;

			/* Advance step: */
			xk = xkp1;
			grad_fk = grad_fkp1;
			cost0 = cost1;

			/* Compute next inverse Hessian approximation (6.17): */
			const double rhok = 1.0 / lina::dot(yk,sk);
			if (std::isnan(rhok) || std::isinf(rhok)){
				/* This might happen due to bad numerical problems. Exit
				 * before this influences anything else. */

				/* Switch back to gradient descent: */
				Hk = lina::identity();
				Hk_age = 0;
				gradient_steps = N_GRADIENT_STEPS;
				continue;
			}
			Hk = (I - rhok * sk * lina::transpose(yk)) * Hk
				         * (I - rhok * yk * lina::transpose(sk))
				 + rhok * sk * lina::transpose(sk);
			++Hk_age;

			/* With increasing number of BFGS steps, increase gradient
			 * descent steps, i.e. forget the memory of where we were
			 * in terms of step size: */
			if (step != INITIAL_STEP)
				step = std::min(2.0*step, INITIAL_STEP);
		}

		/* Early exit: */
		if (lina::norm(grad_fk) < epsilon){
			result.exit_code = CONVERGED;
			#ifdef DEBUG
			std::cout << "CONVERGED!\n" << std::flush;
			#endif
			break;
		}
	}

	#ifdef DEBUG
	if (result.exit_code == MAX_ITERATIONS){
		std::cout << "MAX_ITERATIONS\ncost=" << cost0 << "\n" << std::flush;
	}
		std::cout << "exit loop.\n" << std::flush;
	#endif

	/* Last point: */
	lina::fill_array(P, xk);
	result.history.emplace_back(cost0, BFGS_mode_t::FINISHED, P,
	                            convert_gradient(grad_fk), step);

	return result;
}

#endif // include guard