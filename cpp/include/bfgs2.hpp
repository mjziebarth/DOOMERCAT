/*
 * Implementation of the BFGS method.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
 *               2024 Technische Unviversität München
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
#include <iostream>

#include <../include/linalg.hpp>
#include <../include/exitcode.hpp>


#ifndef DOOMERCAT_BFGS2_H
#define DOOMERCAT_BFGS2_H

namespace doomercat {

template<typename vd_t>
bool any_nan(const vd_t& vec){
    for (uint_fast8_t i=0; i<4; ++i)
        if (std::isnan(vec[i]))
            return true;

    return false;
}

template<typename vd_t>
bool equal_to_double_precision(const vd_t& v0, const vd_t& v1)
{
    for (uint_fast8_t i=0; i<4; ++i)
    {
        double v0i = v0[i];
        double v1i = v1[i];
        if (v0i != v1i)
            return false;
    }
    return true;
}


template<
    typename real_t,
    typename parameters_t,
    typename vd_t,
    typename cost_fun_t,
    typename gradient_fun_t,
    typename in_bounds_fun_t
>
bool
wolfe_linesearch(
    const vd_t& xk,
    const vd_t& pk,
    const vd_t& grad_fk,
    const double cost0,
    vd_t& xkp1,
    vd_t& grad_fkp1,
    double& cost1,
    double& alpha_memory,
    size_t Bk_age,
    cost_fun_t evaluate_cost,
    gradient_fun_t evaluate_gradient,
    in_bounds_fun_t in_bounds
)
{

    /* Configuration of the Wolfe condition: */
    constexpr double c1 = 1e-4;
    constexpr double c2 = 0.9;


    constexpr size_t d = parameters_t::ndim;
    typedef linalg_t<d, real_t> lina;
    typedef typename lina::point_t point_t;

    point_t P;
    lina::fill_array(P, xk);

    /*
     * Here we compute the factor by which we reduce 'alpha' in each iteration
     * of the backtracking line search.
     * Our goal is to perform few function evaluations in the line search. To
     * this end, we decrement the step size by the square root of the previous
     * step size. This ensures that when the required step size is similar over
     * multiple consecutive iterations, the number of necessary function
     * evaluations is in the order of 2-3.
     * Since we always start from 1.0, the asymptotic quadratic convergence
     * is retained.
     */
    const double ALPHA_REDUX
        = (alpha_memory == 0) ? 0.5
            : std::min(std::sqrt(alpha_memory), 0.5);

    /* A note on alpha_min: Here, we compute alpha_min so that the resulting
     * step does not change the parameters 'xk' expressed in double: */
    const double alpha_min = 0.1 * std::numeric_limits<double>::epsilon()
                                 * lina::norm(xk) / lina::norm(pk);

    /* Start from a slightly higher alpha than before: */
    double alpha = 1.0;

    const size_t jmax_down
    = 2 * std::max(
            static_cast<size_t>(std::floor(
                std::log(alpha_min) / std::log(ALPHA_REDUX)
            )),
            static_cast<size_t>(3)
    );
    xkp1 = xk + alpha * pk;

    point_t Pp1;
    lina::fill_array(Pp1, xkp1);
    lina::init_column_vectord(grad_fkp1, evaluate_gradient(Pp1));
    cost1 = evaluate_cost(Pp1);

    const real_t grad_fk_dot_pk = lina::dot(grad_fk, pk);
    bool wolfe_success = false;
    bool wolfe_1 = false;
    bool wolfe_2 = false;
    bool strong_wolfe_2 = false;
    /* Right hand side of the strong Wolfe condition #2: */
    const real_t sw2_rhs = c2 * std::abs(grad_fk_dot_pk);

    /*
     * In a first iteration we try to find a step size that satisfies the
     * strong Wolfe conditions.
     * While doing so, we already look for step sizes that satisfy the
     * normal Wolfe conditions (free of charge!)
     */
    std::vector<std::tuple<double, real_t, vd_t>> history;
    std::vector<size_t> normal_wolfe_2_alphas;
    std::vector<size_t> wolfe_1_alphas;
    for (size_t j=0; j<jmax_down; ++j)
    {
        /* Remember this step: */
        history.push_back(
            std::make_tuple(alpha, cost1, grad_fkp1)
        );

        /* Check whether it's valid: */
        bool in_bds = in_bounds(Pp1);

        /* Wolfe 1 (3.6a): */
        wolfe_1 = (cost1 <= cost0 + c1 * alpha * grad_fk_dot_pk);

        /* Wolfe 2 (3.7b): */
        real_t grad_fkp1__dot__pk = lina::dot(grad_fkp1,pk);
        wolfe_2 = (grad_fkp1__dot__pk >= c2 * grad_fk_dot_pk);

        /* Strong Wolfe 2 (3.7b): */
        strong_wolfe_2 = (std::abs(grad_fkp1__dot__pk) <= sw2_rhs);

        if (in_bds && wolfe_1)
        {
            /* Keep track of the successful */
            wolfe_1_alphas.push_back(j);

            if (strong_wolfe_2)
            {
                alpha_memory = alpha;
                return true;
            }
            else if (wolfe_2)
            {
                normal_wolfe_2_alphas.push_back(j);
            }
        }
        /* Reduce alpha and compute the new point: */
        alpha *= ALPHA_REDUX;
        xkp1 = xk + alpha * pk;

        /* Check if we have maxed out the reduction.
         * If so, we have not found a step width that satisfies the strong
         * Wolfe conditions. We might have still found a step width that
         * satisfies the weak Wolfe conditions, or at least a step width
         * that satisfies the first Wolfe condition (i.e. reduces the cost
         * function).
         * To check these cases, exit this loop: */
        if (equal_to_double_precision(xkp1, xk)){
            alpha_memory = 1.0;
            break;
        }

        /* Compute the cost and gradient at the newly proposed point: */
        lina::fill_array(Pp1, xkp1);
        lina::init_column_vectord(grad_fkp1, evaluate_gradient(Pp1));
        cost1 = evaluate_cost(Pp1);
    }

    if (!normal_wolfe_2_alphas.empty())
    {
        /* While the search for strong Wolfe conditions failed,
         * we succeeded in finding points that comply to the normal
         * Wolfe conditions. Use one of them!
         * Which one, you may ask? The one with least cost.
         */
        auto it = normal_wolfe_2_alphas.cbegin();
        std::tuple<double,real_t,vd_t> best;
        best = history[*it];
        for (++it; it != normal_wolfe_2_alphas.cend(); ++it)
        {
            if (std::get<1>(history[*it]) < std::get<1>(best))
                best = history[*it];
        }

        /* Unwrap that solution */
        alpha = std::get<0>(best);
        xkp1 = xk + alpha * pk;
        grad_fkp1 = std::get<2>(best);
        cost1 = std::get<1>(best);
        alpha_memory = alpha;
        return true;
    }

    if (!wolfe_1_alphas.empty())
    {
        /* Wolfe 2 was generally not successful but we found at least
         * a success of Wolfe 1. */
        auto it = wolfe_1_alphas.cbegin();
        std::tuple<double,real_t,vd_t> best;
        best = history[*it];
        for (++it; it != wolfe_1_alphas.cend(); ++it)
        {
            if (std::get<1>(history[*it]) < std::get<1>(best))
                best = history[*it];
        }

        /* Unwrap that solution */
        alpha = std::get<0>(best);
        xkp1 = xk + alpha * pk;
        grad_fkp1 = std::get<2>(best);
        cost1 = std::get<1>(best);
        alpha_memory = alpha;
        return true;
    }

    alpha_memory = 1.0;
    return false;
};


enum class BFGS_mode_t {
	BFGS=1, FINISHED=0
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
	exit_code_t exit_code;
	std::vector<BFGS_step_t<lina>> history;
};



template<
    bool last_redux,
    typename real_t,
    typename parameters_t,
    typename costfun_t,
    typename gradfun_t,
    typename boundsfun_t
>
BFGS_result_t<
    linalg_t<
        parameters_t::ndim,
        typename parameters_t::real_t
    >
>
dampened_BFGS
(
    const parameters_t& x0,
    costfun_t costfun,
    gradfun_t gradient,
    const size_t Nmax,
    const double epsilon,
    boundsfun_t in_bounds
)
{
    /* Typedefs for brevity: */
    constexpr size_t d = parameters_t::ndim;
    typedef linalg_t<d, real_t> lina;
    typedef typename lina::matrix_dxd_t Mdxd_t;
    typedef typename lina::column_vectord_t vd_t;
    typedef typename lina::point_t point_t;

    static_assert(d < 256, "Need to change dimension iteration types.");

    /* Wolfe line search starting point: */
    double alpha = 1.0;

    /* Follows Nocedal & Wright: Numerical Optimization */
    Mdxd_t Bk;
    size_t Bk_age = 0;
    const Mdxd_t I(lina::identity());

    auto reset_Bk = [&Bk, &I, &Bk_age, &alpha]()
    {
        Bk = 1e3 * I;
        if constexpr (last_redux)
            Bk(d-1,d-1) *= 1e5;
        Bk_age = 0;
        alpha = 1.0;
    };
    reset_Bk();


    /* Init the starting point from the value given: */
    vd_t xk;
    lina::init_column_vectord(xk, x0);

    /* The trajectory in optimization space; and early exit: */
    BFGS_result_t<lina> result;
    result.exit_code = MAX_ITERATIONS;
    if (Nmax == 0)
        return result;

    /* Functions to take into consideration the boundary: */
    size_t cost_eval = 0;
    auto evaluate_cost = [&](const point_t& p) -> typename lina::real_t {
        if (!in_bounds(p))
            return std::numeric_limits<typename lina::real_t>::infinity();
        ++cost_eval;
        return costfun(p);
    };

    auto evaluate_gradient = [&](const point_t& p) -> typename lina::grad_t {
        if (!in_bounds(p))
            return typename lina::grad_t();
        return gradient(p);
    };

    auto is_finite = [](double cost_i, const vd_t& grad_fk_i) -> bool
    {
        if (std::isnan(cost_i) || std::isinf(cost_i)){
            return false;
        }
        for (uint_fast8_t j=0; j<d; ++j){
            if (std::isnan(grad_fk_i[j]) || std::isinf(grad_fk_i[j])){
                return false;
            }
        }
        return true;
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
    double cost0 = costfun(x0);
    for (size_t i=0; i<Nmax-1; ++i)
    {
        /* Check sanity of the parameters: */
        bool sane = true;
        for (uint_fast8_t j=0; j<parameters_t::ndim; ++j){
            if (std::isnan(xk[j]) || std::isinf(xk[j])){
                sane = false;
                std::cerr << "The parameter vector was corrupted. Aborting.\n";
                break;
            }
        }
        if (!sane)
            break;

        result.exit_code = MAX_ITERATIONS;
        /* From xk, compute the point: */
        lina::fill_array(P, xk);
        lina::fill_array(G, grad_fk);

        /* Keep track: */
        result.history.emplace_back(
            cost0,
            BFGS_mode_t::BFGS,
            P,
            G,
            alpha
        );

        /* Check finiteness of cost function and gradient: */
        if (!is_finite(cost0, grad_fk)){
            /* This might happen due to bad numerical problems. Exit
             * before this influences anything else. */
            result.exit_code = COST_DIVERGENCE;
            break;
        }



        /* Compute the search direction: */
        vd_t pk = (Bk_age > 3)
            ? lina::solve(Bk, -1.0 * grad_fk)
            : -1.0 * grad_fk;

        /* Compute the new point and gradient (Wolfe search, (3.6a & 3.6b).
         * First, decide how deep we can reduce alpha until we do not change
         * xk any more:
         */

        vd_t xkp1;
        vd_t grad_fkp1;
        double cost1;
        const double alpha_on_enter = alpha;
        bool wolfe_success
        = wolfe_linesearch<real_t,parameters_t>(
            xk, pk, grad_fk, cost0, xkp1,
            grad_fkp1, cost1, alpha, Bk_age,
            evaluate_cost, evaluate_gradient,
            in_bounds);

        if (!wolfe_success)
        {
            /* No condition was fulfilled. */
            result.exit_code = LINESEARCH_FAIL;
            if (Bk_age == 0 && alpha_on_enter == 1.0){
                /* We have already previously reset the Hessian.
                 * Even with pure gradient descent, we cannot obtain a
                 * sufficient reduction.
                 * We have to end the program here.
                 */
                result.exit_code = LINESEARCH_FAIL;
                break;
            }
            reset_Bk();

            /* Since we have not found an acceptable step, continue: */
            continue;
        }

        /* Compute sk and yk: */
        vd_t sk = xkp1 - xk;
        vd_t yk = grad_fkp1 - grad_fk;

        /* Advance step: */
        xk = xkp1;
        grad_fk = grad_fkp1;
        cost0 = cost1;


        /* Here we use the dampened BFGS update
        * (procedure 18.2 from Nocedal & Wright, 2006) */
        real_t theta;
        const real_t sk_yk = lina::dot(sk, yk);
        vd_t Bk_sk = Bk * sk;
        const real_t sk_Bk_sk = lina::dot(sk, Bk_sk);
        constexpr double c3 = 0.1;
        if (sk_yk >= c3 * sk_Bk_sk)
            theta = 1.0;
        else
            theta = (1.0 - c3) * sk_Bk_sk / (sk_Bk_sk - sk_yk);


        /* Compute next Hessian approximation (18.16): */
        if (std::isnan(theta) || std::isinf(theta)){
            /* This might happen due to bad numerical problems. Exit
             * before this influences anything else. */
            reset_Bk();
            continue;
        } else {
            vd_t rk = theta * yk + (1.0 - theta) * Bk_sk;
            Bk = Bk - Bk * sk * lina::transpose(sk) * Bk * (1.0 / sk_Bk_sk)
                 + rk * lina::transpose(rk) * (1.0 / lina::dot(sk, rk));
            ++Bk_age;
        }

        /* Early exit: */
        if (lina::norm(grad_fk) < epsilon){
            result.exit_code = CONVERGED;
            break;
        }
    }


    /* Last point: */
    lina::fill_array(P, xk);
    result.history.emplace_back(
        cost0,
        BFGS_mode_t::FINISHED,
        P,
        convert_gradient(grad_fk),
        alpha
    );

    return result;
}

} // end namespace

#endif // include guard