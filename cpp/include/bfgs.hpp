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

// Debug:
#ifdef DEBUG
#include <iostream>
#endif

#include <../include/linalg.hpp>


enum BFGS_exit_code_t {
	MAX_ITERATIONS, CONVERGED, RHO_DIVERGENCE, LINESEARCH_FAIL
};

template<typename point_t>
struct BFGS_result_t {
	BFGS_exit_code_t exit_code;
	std::vector<std::pair<double, point_t>> history;
};


template<size_t d, typename lina>
BFGS_result_t<typename lina::point_t>
    BFGS(const typename lina::point_t& x0,
         std::function<typename lina::real_t (const typename lina::point_t&)>
              cost,
         std::function<typename lina::grad_t (const typename lina::point_t&)>
              gradient,
         const size_t Nmax, const double epsilon
    )
{
	typedef typename lina::real_t real_t;
	typedef typename lina::point_t point_t;
	typedef typename lina::matrix_dxd_t Mdxd_t;
	typedef typename lina::column_vectord_t vd_t;
	typedef typename lina::row_vectord_t row_vd_t;

	/* Follows Nocedal & Wright: Numerical Optimization */
	Mdxd_t Hk(lina::identity());
	const Mdxd_t I(lina::identity());

	/* Configuration of the Wolfe condition: */
	const double c1 = 1e-4;
	const double c2 = 0.9;

	/* Init the starting point from the value given: */
	vd_t xk;
	lina::init_column_vectord(xk, x0);

	/* The trajectory in optimization space; and early exit: */
	BFGS_result_t<point_t> result;
	result.exit_code = MAX_ITERATIONS;
	if (Nmax == 0)
		return result;

	/* The loop: */
	point_t P(x0);
	vd_t grad_fk;
	lina::init_column_vectord(grad_fk, gradient(x0));
	double cost0 = cost(x0);
	for (size_t i=0; i<Nmax-1; ++i){
		// From xk, compute the point:
		lina::fill_array(P, xk);

		// Keep track:
		result.history.emplace_back(cost0, P);

		#ifdef DEBUG
			std::cout << "gradient[" << i << "] = (";
			for (size_t j=0; j<d; ++j)
				std::cout << grad_fk[j] << ",";
			std::cout << ")\n";
		#endif

		// Compute the search direction:
		vd_t pk = - (Hk * grad_fk);


		#ifdef DEBUG
			std::cout << "   -> pk    = (";
			for (size_t j=0; j<d; ++j)
				std::cout << pk[j] << ",";
			std::cout << ")\n";
		#endif



		// Compute the new point and gradient (Wolfe search, (3.6a & 3.6b).
		// First, decide how deep we can reduce alpha until we do not change
		// xk any more:
		constexpr double ALPHA_REDUX = 0.5;
		const double alpha_min = 1e-10 * lina::norm(xk) / lina::norm(pk);
		const size_t jmax_down
		   = std::min(static_cast<size_t>(std::floor(std::log(alpha_min)
		                                             / std::log(ALPHA_REDUX))),
		              static_cast<size_t>(100));
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
			wolfe_1 = cost1 <= cost0 + c1 * alpha * grad_fk_dot_pk;   // Wolfe 1, 3.6a
			wolfe_2 = lina::dot(grad_fkp1,pk) >= c2 * grad_fk_dot_pk; // Wolfe 2, 3.6b
			if (wolfe_1 && wolfe_2)
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
			} else {
				/* No sufficient reduction possible. */
				result.exit_code = LINESEARCH_FAIL;
				return result;
			}
			#ifdef DEBUG
			std::cout << "Wolfe failed. W1=" << wolfe_1 << ", W2="
			          << wolfe_2 << "\n";
			#endif
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


		#ifdef DEBUG
			std::cout << "   -> rhok  =  " << rhok << "\n";
		#endif

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
			break;
		}
		Hk = (I - rhok * sk * lina::transpose(yk)) * Hk
		             * (I - rhok * yk * lina::transpose(sk))
		     + rhok * sk * lina::transpose(sk);

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
	result.history.emplace_back(cost0, P);

	return result;
}

