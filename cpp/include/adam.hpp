/*
 * Implementation of the ADAM method.
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
#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

#include <../include/linalg.hpp>


enum class ADAM_exit_code_t {
	MAX_ITERATIONS, CONVERGED
};

template<typename point_t>
struct ADAM_result_t {
	ADAM_exit_code_t exit_code;
	std::vector<std::pair<double, point_t>> history;
};


template<size_t d, typename lina>
ADAM_result_t<typename lina::point_t>
    ADAM(const typename lina::point_t& x0,
         std::function<typename lina::real_t (const typename lina::point_t&)>
              cost,
         std::function<typename lina::grad_t (const typename lina::point_t&)>
              gradient,
         const size_t Nmax, const double epsilon, const double a,
         const double b1, const double b2
    )
{
	typedef typename lina::point_t point_t;
	typedef typename lina::grad_t grad_t;
	typedef typename lina::real_t real_t;
	typedef typename lina::column_vectord_t vd_t;
	vd_t mt,vt;
	for (size_t i=0; i<d; ++i){
		mt[i] = 0.0;
		vt[i] = 0.0;
	}

	constexpr double eps = 1e-8;

	vd_t gt, gt2;
	vd_t xk;
	lina::init_column_vectord(xk, x0);
	point_t P(x0);
	real_t b1t = b1;
	real_t b2t = b2;
	std::vector<std::pair<double, point_t>> history;
	double step = a;
	double C = cost(P);
	for (size_t i=0; i<Nmax; ++i){
		/* From xk, compute the point: */
		lina::fill_array(P, xk);

		/* Compute gradient and cost: */
		grad_t grad_f(gradient(P));
		lina::init_column_vectord(gt, grad_f);

		/* History and exit condition:: */
		history.emplace_back(C, P);
		if (lina::norm(gt) < epsilon)
			return {ADAM_exit_code_t::CONVERGED, history};

		/* Compute element-wise squared gradient: */
		lina::init_column_vectord(gt2, grad_f);
		for (size_t j=0; j<d; ++j){
			gt2[j] *= gt2[j];
		}

		/* Compute biased first and second moment estimates: */
		mt = b1 * mt + (1.0 - b1) * gt;
		vt = b2 * vt + (1.0 - b2) * gt2;

		/* Bias correction: */
		vd_t mt_hat = mt * (1.0 / (1.0 - b1t));
		vd_t vt_hat = vt * (1.0 / (1.0 - b2t));

		/* Propose parameter update but only accept a reduction: */
		vd_t xkp1;
		xkp1 = xk;
		for (size_t j=0; j<d; ++j)
			xkp1[j] -= step * mt_hat[j] / (std::sqrt(vt_hat[j]) + eps);

		/* Evaluate new cost: */
		lina::fill_array(P, xkp1);
		const double C_new = cost(P);
		if (C_new < C){
			xk = xkp1;
			step = std::min(step * 1.1, 1e15);
		} else {
			/* Reject step. */
			step = std::max(step * 0.25, 1e-10);
		}

		/* Update parameters: */

		/* Update the memory parameters: */
		b1t *= b1;
		b2t *= b2;

		std::cout << "step: " << step << "\n" << std::flush;
	}

	return {ADAM_exit_code_t::MAX_ITERATIONS, history};
}
