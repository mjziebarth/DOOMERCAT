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


// Bugfix code:
#include <chrono>
#include <thread>
#include <iostream>

using doomercat::LabordeCylinder;
using doomercat::CostFunction;
using doomercat::DataSet;
using doomercat::Cost;
using doomercat::LabordeProjectedDataSet;
using doomercat::result_t;
using doomercat::return_state_t;



std::vector<result_t>
doomercat::billo_gradient_descent(std::shared_ptr<const DataSet> data,
                                  std::shared_ptr<const LabordeCylinder> cyl0,
                                  const CostFunction& cost_function,
                                  size_t Nmax)
{
	typedef Quaternion<real5v> quat_t;
	std::vector<result_t> result;
	result.reserve(Nmax);
	const double f = cyl0->f();

	double step = 1e-3;

	/* Init the initial config: */
	std::shared_ptr<const LabordeCylinder> cylinder(cyl0);
	Cost cost(cost_function(LabordeProjectedDataSet(data,cylinder)));

	double cost_old = cost;
	std::array<double,5> grad = cost.grad();
	for (size_t i=0; i<Nmax; ++i){
		/* Propose a new cylinder based on the gradient: */
		const quat_t& qold(cylinder->rotation_quaternion());
		const double qr = qold.r().value();
		const double qi = qold.i().value();
		const double qj = qold.j().value();
		const double qk = qold.k().value();
		const double k0 = cylinder->k0().value();
		std::shared_ptr<const LabordeCylinder> proposed_cylinder
		   = LabordeCylinder::from_parameters(qr - step*grad[0],
		                                      qi - step*grad[1],
		                                      qj - step*grad[2],
		                                      qk - step*grad[3],
		                                      k0,// - step*grad[4],
		                                      f);

		/* Compute cost and gradient: */
		LabordeProjectedDataSet pd(data, proposed_cylinder);
		cost = cost_function(pd);
		double C = cost;

		/* Accept or decline and adjust step width accordingly: */
		if (C < cost_old){
			/* accept. */
			step *= 1.1;
			cylinder = proposed_cylinder;
			grad = cost.grad();
			cost_old = C;
		} else {
			step = std::max(1e-10, step*0.25);
		}

		result.push_back({cost_old, *cylinder});
	}

	return result;
}

std::vector<result_t>
doomercat::bfgs_optimize(std::shared_ptr<const DataSet> data,
                         std::shared_ptr<const LabordeCylinder> cyl0,
                         const CostFunction& cost_function,
                         const size_t Nmax)
{
	if (!data || !cyl0)
		throw std::runtime_error("data or cyl0 nullptr.");

	/* Defining the linear algebra implementation: */
	typedef linalg_t<5,double> lina_t;

	/* Initial config: */
	const double f = cyl0->f();
	std::shared_ptr<const LabordeCylinder> cylinder(cyl0);
	Cost cost(cost_function(LabordeProjectedDataSet(data,cylinder)));

	/* Indicating the cached cost: */
	std::array<double,5> last_x;

	/* Variable transformation in k0: */
	auto x_to_k0 = [](double x) -> double {
		return 1.0 / (1.0 + std::exp(x));
	};

	auto d_k0_d_x = [](double x) -> double {
		const double ex = std::exp(x);
		const double k0 = 1.0 / (1.0 + ex);
		return - k0 * k0 * ex;
	};

	auto k0_to_x = [](double k0) -> double {
		return std::log(1.0 / k0 - 1.0);
	};


	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	auto check_cached_cost = [&](const std::array<double,5>& x) {
		bool x_new = false;
		for (int i=0; i<5; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			cylinder = LabordeCylinder::from_parameters(x[0], x[1], x[2], x[3],
			                                            x_to_k0(x[4]), f);
			cost = cost_function(LabordeProjectedDataSet(data,cylinder));
			last_x = x;
		}
	};


	auto cost_lambda = [&](const std::array<double,5>& x) -> double {
		/* Check if we can used cached computation: */
		check_cached_cost(x);
		return cost;
	};


	auto gradient_lambda = [&](const std::array<double,5>& x)
	   -> std::array<double,5>
	{
		/* Check if we can used cached computation: */
		check_cached_cost(x);

		/* Comupte the gradient and apply chain rule for k0: */
		std::array<double,5> grad = cost.grad();
		grad[4] *= d_k0_d_x(x[4]);

		/* Return the gradient: */
		return grad;
	};

	/* Initial config and optimization: */
	const std::array<double,5> x0({cyl0->rotation_quaternion().r().value(),
	                               cyl0->rotation_quaternion().i().value(),
	                               cyl0->rotation_quaternion().j().value(),
	                               cyl0->rotation_quaternion().k().value(),
	                               std::max(k0_to_x(cyl0->k0().value()),
	                                        -15.0)});
	BFGS_result_t<std::array<double,5>> y
	//   = BFGS<5,lina_t>(x0, cost_lambda, gradient_lambda,
	//                              Nmax, 1e-5);
	   = fallback_gradient_BFGS<5,lina_t>(x0, cost_lambda, gradient_lambda,
	                                      Nmax, 1e-5);


	std::vector<result_t> res;
	res.reserve(y.history.size());
	for (auto z : y.history){
		LabordeCylinder cyl(z.second[0], z.second[1], z.second[2],
		                    z.second[3], x_to_k0(z.second[4]), f);
		return_state_t state = ERROR;
		switch (y.exit_code){
			case CONVERGED:
				state = CONVERGED;
				break;
			case MAX_ITERATIONS:
				state = MAX_ITERATIONS;
				break;
			case RHO_DIVERGENCE:
			case LINESEARCH_FAIL:
			case COST_DIVERGENCE:
				state = ERROR;
		}

		res.push_back({z.first, cyl, state});
	}

	return res;
}

