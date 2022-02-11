/*
 * Cost function.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
 *          Sebastian von Specht
 *
 * Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam,
 *                         Sebastian von Specht
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

#include <../include/cost.hpp>
#include <stdexcept>
#include <iostream>


using doomercat::LabordeProjectedDataSet;
using doomercat::CostFunction;
using doomercat::Cost;

static real5v mercator_project_residual(const Quaternion<real5v>& qv,
                                        const vec3_5v& uvw, real5v k0)
{
	/* Create the quaternion representing uvw:
	 *   const Quaternion<real5v> l(constant5(0.0), uvw[0], uvw[1], uvw[2]);
	 * Rotate l back to standard cylinder coordinates:
	 *	 const Quaternion<real5v> r(qrot(qv.conj(), l, qv));
	 */

	/* Rotate l back to standard cylinder coordinates: */
	const Quaternion<real5v> r(
	     qrot(qv.conj(), ImaginaryQuaternion<real5v>(uvw[0], uvw[1], uvw[2]),
	          qv)
	);

	/* Compute the Mercator projection: */
	const real5v norm(sqrt(  r.i()*r.i()
	                       + r.j()*r.j()
	                       + r.k()*r.k()));
	/* TODO: Is the norm important? Do we need one? Two? None? */
	//real5v x(k0 * r.i() / norm_h);
	//real5v y(k0 * r.j() / norm_h);
	//real5v z(k0 * norm * atanh(r.k() / norm));
	//const real5v x(k0 * r.i() / norm);
	//const real5v y(k0 * r.j() / norm);
	//const real5v z(k0 * atanh(r.k() / norm));
	/* Now we got the scalar residual: */
	return sqrt(  pow2(r.i() - (k0 * r.i() / norm)) // r.i() - x
	            + pow2(r.j() - (k0 * r.j() / norm)) // r.j() - y
	            + pow2(r.k() - (k0 * atanh(r.k() / norm))) // r.k() - z
	           );
}


static real5v compute_cost(const LabordeProjectedDataSet& projected_data,
                           unsigned int pnorm, double k0_ap, double sigma_k0)
{
	/* Initialize the sum. */
	real5v cost = constant5(0.0);
	const Quaternion<real5v>& q(projected_data.cylinder()
	                              .rotation_quaternion());

	/* Unit quaternion (versor) in q direction: */
	const Quaternion<real5v> qv(q / q.norm());

	const real5v& k0(projected_data.cylinder().k0());

	/* Compute the sum of weighted, potentiated residuals: */
	for (size_t i=0; i<projected_data.size(); ++i){
		// Compute delta = mercator_project_residual(qv, uvw[i], k0)
		// then cost += w[i] * (delta)^pnorm
		cost += projected_data.w(i)
		        * pow(mercator_project_residual(qv, projected_data.uvw(i), k0),
		              (int)pnorm);
	}

	/* The pnorm normalization: */
	//cost = pow(cost, 1.0 / (double)pnorm);

	/* Add the k0  prior: */
	if (k0.value() < k0_ap)
		cost += pow2((k0 - k0_ap)/sigma_k0);

	return log(cost);
}



Cost::Cost(const LabordeProjectedDataSet& projected_data, unsigned int pnorm,
           double k0_ap, double sigma_k0)
    : cost(compute_cost(projected_data, pnorm, k0_ap, sigma_k0))
{
}

Cost::operator double() const
{
	return cost.value();
}

std::array<double,5> Cost::grad() const
{
	std::array<double,5> g;
	for (int i=0; i<5; ++i)
		g[i] = cost.derivative(i);
	return g;
}




CostFunction::CostFunction(unsigned int pnorm, double k0_ap, double sigma_k0)
    : pnorm(pnorm), k0_ap(k0_ap), sigma_k0(sigma_k0)
{
	if (pnorm < 2)
		throw std::runtime_error("pnorm too small");
}

Cost CostFunction::operator()(const LabordeProjectedDataSet& data) const
{
	return Cost(data, pnorm, k0_ap, sigma_k0);
}

