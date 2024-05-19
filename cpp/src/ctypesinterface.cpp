/*
 * Methods suitable for interfacing with Ctypes.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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

#include <../include/optimize.hpp>
#include <../include/ctypesinterface.hpp>
#include <../include/cost_hotine.hpp>
#include <../include/hotine.hpp>
#include <../include/parameters.hpp>
#include <iostream>
#include <memory>
#include <cmath>
#include <cstddef>
#include <vector>


// Bugfix code:
#include <chrono>
#include <thread>

using doomercat::CostFunctionHotine;
using doomercat::CostFunctionHotineInf;
using doomercat::CostHotine;
using doomercat::HotineParameters;
using doomercat::HotineObliqueMercator;
using doomercat::HotineObliqueMercatorProjection;
using doomercat::SimpleDataSet;
using doomercat::WeightedDataSet;
using doomercat::DataSetWithHeight;
using doomercat::WeightedDataSetWithHeight;

template<bool wrap_plane, typename numeric_t, typename T>
void cost_batch(const T& data, double pnorm, double k0_ap, double sigma_k0,
                unsigned short proot, bool logarithmic, const size_t M,
                const double* lonc, const double* lat0, const double* alpha,
                const double* k0, double f, double* result)
{
	if (std::isinf(pnorm)){
		const CostFunctionHotineInf<numeric_t> cfun(k0_ap, sigma_k0,
		                                            logarithmic,
		                                            false);

		/* Compute the Hotine Mercator projections now: */
		#pragma omp parallel for
		for (size_t i=0; i<M; ++i){
			if constexpr (wrap_plane){
				HotineObliqueMercator<numeric_t> hom(
					HotineParameters<numeric_t>(
						deg2rad(lonc[i]),
						deg2rad(lat0[i]),
						deg2rad(alpha[i]),
						k0[i]
					),
					f
				);

				/* Compute the cost: */
				result[i] = cfun(data, hom);
			} else {
				HotineObliqueMercator<numeric_t> hom(
					deg2rad(lonc[i]),
					deg2rad(lat0[i]),
					deg2rad(alpha[i]),
					k0[i],
					f
				);

				/* Compute the cost: */
				result[i] = cfun(data, hom);
			}
		}

	} else {
		const CostFunctionHotine<numeric_t> cfun(pnorm, k0_ap, sigma_k0,
		                                         proot > 0u, logarithmic,
		                                         false);

		/* Compute the Hotine Mercator projections now: */
		#pragma omp parallel for
		for (size_t i=0; i<M; ++i){
			if constexpr (wrap_plane){
				HotineObliqueMercator<numeric_t> hom(
					HotineParameters<numeric_t>(
						deg2rad(lonc[i]),
						deg2rad(lat0[i]),
						deg2rad(alpha[i]),
						k0[i]
					),
					f
				);

				/* Compute the cost: */
				result[i] = cfun(data, hom);
			} else {
				HotineObliqueMercator<numeric_t> hom(
					deg2rad(lonc[i]),
					deg2rad(lat0[i]),
					deg2rad(alpha[i]),
					k0[i],
					f
				);

				/* Compute the cost: */
				result[i] = cfun(data, hom);
			};
		}
	}
}


int compute_cost_hotine_batch(const size_t N, const double* lon,
        const double* lat, const double* h, const double* w, const size_t M,
        const double* lonc, const double* lat0, const double* alpha,
        const double* k0, double a, double f, double pnorm, double k0_ap,
        double sigma_k0, unsigned short proot,
        unsigned short logarithmic, unsigned short wrap_plane,
        unsigned short precision,
        double* result)
{
	/*
	 * Compute the cost function for many different HOM parameters
	 * given a single data set.
	 */
	typedef std::variant<WeightedDataSetWithHeight,
	                     WeightedDataSet,
	                     DataSetWithHeight,
	                     SimpleDataSet>
	    dataset_t;
	auto make_data = [=]() -> dataset_t
	{
		if (w){
			if (h){
				return WeightedDataSetWithHeight(
					N, lon, lat, h, w, a, f
				);
			} else {
				return WeightedDataSet(
					N, lon, lat, w
				);
			}
		} else {
			if (h){
				return DataSetWithHeight(
					N, lon, lat, h, a, f
				);
			} else {
				return SimpleDataSet(
					N, lon, lat
				);
			}
		}
	};
	dataset_t data = make_data();

	return std::visit(
		[=](auto&& data_) -> int
		{
			if (wrap_plane != 0){
				switch (precision)
				{
				case 0:
					cost_batch<true, double>(
					    data_, pnorm, k0_ap, sigma_k0, proot,
					    logarithmic > 0u, M, lonc, lat0, alpha,
					    k0, f, result
					);
					break;
				case 1:
					cost_batch<true, long double>(
					    data_, pnorm, k0_ap, sigma_k0, proot,
					    logarithmic > 0u, M, lonc, lat0, alpha,
					    k0, f, result
					);
					break;
				default:
					std::cerr << "Unknown 'precision' parameter passed to "
					    "compute_cost_hotine_batch. Needs to be 0 (double) or "
					    "1 (long double).\n";
					return 1;
				}
			} else {
				switch (precision)
				{
				case 0:
					cost_batch<false, double>(
					    data_, pnorm, k0_ap, sigma_k0, proot,
					    logarithmic > 0u, M, lonc, lat0, alpha,
					    k0, f, result
					);
					break;
				case 1:
					cost_batch<false, long double>(
					    data_, pnorm, k0_ap, sigma_k0, proot,
					    logarithmic > 0u, M, lonc, lat0, alpha,
					    k0, f, result
					);
					break;
				default:
					std::cerr << "Unknown 'precision' parameter passed to "
					    "compute_cost_hotine_batch. Needs to be 0 (double) or "
					    "1 (long double).\n";
					return 1;
				}
			}
			return 0;
		},
		data
	);
}


template<bool wrap_plane, typename T>
void cost_gradient_batch(
    const T& data, double pnorm, double k0_ap, double sigma_k0,
    unsigned short proot, bool logarithmic, const size_t M,
    const double* lonc, const double* lat0, const double* alpha,
    const double* k0, double f, double* result
)
{
	/* Generate the cost function: */
	typedef long double real_t;
	typedef autodouble<4, real_t> real4v;
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

	/* Compute the Hotine Mercator projections now: */
	#pragma omp parallel for
	for (size_t i=0; i<M; ++i){
		/* Wrap or don't wrap the parameters, as requested: */
		std::optional<HotineObliqueMercator<real4v>> hom;
		if constexpr (wrap_plane){
			hom.emplace(
				HotineParameters<real4v>(
					variable4<0>(deg2rad(static_cast<real_t>(lonc[i]))),
					variable4<1>(deg2rad(static_cast<real_t>(lat0[i]))),
					variable4<2>(deg2rad(static_cast<real_t>(alpha[i]))),
					variable4<3>(static_cast<real_t>(k0[i]))
				),
				f
			);
		} else {
			hom.emplace(
				variable4<0>(static_cast<real_t>(deg2rad(lonc[i]))),
				variable4<1>(static_cast<real_t>(deg2rad(lat0[i]))),
				variable4<2>(static_cast<real_t>(deg2rad(alpha[i]))),
				variable4<3>(static_cast<real_t>(k0[i])),
				f
			);
		}

		/* Compute the cost and gradient */
		std::array<long double,4> grad = std::visit(
			[&hom, &data](auto&& cfun) -> std::array<long double,4>
			{
				return cfun(data, *hom).grad();
			},
			cost_function
		);
		result[4*i] = deg2rad(grad[0]);
		result[4*i+1] = deg2rad(grad[1]);
		result[4*i+2] = deg2rad(grad[2]);
		result[4*i+3] = grad[3];

	}
}


int compute_cost_gradient_hotine_batch(const size_t N, const double* lon,
        const double* lat, const double* h, const double* w, const size_t M,
        const double* lonc, const double* lat0, const double* alpha,
        const double* k0, double a, double f, double pnorm, double k0_ap,
        double sigma_k0, unsigned short proot,
        unsigned short logarithmic, unsigned short wrap_plane,
        double* result)
{
	/*
	 * Compute the cost function for many different HOM parameters
	 * given a single data set.
	 */
	typedef std::variant<WeightedDataSetWithHeight,
	                     WeightedDataSet,
	                     DataSetWithHeight,
	                     SimpleDataSet>
	    dataset_t;
	auto make_data = [=]() -> dataset_t
	{
		if (w){
			if (h){
				return WeightedDataSetWithHeight(
					N, lon, lat, h, w, a, f
				);
			} else {
				return WeightedDataSet(
					N, lon, lat, w
				);
			}
		} else {
			if (h){
				return DataSetWithHeight(
					N, lon, lat, h, a, f
				);
			} else {
				return SimpleDataSet(
					N, lon, lat
				);
			}
		}
	};
	dataset_t data = make_data();

	std::visit(
		[=](auto&& data_)
		{
			if (wrap_plane != 0){
				cost_gradient_batch<true>(data_, pnorm, k0_ap, sigma_k0, proot,
				    logarithmic > 0u, M, lonc, lat0, alpha, k0, f, result
				);
			} else {
				cost_gradient_batch<false>(data_, pnorm, k0_ap, sigma_k0, proot,
				    logarithmic > 0u, M, lonc, lat0, alpha, k0, f, result
				);
			}
		},
		data
	);

	return 0;
}


int compute_k_hotine(const size_t N, const double* lon,
        const double* lat, const double* w,
        double lonc, double lat0, double alpha, double k0, double f,
        double* result)
{
	/* Compute the Hotine Mercator projections now: */
	const HotineObliqueMercator<long double>
	   hom(deg2rad(lonc), deg2rad(lat0), deg2rad(alpha), k0, f);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		/* Compute the cost: */
		result[i] = hom.k(deg2rad(lon[i]), deg2rad(lat[i]));
	}

	return 0;
}


int hotine_project_uv(const size_t N, const double* lon,
        const double* lat, double lonc, double lat0, double alpha,
        double k0, double f, double* result)
{
	/* Compute the Hotine Mercator projections now: */
	const HotineObliqueMercator<long double>
	   hom(deg2rad(lonc),deg2rad(lat0), deg2rad(alpha), k0, f);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		/* Compute the cost: */
		HotineObliqueMercator<long double>::uv_t uv(hom.uv(deg2rad(lon[i]),
		                                                   deg2rad(lat[i])));
		result[2*i]   = uv.u;
		result[2*i+1] = uv.v;
	}

	return 0;
}


int hotine_project(const size_t N, const double* lon,
        const double* lat, double lonc, double lat0, double alpha,
        double k0, double gamma, double f, double* result)
{
	/* Compute the Hotine Mercator projections now: */
	const HotineObliqueMercatorProjection
	   hom(deg2rad(lonc),deg2rad(lat0), deg2rad(alpha), k0, deg2rad(gamma), f);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		/* Compute the cost: */
		HotineObliqueMercatorProjection::xy_t
		    xy(hom.project(deg2rad(lon[i]), deg2rad(lat[i])));
		result[2*i]   = xy.x;
		result[2*i+1] = xy.y;
	}

	return 0;
}


int hotine_inverse(const size_t N, const double* x,
        const double* y, double lonc, double lat0, double alpha,
        double k0, double gamma, double f, double* result)
{
	/* Compute the Hotine Mercator projections now: */
	const HotineObliqueMercatorProjection
	   hom(deg2rad(lonc),deg2rad(lat0), deg2rad(alpha), k0, deg2rad(gamma), f);

	#pragma omp parallel for
	for (size_t i=0; i<N; ++i){
		/* Compute the cost: */
		HotineObliqueMercatorProjection::geo_t lp(hom.inverse(x[i], y[i]));
		result[2*i]   = rad2deg(lp.lambda);
		result[2*i+1] = rad2deg(lp.phi);
	}

	return 0;
}


template<>
double doomercat::to_double<double>(const double& t)
{
	return t;
}

template<>
double doomercat::to_double<long double>(const long double& t)
{
	return t;
}


int hotine_bfgs(const size_t N, const double* lon, const double* lat,
                const double* h, const double* w, double a, double f,
                double pnorm, double k0_ap, double sigma_k0, double lonc_0,
                double lat_0_0, double alpha_0, double k_0_0, unsigned int Nmax,
                unsigned short proot, double epsilon, double* result,
                unsigned int* n_steps, uint64_t* n_fun_eval)
{
	// Sanity check on weights (probably very much redundant):
	if (w == 0)
		w = nullptr;

	/* Result of the optimization goes here: */
	std::vector<doomercat::hotine_result_t> history;
	std::optional<size_t> function_evaluations = 0;

	/* Init the data set and optimize: */
	if (w){
		if (h){
			WeightedDataSetWithHeight data(N, lon, lat, h, w, a, f);

			/* Optimize: */
			if (std::isinf(pnorm)){
				history = bfgs_optimize_hotine_pinf(data, lonc_0, lat_0_0,
				                                    alpha_0, k_0_0, f, k0_ap,
				                                    sigma_k0, Nmax, epsilon,
				                                    function_evaluations);
			} else {
				history = bfgs_optimize_hotine(data, lonc_0, lat_0_0, alpha_0,
				                               k_0_0, f, pnorm, k0_ap, sigma_k0,
				                               Nmax, proot > 0u, epsilon,
				                               function_evaluations);
			}
		} else {
			WeightedDataSet data(N, lon, lat, w);

			/* Optimize: */
			if (std::isinf(pnorm)){
				history = bfgs_optimize_hotine_pinf(data, lonc_0, lat_0_0,
				                                    alpha_0, k_0_0, f, k0_ap,
				                                    sigma_k0, Nmax, epsilon,
				                                    function_evaluations);
			} else {
				history = bfgs_optimize_hotine(data, lonc_0, lat_0_0, alpha_0,
				                               k_0_0, f, pnorm, k0_ap, sigma_k0,
				                               Nmax, proot > 0u, epsilon,
				                               function_evaluations);
			}
		}
	} else {
		if (h){
			DataSetWithHeight data(N, lon, lat, h, a, f);

			/* Optimize: */
			if (std::isinf(pnorm)){
				history = bfgs_optimize_hotine_pinf(data, lonc_0, lat_0_0,
				                                    alpha_0, k_0_0, f, k0_ap,
				                                    sigma_k0, Nmax, epsilon,
				                                    function_evaluations);
			} else {
				history = bfgs_optimize_hotine(data, lonc_0, lat_0_0, alpha_0,
				                               k_0_0, f, pnorm, k0_ap, sigma_k0,
				                               Nmax, proot > 0u, epsilon,
				                               function_evaluations);
			}
		} else {
			SimpleDataSet data(N, lon, lat);

			/* Optimize: */
			if (std::isinf(pnorm)){
				history = bfgs_optimize_hotine_pinf(data, lonc_0, lat_0_0,
				                                    alpha_0, k_0_0, f, k0_ap,
				                                    sigma_k0, Nmax, epsilon,
				                                    function_evaluations);
			} else {
				history = bfgs_optimize_hotine(data, lonc_0, lat_0_0, alpha_0,
				                               k_0_0, f, pnorm, k0_ap, sigma_k0,
				                               Nmax, proot > 0u, epsilon,
				                               function_evaluations);
			}
		}
	}

	/* Return the results: */
	for (size_t i=0; i<history.size(); ++i){
		result[11*i]    = history[i].cost;
		result[11*i+1]  = history[i].lonc;
		result[11*i+2]  = history[i].lat_0;
		result[11*i+3]  = history[i].alpha;
		result[11*i+4]  = history[i].k0;
		result[11*i+5]  = history[i].grad_lonc;
		result[11*i+6]  = history[i].grad_lat0;
		result[11*i+7]  = history[i].grad_alpha;
		result[11*i+8]  = history[i].grad_k0;
		result[11*i+9]  = history[i].algorithm_state;
		result[11*i+10] = history[i].step;
	}

	if (n_fun_eval){
		*n_fun_eval = *function_evaluations;
	}

	if (n_steps){
		*n_steps = static_cast<unsigned int>(history.size());
	}

	return 0;
}


int hotine_backtrack_GD(
    const size_t N, const double* lon, const double* lat,
    const double* h, const double* w, double a, double f,
    double pnorm, double k0_ap, double sigma_k0, double lonc_0,
    double lat_0_0, double alpha_0, double k_0_0, unsigned int Nmax,
    unsigned short proot, double epsilon, double* result,
    unsigned int* n_steps, unsigned long* n_fun_eval)
{
	// Sanity check on weights (probably very much redundant):
	if (w == 0)
		w = nullptr;

	/* Init the data set and optimize: */
	typedef std::variant<WeightedDataSetWithHeight,
	                     WeightedDataSet,
				         DataSetWithHeight,
				         SimpleDataSet>
		dataset_t;
	auto make_data = [=]() -> dataset_t
	{
		if (w){
			if (h){
				return WeightedDataSetWithHeight(
					N, lon, lat, h, w, a, f
				);
			} else {
				return WeightedDataSet(
					N, lon, lat, w
				);
			}
		} else {
			if (h){
				return DataSetWithHeight(
					N, lon, lat, h, a, f
				);
			} else {
				return SimpleDataSet(
					N, lon, lat
				);
			}
		}
	};
	dataset_t data = make_data();

	/* Result of the optimization goes here: */
	std::optional<size_t> function_evaluations = 0;
	std::vector<doomercat::hotine_result_t>
	history = std::visit(
		[=,&function_evaluations](auto&& data_)
		-> std::vector<doomercat::hotine_result_t>
		{
			return backtrack_GD_optimize_hotine(
				data_, lonc_0, lat_0_0, alpha_0,
				k_0_0, f, pnorm, k0_ap, sigma_k0,
				Nmax, epsilon, function_evaluations);
		},
		data
	);

	/* Return the results: */
	for (size_t i=0; i<history.size(); ++i){
		result[11*i]    = history[i].cost;
		result[11*i+1]  = history[i].lonc;
		result[11*i+2]  = history[i].lat_0;
		result[11*i+3]  = history[i].alpha;
		result[11*i+4]  = history[i].k0;
		result[11*i+5]  = history[i].grad_lonc;
		result[11*i+6]  = history[i].grad_lat0;
		result[11*i+7]  = history[i].grad_alpha;
		result[11*i+8]  = history[i].grad_k0;
		result[11*i+9]  = history[i].algorithm_state;
		result[11*i+10] = history[i].step;
	}

	if (n_fun_eval){
		*n_fun_eval = *function_evaluations;
	}

	if (n_steps){
		*n_steps = static_cast<unsigned int>(history.size());
	}

	return 0;
}


int hotine_parameters_debug(double lonc, double lat0, double alpha,
                            double k0, double f, double* result)
{
	/* Return parameters E, gamma0, lambda0 for debugging purposes. */
	HotineObliqueMercator<autodouble<4,long double>>
	    hom(constant4(deg2rad(static_cast<long double>(lonc))),
	        constant4(deg2rad(static_cast<long double>(lat0))),
	        constant4(deg2rad(static_cast<long double>(alpha))),
	        constant4(static_cast<long double>(k0)),
	        f
	);

	result[0] = hom.E().value();
	result[1] = hom.gamma0().value();
	result[2] = hom.lambda0().value();

	return 0;
}
