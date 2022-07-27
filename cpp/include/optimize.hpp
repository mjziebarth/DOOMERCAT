/*
 * Optimization routines.
 */

#include <memory>
#include <../include/cost_hotine.hpp>
#include <../include/hotine.hpp>
#include <../include/linalg.hpp>
#include <../include/bfgs.hpp>

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

template<typename DS>
std::vector<hotine_result_t>
bfgs_optimize_hotine(const DS& data, const double lonc0,
                     const double lat_00, const double alpha0,
                     const double k00, const double f,
                     const double pnorm, const double k0_ap,
                     const double sigma_k0,
                     const size_t Nmax, const bool proot, const double epsilon)
{
	/* Number of parameters used in the optimization: */
	constexpr size_t P = 4;
	static_assert(P < 256, "If P >= 256, the iteration types need to be "
	                       "increased.");
	typedef uint_fast8_t p_iter_t;

	/* Defining the linear algebra implementation: */
	typedef linalg_t<P,double> lina_t;
	typedef HotineObliqueMercator<real4v> hom_t;


	/* Initial config:
	 * Optimize with proot=true first to prevent instable descent
	 * to very low k_0: */
	CostFunctionHotine<real4v> cost_function(pnorm, k0_ap, sigma_k0, true,
	                                         true, true);
	CostHotine<real4v> cost(cost_function(data,
	                             hom_t(variable4<0>(deg2rad(lonc0)),
	                                   variable4<1>(deg2rad(lat_00)),
	                                   variable4<2>(deg2rad(alpha0)),
	                                   variable4<3>(k00), f)));

	auto fmod = [=](double a, double b) -> double {
		/* True modulo operation (similar to Python's (a % b)).
		 * Implemented here only for positive b (which is what we use).
		 */
		double y = std::fmod(a,b);
		if (y < 0.0)
			return y+b;
		return y;
	};

	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	std::array<double,P> last_x;
	auto check_cached_cost = [&](const std::array<double,P>& x) {
		bool x_new = false;
		for (p_iter_t i=0; i<P; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			const double lambda_c = fmod(x[0] + PI, 2*PI) - PI;
			const double alpha    = fmod(x[2] + PI/2, PI) - PI/2;
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
		        && x[3] > 0.0 && x[3] < 2.0);
	};

	auto propose_jump = [&](const std::array<double,P>& x,
	                        const std::array<double,P>& grad)
	      -> std::unique_ptr<std::array<double,P>>
	{
		if (   (x[1] > 89.0 * PI/180.0  && grad[1] < 0)
		    || (x[1] < -89.0 * PI/180.0 && grad[1] > 0))
		{
			double grad_scale = 1.1 * (0.5*PI - std::abs(x[1]))
			                    / std::abs(grad[1]);
			double lonc  = x[0] - grad_scale * grad[0] + PI;
			double lat0;
			if (x[1] > 0)
				lat0 = PI - (x[1] - grad_scale * grad[1]);
			else
				lat0 = -PI - (x[1] - grad_scale * grad[1]);
			double alpha = x[2] - grad_scale * grad[2];
			double k0    = x[3] - grad_scale * grad[3];
			std::unique_ptr<std::array<double,P>> res
			   = std::make_unique<std::array<double,P>>();
			(*res)[0] = lonc;
			(*res)[1] = lat0;
			(*res)[2] = alpha;
			(*res)[3] = k0;
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

	/* Preconditioning for gradient descent: */
	int angle_steps = 20;
	int k0_steps = 0;
	static_assert(P==4, "Adjust preconditioner.");
	auto preconditioner = [&](std::array<double,P>& grad)
	   -> bool
	{
		if (angle_steps > 0){
			grad[3] = 0.0;
			--angle_steps;
			if (angle_steps == 0){
				k0_steps = 20;
				return true;
			}
		} else {
			grad[0] = 0.0;
			grad[1] = 0.0;
			grad[2] = 0.0;
			grad[3] = std::min(std::max(-1.0, grad[3]), 1.0);
			--k0_steps;
			if (k0_steps == 0){
				angle_steps = 20;
				return true;
			}
		}

		return false;
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
	                                     propose_jump,
	                                     preconditioner);
	// Poor man's cache invalidation.
	for (p_iter_t i=0; i<P; ++i){
		last_x[i] *= 2;
	}
	x0 = y.history.back().parameters;

	/* Optimize without proot if wanted: */
	if (!proot){
		cost_function = CostFunctionHotine<real4v>(pnorm, k0_ap, sigma_k0,
		                                           proot, true, true);

		y2 = fallback_gradient_BFGS<P,lina_t>(x0, cost_lambda,
		                                      gradient_lambda_redux,
		                                      Nmax-y.history.size(),
		                                      epsilon,
		                                      in_boundary,
		                                      propose_jump,
		                                      preconditioner);
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
			const double lambda_c = fmod(z.parameters[0]+PI, 2*PI) - PI;

			/* Same for alpha: */
			const double alpha = fmod(z.parameters[2]+PI/2, PI) - PI/2;

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
				           static_cast<unsigned int>(z.mode),
				           z.step});
		}
	};

	append_history(y, true);
	append_history(y2, true);

	return res;
}



template<typename DS>
std::vector<hotine_result_t>
bfgs_optimize_hotine_pinf(const DS& data, const double lonc0,
                          const double lat_00, const double alpha0,
                          const double k00, const double f,
                          const double k0_ap, const double sigma_k0,
                          const size_t Nmax, const double epsilon)
{
	/* Number of parameters used in the optimization: */
	constexpr size_t P = 4;
	static_assert(P < 256, "If P >= 256, the iteration types need to be "
	                       "increased.");
	typedef uint_fast8_t p_iter_t;

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

	auto fmod = [=](double a, double b) -> double {
		/* True modulo operation (similar to Python's (a % b)).
		 * Implemented here only for positive b (which is what we use).
		 */
		double y = std::fmod(a,b);
		if (y < 0.0)
			return y+b;
		return y;
	};


	/*  A lambda function that checks whether the currently cached version
	 * of 'cost' is equal to a given one: */
	std::array<double,P> last_x;
	auto check_cached_cost = [&](const std::array<double,P>& x) {
		bool x_new = false;
		for (p_iter_t i=0; i<P; ++i){
			if (x[i] != last_x[i]){
				x_new = true;
				break;
			}
		}

		if (x_new){
			/* Recalculate. */
			const double lambda_c = fmod(x[0] + PI, 2*PI) - PI;
			const double alpha    = fmod(x[2] + PI/2, PI) - PI/2;
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

	auto propose_jump = [&](const std::array<double,P>& x,
	                        const std::array<double,P>& grad)
	      -> std::unique_ptr<std::array<double,P>>
	{
		if (   (x[1] > 89.0 * PI/180.0  && grad[1] < 0)
		    || (x[1] < -89.0 * PI/180.0 && grad[1] > 0))
		{
			double grad_scale = 1.1 * (0.5*PI - std::abs(x[1]))
			                    / std::abs(grad[1]);
			double lonc  = x[0] - grad_scale * grad[0] + PI;
			double lat0;
			if (x[1] > 0)
				lat0 = PI - (x[1] - grad_scale * grad[1]);
			else
				lat0 = -PI - (x[1] - grad_scale * grad[1]);
			double alpha = x[2] - grad_scale * grad[2];
			double k0    = x[3] - grad_scale * grad[3];
			std::unique_ptr<std::array<double,P>> res
			   = std::make_unique<std::array<double,P>>();
			(*res)[0] = lonc;
			(*res)[1] = lat0;
			(*res)[2] = alpha;
			(*res)[3] = k0;
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

	/* Preconditioning for gradient descent: */
	int angle_steps = 20;
	int k0_steps = 0;
	static_assert(P==4, "Adjust preconditioner.");
	auto preconditioner = [&](std::array<double,P>& grad)
	   -> bool
	{
		if (angle_steps > 0){
			grad[3] = 0.0;
			--angle_steps;
			if (angle_steps == 0){
				k0_steps = 20;
				return true;
			}
		} else {
			grad[0] = 0.0;
			grad[1] = 0.0;
			grad[2] = 0.0;
			grad[3] = std::min(std::max(-1.0, grad[3]), 1.0);
			--k0_steps;
			if (k0_steps == 0){
				angle_steps = 20;
				return true;
			}
		}
		/* No step reset required. */
		return false;
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
	                                     propose_jump,
	                                     preconditioner);
	// Poor man's cache invalidation.
	for (p_iter_t i=0; i<P; ++i){
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
			const double lambda_c = fmod(z.parameters[0]+PI, 2*PI) - PI;

			/* Same for alpha: */
			const double alpha = fmod(z.parameters[2]+PI/2, PI) - PI/2;

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
				           static_cast<unsigned int>(z.mode),
				           z.step});
		}
	};

	append_history(y, true);
	append_history(y2, false);

	return res;
}

}
#endif
