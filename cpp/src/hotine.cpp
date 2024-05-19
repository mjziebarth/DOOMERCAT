/*
 * HotineObliqueMercator specialization.
 */

#include <../include/hotine.hpp>
#include <../include/linalg.hpp>

using doomercat::HotineObliqueMercator;
using doomercat::HotineObliqueMercatorProjection;
using doomercat::HOM_constants;

template<>
template<>
HotineObliqueMercator<long double>::HotineObliqueMercator(
	const HotineObliqueMercator<autodouble<4,long double>>& other
)
	: e2(other.e2), e(other.e), k0_(other.k0_.value()),
	  phi0(other.phi0.value()), alpha(other.alpha.value()),
	  B(other.B.value()), A(other.A.value()), E_(other.E_.value()),
	  g0(other.g0.value()), cos_g0(other.cos_g0.value()),
	  sin_g0(other.sin_g0.value()), l0(other.l0.value())
{
}

template<>
template<>
HotineObliqueMercator<double>::HotineObliqueMercator(
	const HotineObliqueMercator<autodouble<4,double>>& other
)
	: e2(other.e2), e(other.e), k0_(other.k0_.value()),
	  phi0(other.phi0.value()), alpha(other.alpha.value()),
	  B(other.B.value()), A(other.A.value()), E_(other.E_.value()),
	  g0(other.g0.value()), cos_g0(other.cos_g0.value()),
	  sin_g0(other.sin_g0.value()), l0(other.l0.value())
{
}


static double compute_uc(double A, double B, double alpha, double phi0,
                         double e2)
{
	typedef HOM_constants<double> hom;
	if (std::abs(alpha) == PI/2){
		return 0.0;
	} else {
		if (std::abs(phi0) > deg2rad(89.9l))
			throw std::runtime_error("Error computing central u: phi_0 out "
			                         "of range (too close to +/- pi/2).");
		const double D = hom::compute_D(std::cos(phi0), std::sin(phi0), B, e2);
		return A/B * std::atan2(std::sqrt(D*D - 1.0), std::cos(alpha))
		       * ((phi0 >= 0.0) ? 1.0 : -1.0);
	}
}


HotineObliqueMercatorProjection::HotineObliqueMercatorProjection(
                                     double lambda_c, double phi0, double alpha,
                                     double k0, double gamma, double f
)
    : HotineObliqueMercator<double>(lambda_c, phi0, alpha, k0, f),
      lambda_c(lambda_c),
      uc(compute_uc(A, B, alpha, phi0, e2)),
      cosg(std::cos(gamma)),
      sing(std::sin(gamma))
{
}


HotineObliqueMercatorProjection::xy_t
HotineObliqueMercatorProjection::project(double lambda, double phi) const
{
	/* Project to uv: */
	uv_t uv(this->uv(lambda, phi));

	/* Subtract the central u: */
	uv.u -= uc;

	/* Rotate: */
	xy_t xy({uv.v * cosg + uv.u * sing,
	         uv.u * cosg - uv.v * sing});

	return xy;
}

#include <iostream>

template<typename real>
static real modulo(real a, real b){
	/* True modulo operation (similar to Python's (a % b)).
	 * Implemented here only for positive b (which is what we use).
	 */
	real y = std::fmod(a,b);
	if (y < 0.0)
		return y+b;
	return y;
}

HotineObliqueMercatorProjection::geo_t
HotineObliqueMercatorProjection::inverse(double x, double y) const
{
	/* Inverse formulas might not always work.
	 * Use root finding instead. */

	/* Problem definition: */
	typedef linalg_t<2,double> lina;
	typedef typename lina::column_vectord_t vd_t;
	vd_t x0;
	x0[0] = lambda_c;
	x0[1] = phi0;

	auto compute_lambda_phi = [](const vd_t& lola) -> geo_t {
		if (lola[1] > 1e3*PI || lola[1] < -1e3*PI)
			std::cerr << "Warning: |phi| > 1e3 * pi!\n";
		geo_t geo;
		const int winding_number = (lola[1] >= 0.0) ?
		                              std::floor(std::abs(lola[1]+PI/2) / PI)
		                           : std::floor(std::abs(lola[1]-PI/2) / PI);
		const bool even = (winding_number % 2) == 0;
		geo.lambda = (even) ? modulo<double>(lola[0], 2*PI)
		                    : modulo<double>(-lola[0],2*PI);
		geo.phi = modulo(lola[1]+PI/2, PI) - PI/2;
		if (!even)
			geo.phi = -geo.phi;
		return geo;
	};

	auto cost = [&](const vd_t& lola) -> double {
		const geo_t geo(compute_lambda_phi(lola));
		const xy_t xy(project(geo.lambda, geo.phi));
		const double dx = (xy.x-x);
		const double dy = (xy.y-y);
		return dx * dx + dy * dy;
	};

	auto gradient = [&](const vd_t& x) -> vd_t {
		/* Numerical gradient. */
		constexpr double delta = 1e-5;
		const vd_t dx({delta, 0.0});
		const vd_t dy({0.0, delta});
		vd_t g;
		/* Fourth-order accurate symmetric difference: */
		g[0] = 1.0/(12*delta) * (-cost(x+2*dx) + 8*cost(x+dx)
		                         - 8*cost(x-dx) + cost(x-2*dx));
		g[1] = 1.0/(12*delta) * (-cost(x+2*dy) + 8*cost(x+dy)
		                         - 8*cost(x-dy) + cost(x-2*dy));
		return g;
	};


	double step = 0.5;
	double cost_i = cost(x0);
	int i;
	for (i=0; i<100000; ++i){
		if (cost_i < 1e-30){
			break;
		}

		/* Compute gradient: */
		vd_t g(gradient(x0));

		/* Make sure that we choose a step size that reduces the cost: */
		double cost_n0 = cost(x0 - step*g);
		while (cost_n0 > cost_i && lina::norm(step*g) > 1e-16){
			step *= 0.5;
			cost_n0 = cost(x0 - step*g);
		}

		/* Compare costs for different variations of step size: */
		const double cost_n1 = cost(x0 - 1.1*step*g);
		const double cost_n2 = cost(x0 - 0.9*step*g);
		if (cost_n1 < cost_n0){
			if (cost_n1 < cost_n2){
				cost_i = cost_n1;
				step *= 1.1;
			} else {
				cost_i = cost_n2;
				step *= 0.9;
			}
		} else if (cost_n2 < cost_n0){
			cost_i = cost_n1;
			step *= 0.9;
		} else {
			cost_i = cost_n0;
		}

		/* Proceed in x0: */
		x0 -= step*g;
	}

	/* Polish the coordinate results: */
	geo_t lp(compute_lambda_phi(x0));


	if (lp.lambda > PI)
		lp.lambda -= 2*PI;

	return lp;
}

template<>
long double doomercat::to_long_double<double>(const double& t)
{
	return t;
}
template<>
long double doomercat::to_long_double<long double>(const long double& t)
{
	return t;
}