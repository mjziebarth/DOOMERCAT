/*
 * Hotine oblique Mercator projection.
 */

#ifndef DOOMERCAT_HOTINE_HPP
#define DOOMERCAT_HOTINE_HPP

#include <../include/arithmetic.hpp>
#include <../include/dataset.hpp>
#include <../include/constants.hpp>
#include <cmath>

namespace doomercat {



template<typename T>
class HotineObliqueMercator {
public:
	HotineObliqueMercator(const T& lambda_c, const T& phi0, const T& alpha,
	                      const T& k0, double f);

	T k(double lambda, double phi) const;

	const T& k0() const;

private:
	constexpr static double EPS_LARGE_PHI = 1e-9;

	typedef Arithmetic<T> AR;
	const double e2;
	const double e;
	const T k0_;
	const T phi0;
	const T alpha;
	const T sin_phi0;
	const T B;
	const T A;
	const T t0;
	const T D;
	const T F;
	const T E;
	const T G;
	const T g0;
	const T cos_g0;
	const T sin_g0;
	const T l0;

	/* Computation routines for all of the constants: */
	static T compute_B(const T& phi0, double e2);
	static T compute_A(const T& phi0, const T& k0, const T& sin_phi0,
	                   const T& B, double e2);
	static T compute_t0(const T& phi0, const T& sin_phi0, double e);
	static T compute_D(const T& phi0, T&& cos_phi0, const T& sin_phi0,
	                   const T& B, double e2);
	static T compute_F(const T& D, const T& phi0);
	static T compute_E(const T& F, const T& t0, const T& B);
	static T compute_G(const T& F);
	static T compute_g0(const T& alpha_c, const T& D);
	static T compute_l0(const T& lambda_c, const T& G, const T& g0,
	                    const T& B);

	struct uv_t {
		T u;
		T v;
	};

	uv_t uv(double lambda, double phi) const;

};


template<typename T>
HotineObliqueMercator<T>::HotineObliqueMercator(const T& lambda_c,
                              const T& phi0, const T& alpha, const T& k0,
                              double f)
   : e2(f*(2.0-f)), e(std::sqrt(e2)), k0_(k0), phi0(phi0), alpha(alpha),
     sin_phi0(AR::sin(phi0)), B(compute_B(phi0, e2)),
     A(compute_A(phi0, k0, sin_phi0, B, e2)),
     t0(compute_t0(phi0, sin_phi0, e)),
     D(compute_D(phi0, std::move(AR::cos(phi0)), sin_phi0, B, e2)),
     F(compute_F(D, phi0)), E(compute_E(F, t0, B)), G(compute_G(F)),
     g0(compute_g0(alpha, D)), cos_g0(AR::cos(g0)), sin_g0(AR::sin(g0)),
     l0(compute_l0(lambda_c, G, g0, B))
{
}


template<typename T>
T HotineObliqueMercator<T>::compute_B(const T& phi0, double e2)
{
	return AR::sqrt(1.0 + e2 * AR::pow(AR::cos(phi0),4) / (1.0 - e2));
}

template<typename T>
T HotineObliqueMercator<T>::compute_A(const T& phi0, const T& k0,
                                      const T& sin_phi0, const T& B,
                                      double e2)
{
	return B * k0 * std::sqrt(1.0 - e2)
	       / (1.0 - e2 * AR::pow2(sin_phi0));
}

template<typename T>
T HotineObliqueMercator<T>::compute_t0(const T& phi0, const T& sin_phi0,
                                       double e)
{
	return AR::tan(PI/4 - 0.5*phi0)
	     * AR::pow((1.0 + e * sin_phi0) / (1.0 - e * sin_phi0), 0.5*e);
}

template<typename T>
T HotineObliqueMercator<T>::compute_D(const T& phi0, T&& cos_phi0,
                                      const T& sin_phi0, const T& B,
                                      double e2)
{
	return AR::max(B * std::sqrt(1.0 - e2)
	               / (cos_phi0 * AR::sqrt(1.0 - e2 * sin_phi0 * sin_phi0)),
	               AR::constant(1.0));
}

template<typename T>
T HotineObliqueMercator<T>::compute_F(const T& D, const T& phi0)
{
	if (phi0 >= 0.0)
		return D + AR::sqrt(D*D - 1.0);
	return D - AR::sqrt(D*D - 1.0);
}

template<typename T>
T HotineObliqueMercator<T>::compute_E(const T& F, const T& t0, const T& B)
{
	return F * AR::pow(t0, B);
}

template<typename T>
T HotineObliqueMercator<T>::compute_G(const T& F)
{
	return 0.5*(F - 1.0 / F);
}

template<typename T>
T HotineObliqueMercator<T>::compute_g0(const T& alpha_c, const T& D)
{
	return AR::asin(AR::sin(alpha_c) / D);
}

template<typename T>
T HotineObliqueMercator<T>::compute_l0(const T& lambda_c, const T& G,
                                       const T& g0, const T& B)
{
	return lambda_c - AR::asin(G * AR::tan(g0)) / B;
}

template<typename T>
typename HotineObliqueMercator<T>::uv_t
HotineObliqueMercator<T>::uv(double lambda, double phi) const
{
	if (phi >  (1.0 - EPS_LARGE_PHI) * 0.5 * PI ||
	    phi < -(1.0 - EPS_LARGE_PHI) * 0.5 * PI){
		/* We are in the zone where the limiting approximation for
		 * phi = +/- pi/2 is better than the actual code (1e-9 stemming from
		 * some numerical investigations): */
		const T AoB(A/B);
		if (phi >= 0.0)
			return {AoB*phi, AoB * AR::log(AR::tan(PI/4 - 0.5*g0))};
		return {AoB*phi, AoB * AR::log(AR::tan(PI/4 + 0.5*g0))};
	}
	/* Can use the full equations. */
	const double sp = std::sin(phi);
	double t = std::sqrt((1.0 - sp) / (1.0 + sp)
	                     * std::pow((1.0 + e*sp) / (1.0 - e*sp), e));
	T Q(E * AR::pow(t, -B));
	T S(0.5*(Q - 1.0/Q));
	T T_(0.5*(Q + 1.0/Q));
	/* Delta lambda using the addition / subtraction rule of Snyder (p. 72) */
	T dlambda(lambda - l0);
	if (dlambda < -PI)
		dlambda += 2*PI;
	else if (dlambda > PI)
		dlambda -= 2*PI;
	T V(AR::sin(B*dlambda));
	T U((S * sin_g0 - V * cos_g0) / T_);

	T cBdl(AR::cos(B * dlambda));
	/* Case cos(B*(lambda - lambda_0)) == 0:
	 * Note: seems unproblematic.*/

	const T AoB(A/B);
	return {AoB * AR::atan2(S*cos_g0 + V*sin_g0, cBdl),
	        0.5 * AoB * AR::log((1.0-U)/(1.0 + U))};
}


template<typename T>
T HotineObliqueMercator<T>::k(double lambda, double phi) const
{
	/* Compute coordinate u: */
	T u(uv(lambda, phi).u);

	/* Compute dlambda: */
	T dlambda(lambda - l0);
	if (dlambda < -PI)
		dlambda += 2*PI;
	else if (dlambda > PI)
		dlambda -= 2*PI;

	/* Compute k: */
	double sp = std::sin(phi);
	return A * AR::cos(B*u/A) * std::sqrt(1.0 - e2 * sp * sp)
	       / (std::cos(phi) * AR::cos(B*(dlambda)));
}
template<typename T>
const T& HotineObliqueMercator<T>::k0() const
{
	return k0_;
}



} // namespace.

#endif
