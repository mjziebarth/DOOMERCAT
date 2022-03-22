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


struct hom_E_parabola_params_t {
	double C0;
	double a;
};

template<typename T>
class HOM_constants
{
public:
	static T compute_B(const T& cos_phi0, double e2);
	static T compute_A(const T& sin_phi0, const T& k0, const T& B, double e2);
	static T compute_t0(const T& phi0, const T& sin_phi0, double e);
	static T compute_D(const T& cos_phi0, const T& sin_phi0,
	                   const T& B, double e2);
	static T compute_F(const T& D, const T& phi0);
	static T compute_E(const T& F, const T& t0, const T& B);
	static T compute_G(const T& F);
	static T compute_g0(const T& alpha_c, const T& D);
	static T compute_l0(const T& lambda_c, const T& G, const T& g0,
	                    const T& B);

	/* Asymptotic computations: */

	/* G*sqrt(1-sin(phi0)) in the limit phi0 -> +/-90° */
	static T G_mul_sqx(const T& sin_phi0, const T& sin_alpha, double e2);
	static T G_mul_sqx_neg(const T& z,const T& sa, double e2);

	/* tan(g0)/sqrt(1 - sin(phi0)) in the limit phi0 -> +/-90° */
	static T tan_g0_div_sqx_asymptotic_pos(const T& sin_phi0,
	                                       const T& sin_alpha,
	                                       double e2);
	static T tan_g0_div_sqx_asymptotic_neg(const T& sin_phi0,
	                                       const T& sin_alpha,
	                                       double e2);
	/* */
	static T g0_asymptotic_pos(const T& sin_phi0, const T& sin_alpha,
	                           double e2);
	static T g0_asymptotic_neg(const T& sin_phi0, const T& sin_alpha,
	                           double e2);

	static hom_E_parabola_params_t fit_E_parabola_pos(double e);
	static hom_E_parabola_params_t fit_E_parabola_neg(double e);

private:
	typedef Arithmetic<T> AR;
};

template<typename T>
T HOM_constants<T>::compute_B(const T& cos_phi0, double e2)
{
	return AR::sqrt(1.0 + e2 * AR::pow(cos_phi0,4) / (1.0 - e2));
}

template<typename T>
T HOM_constants<T>::compute_A(const T& sin_phi0, const T& k0, const T& B,
                              double e2)
{
	return B * k0 * std::sqrt(1.0 - e2)
	       / (1.0 - e2 * AR::pow2(sin_phi0));
}

template<typename T>
T HOM_constants<T>::compute_t0(const T& phi0, const T& sin_phi0, double e)
{
	return AR::tan(PI/4 - 0.5*phi0)
	     * AR::pow((1.0 + e * sin_phi0) / (1.0 - e * sin_phi0),
	               0.5*e);
}

template<typename T>
T HOM_constants<T>::compute_D(const T& cos_phi0, const T& sin_phi0,
                              const T& B, double e2)
{
	return AR::max(B * std::sqrt(1.0 - e2)
	               / (cos_phi0 * AR::sqrt(1.0 - e2 * sin_phi0 * sin_phi0)),
	               AR::constant(1.0));
}

template<typename T>
T HOM_constants<T>::compute_F(const T& D, const T& phi0)
{
	if (phi0 >= 0.0)
		return D + AR::sqrt(D*D - 1.0);
	return D - AR::sqrt(D*D - 1.0);
}

template<typename T>
T HOM_constants<T>::compute_E(const T& F, const T& t0, const T& B)
{
	return F * AR::pow(t0, B);
}

template<typename T>
T HOM_constants<T>::compute_G(const T& F)
{
	return 0.5*(F - 1.0 / F);
}

template<typename T>
T HOM_constants<T>::compute_g0(const T& alpha_c, const T& D)
{
	return AR::asin(AR::max(AR::min(AR::sin(alpha_c) / D, AR::constant(1.0)),
	                        AR::constant(-1.0)));
}

template<typename T>
T HOM_constants<T>::G_mul_sqx(const T& z,const T& sa, double e2)
{
	/* Computes G*sqrt(1-sin_phi0) */
	T xp(1.0 + z);
	T xm(1.0 - z);
	double Y = e2/(1.0 - e2);
	T V((1.0 - e2)*(Y*AR::pow2(xp*xm) + 1.0)/(xp*(1.0 - e2*z*z)));
	T W(AR::sqrt(V) + AR::sqrt(V-xm));
	return 0.5 * (W - xm/W);
}

template<typename T>
T HOM_constants<T>::G_mul_sqx_neg(const T& z,const T& sa, double e2)
{
	/* Computes G*sqrt(1-sin_phi0) */
	T xp(1.0 + z);
	T xm(1.0 - z);
	double Y = e2/(1.0 - e2);
	T xpxm2 = AR::pow2(xp*xm);
	T V((1.0 - e2)*(Y*xpxm2 + 1.0)/(xm*(1.0 - e2*z*z)));
	T xpV(xp/V);
	// Series of W = AR::sqrt(V) - AR::sqrt(V - xp),
	// evaluated with Horner's method:
	T W(xp/AR::sqrt(V)*(0.5 + xpV*(0.125 + 0.0625*xpV)));
	return 0.5 * (W - xp/W);
}

template<typename T>
T HOM_constants<T>::tan_g0_div_sqx_asymptotic_pos(const T& sin_phi0,
                                                  const T& sa, double e2)
{
	/* Computes tan(g0)/sqrt(1-sin(phi0)) in the asymptotic sin(phi0) -> 1 */
	constexpr double SQ2 = std::sqrt(2.0);
	double e4 = e2*e2;
	T sa2(sa*sa);
	T sa4(sa2*sa2);
	T x(1.0 - sin_phi0);
	double Y = e2/(1.0 - e2);

	return SQ2*sa * (1.0 + x * (Y + sa2 - 0.25
	                            + x * (-Y*(11.0/4.0 + 0.5*Y) - 1.0/32.0
	                                   + 1.5*sa4
	                                   + 0.25*(15.0*e2 - 3.0)/(1.0-e2)*sa2
	                                   )
	                           )
	                );
}

template<typename T>
T HOM_constants<T>::tan_g0_div_sqx_asymptotic_neg(const T& sin_phi0,
                                                  const T& sa, double e2)
{
	/* Computes tan(g0)/sqrt(1-sin(phi0)) in the asymptotic sin(phi0) -> 1 */
	constexpr double SQ2 = std::sqrt(2.0);
	double e4 = e2*e2;
	T sa2(sa*sa);
	T sa4(sa2*sa2);
	T x(1.0 + sin_phi0);
	double Y = e2/(1.0 - e2);

	return SQ2*sa * (1.0 + x * (Y + sa2 - 0.25
	                            + x * (-Y*(11.0/4.0 + 0.5*Y) - 1.0/32.0
	                                   + 1.5*sa4
	                                   + 0.25*(15.0*e2 - 3.0)/(1.0-e2)*sa2
	                                   )
	                           )
	                );
}


template<typename T>
T HOM_constants<T>::g0_asymptotic_pos(const T& sin_phi0,
                                      const T& sa, double e2)
{
	/* Computes g0 in the asymptotic sin(phi0) -> 1 */
	constexpr double SQ2 = std::sqrt(2.0);
	double e4 = e2*e2;
	T sa2(sa*sa);
	T x(1.0 - sin_phi0);
	double Y = e2/(1.0 - e2);
	return SQ2*sa * (1.0 + x * (Y + sa2/3.0 - 0.25
	                            +  x * (sa2*(Y-0.25) - 11.0/4.0 * Y + 0.5*Y*Y
	                                    + 0.3*sa2*sa2 - 1.0/32.0)
	                           )
	                ) * AR::sqrt(x);
}

template<typename T>
T HOM_constants<T>::g0_asymptotic_neg(const T& sin_phi0,
                                      const T& sa, double e2)
{
	/* Computes g0 in the asymptotic sin(phi0) -> -1 */
	constexpr double SQ2 = std::sqrt(2.0);
	double e4 = e2*e2;
	T sa2(sa*sa);
	T x(1.0 + sin_phi0);
	double Y = e2/(1.0 - e2);
	return SQ2*sa * (1.0 + x  * ( Y - 0.25 + sa2/3.0
	                             + x * (sa2*(Y - 0.25) - 11.0/4.0 * Y
	                                    - 0.5*Y*Y - 1.0/32.0 + 0.3*sa2*sa2))
	                ) * AR::sqrt(x);
}



template<typename T>
T HOM_constants<T>::compute_l0(const T& lambda_c, const T& G, const T& g0,
                               const T& B)
{
	return lambda_c - AR::asin(AR::max(AR::min(G * AR::tan(g0),
	                                           AR::constant(1.0)),
	                                   AR::constant(-1.0))) / B;
}

template<typename T>
hom_E_parabola_params_t HOM_constants<T>::fit_E_parabola_pos(double e)
{
	/* Computes the parabola that estimates E for z -> 1: */
	double a = 0.0;
	double e2 = e*e;
	double C0 = std::pow((1.0 + e) / (1.0 - e), 0.5*e);
	for (int i=0; i<8; ++i){
		double phi_i = deg2rad(89.0 + 0.1*i);
		double cp = std::cos(phi_i);
		double B = HOM_constants<double>::compute_B(cp, e2);
		double sp = std::sin(phi_i);
		double t0 = HOM_constants<double>::compute_t0(phi_i, sp, e);
		double D = HOM_constants<double>::compute_D(cp, sp, B, e2);
		double F = HOM_constants<double>::compute_F(D, phi_i);
		double yi = C0 - HOM_constants<double>::compute_E(F, t0, B);
		double xi = 0.5*PI - phi_i;
		a += yi/(xi*xi);
	}
	return {C0, a / 8};
}

template<typename T>
hom_E_parabola_params_t HOM_constants<T>::fit_E_parabola_neg(double e)
{
	/* Computes the parabola that estimates E for z -> 1: */
	double a = 0.0;
	double e2 = e*e;
	double C0 = std::pow((1.0 - e) / (1.0 + e), 0.5*e);
	for (int i=0; i<8; ++i){
		double phi_i = deg2rad(-89.0 - 0.1*i);
		double cp = std::cos(phi_i);
		double B = HOM_constants<double>::compute_B(cp, e2);
		double sp = std::sin(phi_i);
		double t0 = HOM_constants<double>::compute_t0(phi_i, sp, e);
		double D = HOM_constants<double>::compute_D(cp, sp, B, e2);
		double F = HOM_constants<double>::compute_F(D, phi_i);
		double yi = HOM_constants<double>::compute_E(F, t0, B) - C0;
		double xi = 0.5*PI + phi_i;
		a += yi/(xi*xi);
	}
	return {C0, a / 8};
}





template<typename T>
class HotineObliqueMercator {
public:
	HotineObliqueMercator(const T& lambda_c, const T& phi0, const T& alpha,
	                      const T& k0, double f);

	T k(double lambda, double phi) const;

	const T& k0() const;

	struct uv_t {
		T u;
		T v;
	};

	uv_t uv(double lambda, double phi) const;
	T u(double lambda, double phi) const;


	/* Mostly for debugging info: */
	const T& E() const;
	const T& gamma0() const;
	const T& lambda0() const;

private:
	constexpr static double EPS_LARGE_PHI = 1e-9;

	typedef Arithmetic<T> AR;
	const double e2;
	const double e;
	const T k0_;
	const T phi0;
	const T alpha;
	T B;
	T A;
	T E_;
	T g0;
	T cos_g0;
	T sin_g0;
	T l0;

	/* Computation routines for all of the constants: */
};






template<typename T>
double to_double(const T& t);

template<typename T>
HotineObliqueMercator<T>::HotineObliqueMercator(const T& lambda_c,
                              const T& phi0, const T& alpha, const T& k0,
                              double f)
   : e2(f*(2.0-f)), e(std::sqrt(e2)), k0_(k0), phi0(phi0), alpha(alpha)
{
	typedef HOM_constants<T> hom;
	T sin_phi0(AR::sin(phi0));
	T cos_phi0(AR::cos(phi0));
	B = hom::compute_B(cos_phi0, e2);
	A = hom::compute_A(sin_phi0, k0, B, e2);

	if (phi0 > deg2rad(89.9)){
		constexpr double SQ2 = std::sqrt(2);
		auto C0a = hom::fit_E_parabola_pos(e);
		E_ = C0a.C0 - C0a.a * (phi0 - PI/2) * (phi0 - PI/2);
		T sa = AR::sin(alpha);
		g0 = hom::g0_asymptotic_pos(sin_phi0, sa, e2);

		l0 = lambda_c
		   - AR::asin(
		       AR::min(
		         AR::max(
		           hom::G_mul_sqx(sin_phi0, sa, e2)
		              * hom::tan_g0_div_sqx_asymptotic_pos(sin_phi0, sa, e2),
		           AR::constant(-1.0)),
		       AR::constant(1.0))
		     ) / B;

	} else if (phi0 < deg2rad(-89.9)) {
		constexpr double SQ2 = std::sqrt(2);
		auto C0a = hom::fit_E_parabola_neg(e);
		E_ = C0a.C0 + C0a.a * (phi0 + PI/2) * (phi0 + PI/2);
		T sa = AR::sin(alpha);
		g0 = hom::g0_asymptotic_neg(sin_phi0, sa, e2);

		l0 = lambda_c
		   - AR::asin(
		       AR::min(
		         AR::max(
		           hom::G_mul_sqx_neg(sin_phi0, sa, e2)
		              * hom::tan_g0_div_sqx_asymptotic_neg(sin_phi0, sa, e2),
		           AR::constant(-1.0)),
		       AR::constant(1.0))
		     ) / B;

	} else {
		T t0(hom::compute_t0(phi0, sin_phi0, e));
		T D(hom::compute_D(cos_phi0, sin_phi0, B, e2));
		T F(hom::compute_F(D, phi0));
		E_ = hom::compute_E(F, t0, B);
		T G(hom::compute_G(F));
		g0 = hom::compute_g0(alpha, D);
		l0 = hom::compute_l0(lambda_c, G, g0, B);
	}
	cos_g0 = AR::cos(g0);
	sin_g0 = AR::sin(g0);
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
	T Q(E_ * AR::pow(t, -B));
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
T HotineObliqueMercator<T>::u(double lambda, double phi) const
{
	if (phi >  (1.0 - EPS_LARGE_PHI) * 0.5 * PI ||
	    phi < -(1.0 - EPS_LARGE_PHI) * 0.5 * PI){
		/* We are in the zone where the limiting approximation for
		 * phi = +/- pi/2 is better than the actual code (1e-9 stemming from
		 * some numerical investigations): */
		return A/B*phi;
	}
	/* Can use the full equations. */
	const double sp = std::sin(phi);
	double t = std::sqrt((1.0 - sp) / (1.0 + sp)
	                     * std::pow((1.0 + e*sp) / (1.0 - e*sp), e));
	T Q(E_ * AR::pow(t, -B));
	T S(0.5*(Q - 1.0/Q));
	T T_(0.5*(Q + 1.0/Q));
	/* Delta lambda using the addition / subtraction rule of Snyder (p. 72) */
	T dlambda(lambda - l0);
	if (dlambda < -PI)
		dlambda += 2*PI;
	else if (dlambda > PI)
		dlambda -= 2*PI;
	T V(AR::sin(B*dlambda));

	T cBdl(AR::cos(B * dlambda));
	/* Case cos(B*(lambda - lambda_0)) == 0:
	 * Note: seems unproblematic.*/

	return A / B * AR::atan2(S*cos_g0 + V*sin_g0, cBdl);
}


template<typename T>
T HotineObliqueMercator<T>::k(double lambda, double phi) const
{
	/* Compute dlambda: */
	T dlambda(lambda - l0);
	if (dlambda < -PI)
		dlambda += 2*PI;
	else if (dlambda > PI)
		dlambda -= 2*PI;

	/* Compute k: */
	double sp = std::sin(phi);
	return A * AR::cos(B*u(lambda, phi)/A) * std::sqrt(1.0 - e2 * sp * sp)
	       / (std::cos(phi) * AR::cos(B*(dlambda)));
}


template<typename T>
const T& HotineObliqueMercator<T>::k0() const
{
	return k0_;
}


template <typename T>
const T& HotineObliqueMercator<T>::E() const
{
	return E_;
}


template <typename T>
const T& HotineObliqueMercator<T>::gamma0() const
{
	return g0;
}


template <typename T>
const T& HotineObliqueMercator<T>::lambda0() const
{
	return l0;
}



} // namespace.

#endif
