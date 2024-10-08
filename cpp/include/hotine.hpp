/*
 * Hotine oblique Mercator projection.
 */

#ifndef DOOMERCAT_HOTINE_HPP
#define DOOMERCAT_HOTINE_HPP

#include <../include/arithmetic.hpp>
#include <../include/dataset.hpp>
#include <../include/constants.hpp>
#include <../include/parameters.hpp>
#include <cmath>
#include <numbers>

#ifdef DEBUG
#include <iostream>
#endif

namespace doomercat {


template<typename real>
struct hom_E_parabola_params_t {
	real C0;
	real a;
};

template<typename T>
class HOM_constants
{
public:
	typedef Arithmetic<T> AR;
	typedef typename AR::numeric_type number_t;

	static T compute_B(const T& cos_phi0, number_t e2);
	static T compute_A(const T& sin_phi0, const T& k0, const T& B, number_t e2);
	static T compute_t0(const T& phi0, const T& sin_phi0, number_t e);
	static T compute_D(const T& cos_phi0, const T& sin_phi0,
	                   const T& B, number_t e2);
	static T compute_F(const T& D, const T& phi0);
	static T compute_E(const T& F, const T& t0, const T& B);
	static T compute_G(const T& F);
	static T compute_g0(const T& alpha_c, const T& D);
	static T compute_l0(const T& lambda_c, const T& G, const T& g0,
	                    const T& B);

	/* Asymptotic computations: */

	/* G*sqrt(1-sin(phi0)) in the limit phi0 -> +/-90° */
	static T G_mul_sqx_pos(const T& sin_phi0, const T& xp, const T& xm,
	                       const T& sin_alpha, number_t e2);
	static T G_mul_sqx_neg(const T& sin_phi0, const T& xp, const T& xm,
	                       const T& sin_alpha, number_t e2);

	/* tan(g0)/sqrt(1 - sin(phi0)) in the limit phi0 -> +/-90° */
	static T tan_g0_div_sqx_asymptotic(const T& sin_phi0,
	                                   const T& sin_alpha,
	                                   number_t e2);
	/* */
	static T g0_asymptotic(const T& x, const T& sin_alpha, number_t e2);

	static hom_E_parabola_params_t<number_t> fit_E_parabola_pos(number_t e);
	static hom_E_parabola_params_t<number_t> fit_E_parabola_neg(number_t e);

	static T E_asymptotic_pos(number_t e, const T& phi0);
	static T E_asymptotic_neg(number_t e, const T& phi0);

private:
	constexpr static number_t ONE = 1.0;
};

template<typename T>
T HOM_constants<T>::compute_B(const T& cos_phi0, number_t e2)
{
	return AR::sqrt(ONE + e2 * AR::pow(cos_phi0,4) / (1.0 - e2));
}

template<typename T>
T HOM_constants<T>::compute_A(const T& sin_phi0, const T& k0, const T& B,
                              number_t e2)
{
	return B * k0 * std::sqrt(1.0 - e2) / (ONE - e2 * sin_phi0 * sin_phi0);
}

template<typename T>
T HOM_constants<T>::compute_t0(const T& phi0, const T& sin_phi0, number_t e)
{
	return AR::tan(static_cast<number_t>(PI/4) - (ONE/2)*phi0)
	     * AR::pow((ONE + e * sin_phi0) / (ONE - e * sin_phi0),
	               0.5*e);
}

template<typename T>
T HOM_constants<T>::compute_D(const T& cos_phi0, const T& sin_phi0,
                              const T& B, number_t e2)
{
	return AR::max(B * std::sqrt(1.0 - e2)
	               / (cos_phi0 * AR::sqrt(ONE - e2 * sin_phi0 * sin_phi0)),
	               AR::constant(1.0));
}

template<typename T>
T HOM_constants<T>::compute_F(const T& D, const T& phi0)
{
	if (phi0 >= 0.0)
		return D + AR::sqrt(D*D - ONE);
	return D - AR::sqrt(D*D - ONE);
}

template<typename T>
T HOM_constants<T>::compute_E(const T& F, const T& t0, const T& B)
{
	return F * AR::pow(t0, B);
}

template<typename T>
T HOM_constants<T>::compute_G(const T& F)
{
	return (ONE/2)*(F - ONE / F);
}

template<typename T>
T HOM_constants<T>::compute_g0(const T& alpha_c, const T& D)
{
	return AR::asin(AR::max(AR::min(AR::sin(alpha_c) / D, AR::constant(1.0)),
	                        AR::constant(-1.0)));
}

template<typename T>
T HOM_constants<T>::G_mul_sqx_pos(
    const T& sp0, const T& xp, const T& xm, const T& sa, number_t e2
)
{
	/* Computes G*sqrt(1-sin_phi0) */
	number_t Y = e2/(1.0 - e2);
	T W((1.0 - e2)*(Y*AR::pow2(xp*xm) + ONE)/(xp*(ONE - e2*sp0*sp0)));
	T Z(AR::sqrt(W) + AR::sqrt(W-xm));
	return (ONE/2) * (Z - xm/Z);
}

template<typename T>
T HOM_constants<T>::G_mul_sqx_neg(
    const T& sp0, const T& xp, const T& xm, const T& sa, number_t e2
)
{
	// sp0: sin_phi0
	/* Computes G*sqrt(1-sin_phi0) */
	number_t Y = e2/(1.0 - e2);
	T xpxm2 = AR::pow2(xp*xm);
	T W((1 - e2)*(Y*xpxm2 + ONE)/(xm*(ONE - e2*sp0*sp0)));
	T xpW(xp/W);
	// Series of Z = AR::sqrt(W) - AR::sqrt(W - xp),
	// evaluated with Horner's method:
	T Z(ONE/AR::sqrt(W)*(ONE / 2 + xpW*(ONE / 8 + ONE / 16 * xpW)));
	return (ONE/2) * (xp*Z - ONE/Z);
}

template<typename T>
T HOM_constants<T>::tan_g0_div_sqx_asymptotic(const T& x,
                                              const T& sa, number_t e2)
{
	/* Computes tan(g0)/sqrt(1-sin(phi0)) in the asymptotic sin(phi0) -> 1 */
	constexpr number_t SQ2 = std::sqrt(2.0l);
	T sa2(sa*sa);
	T sa4(sa2*sa2);
	number_t Y = e2/(1.0 - e2);

	return SQ2*sa * (ONE
	                 + x * ((Y - 0.25) + sa2
	                        + x * ((-Y*(11.0/4.0 + 0.5*Y) - 1.0/32.0)
	                               + (3*ONE/2) * sa4
	                               + (0.25*(15.0*e2 - 3.0)/(1.0-e2))*sa2
	                              )
	                           )
	                );
}


template<typename T>
T HOM_constants<T>::g0_asymptotic(const T& x,
                                  const T& sa, number_t e2)
{
	/* Computes g0 in the asymptotic sin(phi0) -> 1 */
	constexpr number_t SQ2 = std::sqrt(2.0l);
	T sa2(sa*sa);
	number_t Y = e2/(1.0 - e2);
	return SQ2*sa * (ONE
	                 + x * ((Y - 0.25)
	                        + sa2/(3*ONE)
	                        + x * (sa2*(Y - 0.25)
	                              + (-11.0/4.0 * Y - 0.5 * Y*Y - 1.0/32.0)
	                              + (3*ONE/10) * sa2*sa2
	                              )
	                        )
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
T HOM_constants<T>::E_asymptotic_neg(number_t e, const T& x)
{
	constexpr number_t ONE = 1.0;
	number_t C0 = std::pow((1 - e) / (1 + e), e/2);
	number_t e2 = e*e;
	number_t C1_div_C0 = 2 * e2 / (1 - e2);
	// Series expansion of
	// T x = 1 + std::sin(phi0)
	return C0 * (ONE + C1_div_C0 * x);
}

template<typename T>
T HOM_constants<T>::E_asymptotic_pos(number_t e, const T& x)
{
	constexpr number_t ONE = 1.0;
	number_t C0 = std::pow((1 + e) / (1 - e), e/2);
	number_t e2 = e*e;
	number_t C1_div_C0 = -2 * e2 / (1 - e2);
	// Series expansion of
	// T x = 1 + std::sin(phi0)
	return C0 * (ONE + C1_div_C0 * x);
}





template<typename T>
class HotineObliqueMercator {
	friend HotineObliqueMercator<long double>;
	friend HotineObliqueMercator<double>;
public:
	HotineObliqueMercator(const T& lambda_c, const T& phi0, const T& alpha,
	                      const T& k0, double f);

	HotineObliqueMercator(
	    const HotineParameters<T>& params, double f
	);

	template<typename T2>
	HotineObliqueMercator(const HotineObliqueMercator<T2>&);

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

protected:
	constexpr static double EPS_LARGE_PHI = 1e-9;

	typedef Arithmetic<T> AR;
	typedef typename AR::numeric_type number_t;

	const number_t e2;
	const number_t e;
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
double to_double(const T& t)
{
	return t.value();
}

template<>
double to_double<double>(const double& t);
template<>
double to_double<long double>(const long double& t);

template<typename T>
long double to_long_double(const T& t)
{
	return t.value();
}
template<>
long double to_long_double<double>(const double& t);
template<>
long double to_long_double<long double>(const long double& t);


template<typename T>
HotineObliqueMercator<T>::HotineObliqueMercator(const T& lambda_c,
                              const T& phi0, const T& alpha, const T& k0,
                              double f)
   : e2(f*(2.0l-f)), e(std::sqrt(e2)), k0_(k0), phi0(phi0), alpha(alpha)
{
	constexpr number_t ONE = 1.0;
	typedef HOM_constants<T> hom;
	T sin_phi0(AR::sin(phi0));
	T cos_phi0(AR::cos(phi0));
	B = hom::compute_B(cos_phi0, e2);
	A = hom::compute_A(sin_phi0, k0, B, e2);

	if (phi0 > deg2rad(89.9l)){
		/* More precise expansion of 1 - sin(phi0) at phi0 -> pi/2: */
		T dphi(std::numbers::pi_v<number_t>/2 - phi0);
		T dphi2(dphi * dphi);
		T xm(dphi2 * (ONE/2 + dphi2 * (-ONE/24 + dphi2/720)));
		T xp = static_cast<number_t>(2) - xm;

		E_ = hom::E_asymptotic_pos(e, xm);
		T sa = AR::sin(alpha);
		g0 = hom::g0_asymptotic(xm, sa, e2);

		l0 = lambda_c
		   - AR::asin(
		       AR::min(
		         AR::max(
		           hom::G_mul_sqx_pos(sin_phi0, xp, xm, sa, e2)
		              * hom::tan_g0_div_sqx_asymptotic(xm, sa, e2),
		           AR::constant(-1.0)),
		       AR::constant(1.0))
		     ) / B;

	} else if (phi0 < deg2rad(-89.9l)) {
		T dphi(phi0 + std::numbers::pi_v<number_t>/2);
		T dphi2(dphi * dphi);
		T xp(dphi2 * (ONE/2 + dphi2 * (-ONE/24 + dphi2/720)));
		T xm = static_cast<number_t>(2) - xp;
		E_ = hom::E_asymptotic_neg(e, xp);
		T sa = AR::sin(alpha);
		g0 = hom::g0_asymptotic(xp, sa, e2);

		l0 = lambda_c
		   - AR::asin(
		       AR::min(
		         AR::max(
		           hom::G_mul_sqx_neg(sin_phi0, xp, xm, sa, e2)
		              * hom::tan_g0_div_sqx_asymptotic(xp, sa, e2),
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

	#ifdef DEBUG
	std::cout << "HotineObliqueMercator("
		<< "k0=" << to_double(k0_) << ", "
		<< "phi0=" << to_double(phi0) << ", "
		<< "alpha=" << to_double(alpha) << ", "
		<< "B=" << to_double(B) << ", "
		<< "A=" << to_double(A) << ", "
		<< "E_=" << to_double(E_) << ", "
		<< "g0=" << to_double(g0) << ", "
		<< "cos_g0=" << to_double(cos_g0) << ", "
		<< "sin_g0=" << to_double(sin_g0) << ", "
		<< "l0=" << to_double(l0) << ")\n";
	#endif
}


template<typename T>
HotineObliqueMercator<T>::HotineObliqueMercator(
    const HotineParameters<T>& params, double f
) : HotineObliqueMercator(params[0], params[1], params[2], params[3], f)
{
}


template<typename T>
typename HotineObliqueMercator<T>::uv_t
HotineObliqueMercator<T>::uv(double lambda, double phi) const
{
	if (phi >  (1.0l - EPS_LARGE_PHI) * 0.5 * PI ||
	    phi < -(1.0l - EPS_LARGE_PHI) * 0.5 * PI){
		/* We are in the zone where the limiting approximation for
		 * phi = +/- pi/2 is better than the actual code (1e-9 stemming from
		 * some numerical investigations): */
		const T AoB(A/B);
		if (phi >= 0.0)
			return {AoB*phi, AoB * AR::log(AR::tan(PI/4 - 0.5*g0))};
		return {AoB*phi, AoB * AR::log(AR::tan(PI/4 + 0.5*g0))};
	}
	/* Can use the full equations. */
	const number_t sp = std::sin(static_cast<number_t>(phi));
	number_t t = std::sqrt((1.0 - sp) / (1.0 + sp)
	                       * std::pow((1.0 + e*sp) / (1.0 - e*sp), e));
	T Q(E_ * AR::pow(t, -B));
	T S(0.5*(Q - 1.0/Q));
	T T_(0.5*(Q + 1.0/Q));
	/* Delta lambda using the addition / subtraction rule of Snyder (p. 72) */
	T dlambda(static_cast<number_t>(lambda) - l0);
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
	if (phi >  (1.0l - EPS_LARGE_PHI) * 0.5 * PI ||
	    phi < -(1.0l - EPS_LARGE_PHI) * 0.5 * PI){
		/* We are in the zone where the limiting approximation for
		 * phi = +/- pi/2 is better than the actual code (1e-9 stemming from
		 * some numerical investigations): */
		return A/B*phi;
	}
	/* Can use the full equations. */
	const number_t sp = std::sin(static_cast<number_t>(phi));
	const number_t t = std::sqrt(
	    (1.0 - sp) / (1.0 + sp)
	    * std::pow((1.0 + e*sp) / (1.0 - e*sp), e)
	);
	T Q(E_ * AR::pow(t, -B));
	T S(static_cast<number_t>(0.5) * (Q - static_cast<number_t>(1.0)/Q));
	/* Delta lambda using the addition / subtraction rule of Snyder (p. 72) */
	T dlambda(static_cast<number_t>(lambda) - l0);
	if (dlambda < -PI)
		dlambda += 2*PI;
	else if (dlambda > PI)
		dlambda -= 2*PI;
	T V(AR::sin(B*dlambda));

	T cBdl(AR::cos(B * dlambda));
	/* Case cos(B*(lambda - lambda_0)) == 0:
	 * Note: this case causes the scale factor to diverge to infinity.
	 *       Since such a scale factor is against *all* optimization goals,
	 *       this case does not really have an important role in DOOMERCAT.
	 */
	if (cBdl == 0.0)
		return A * B * dlambda;

	return A / B * AR::atan2(S*cos_g0 + V*sin_g0, cBdl);
}


template<typename T>
T HotineObliqueMercator<T>::k(double lambda, double phi) const
{
	/* Compute dlambda: */
	T dlambda(static_cast<number_t>(lambda) - l0);
	if (dlambda < -PI)
		dlambda += 2*PI;
	else if (dlambda > PI)
		dlambda -= 2*PI;

	/* Compute k: */
	const number_t sp = std::sin(static_cast<number_t>(phi));
	return A * AR::cos(B*u(lambda, phi)/A) * std::sqrt(1.0 - e2 * sp * sp)
	       / (std::cos(static_cast<number_t>(phi)) * AR::cos(B*(dlambda)));
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


class HotineObliqueMercatorProjection : public HotineObliqueMercator<double> {
public:
	HotineObliqueMercatorProjection(double lambda_c, double phi0, double alpha,
	                                double k0, double gamma, double f);

	struct xy_t {
		double x;
		double y;
	};
	struct geo_t {
		double lambda;
		double phi;
	};

	/* Projection and inverse projection: */
	xy_t project(double lambda, double phi) const;
	geo_t inverse(double x, double y) const;

private:
	const double lambda_c;
	const double uc;
	const double cosg;
	const double sing;
};


} // namespace.

#endif
