/*
 * HotineObliqueMercator specialization.
 */

#include <../include/hotine.hpp>

using doomercat::HotineObliqueMercator;

template<>
template<>
HotineObliqueMercator<double>::HotineObliqueMercator(
	const HotineObliqueMercator<real4v>& other
)
	: e2(other.e2), e(other.e), k0_(other.k0_.value()),
	  phi0(other.phi0.value()), alpha(other.alpha.value()),
	  B(other.B.value()), A(other.A.value()), E_(other.E_.value()),
	  g0(other.g0.value()), cos_g0(other.cos_g0.value()),
	  sin_g0(other.sin_g0.value()), l0(other.l0.value())
{
}
