/*
 * Pure compilation test source file.
 */

#include <../include/hotine.hpp>
#include <../include/functions.hpp>
#include <iostream>

using doomercat::HotineObliqueMercator;

int main(){
	real5v lambda_c(constant5(deg2rad(20.0)));
	real5v phi0(constant5(deg2rad(15.0)));
	real5v alpha(constant5(deg2rad(23.0)));
	real5v k0(constant5(0.99));
	double f = 1.0/298.;
	HotineObliqueMercator hom(lambda_c, phi0, alpha, k0, f);

	std::cout << "k(10,20): " << hom.k(deg2rad(10.0), deg2rad(20.0)).value()
	          << "\n";
}
