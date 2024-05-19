/*
 * Shorthand function definitions for automatically differentiating double.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
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

#include <../include/autodouble.hpp>
#include <../include/constants.hpp>

#ifndef AUTODOUBLE_FUNCTIONS_H
#define AUTODOUBLE_FUNCTIONS_H

template<class T>
T cos(const T& x) {
	return T::cos(x);
}

template<class T>
T sin(const T& x) {
	return T::sin(x);
}

template<class T>
T tan(const T& x) {

}

template<class T>
T exp(const T& x) {
	return T::exp(x);
}

template<class T>
T exp(T&& x) {
	return T::exp(std::move(x));
}

template<class T>
T log(const T& x) {
	return T::log(x);
}

template<class T>
T log(T&& x) {
	return T::log_move(std::move(x));
}

template<class T>
T log(T& x) {
	return T::log(x);
}


template<class T>
T sqrt(const T& x) {
	return T::sqrt(x);
}

template<class T>
T sqrt(T&& x) {
	return T::sqrt(std::move(x));
}

template<class T>
T asin(const T& x) {
	return T::asin(x);
}

template<class T>
T acos(const T& x) {
	return T::acos(x);
}

template<class T>
T atan(const T& x) {
	return T::atan(x);
}

template<class T>
T atan(T&& x) {
	return T::atan(std::move(x));
}

template<class T>
T atan2(const T& y, const T& x) {
	return T::atan2(y,x);
}

template<class T>
T atanh(const T& x) {
	return T::atanh(x);
}

template<class T>
T atanh(T&& x) {
	return T::atanh_move(std::move(x));
}

template<class T>
T pow(const T& x, const T& a) {
	return T::pow(x, a);
}

template<class T>
T pow(const T& x, double a) {
	return T::pow(x,a);
}

template<class T>
T pow(T&& x, double a) {
	return T::pow(std::move(x),a);
}

template<class T>
T pow(double x, const T& a) {
	return T::pow(x,a);
}



template<class T>
T pow(const T& x, int i) {
	if (i < 1 || i > 5)
		return pow(x, static_cast<double>(i));
	if (i == 5)
		return x*x*x*x*x;
	else if (i == 4)
		return x*x*x*x;
	else if (i == 3)
		return x*x*x;
	else if (i == 2)
		return x*x;
	return x;
}

template<class T>
T pow(T&& x, int i) {
	if (i < 1 || i > 5)
		return pow(std::move(x), static_cast<double>(i));
	if (i == 5){
		const T tmp(x);
		x *= x;   // x = x^2
		x *= x;   // x = x^4
		x *= tmp; // x = x^5
	} else if (i == 4) {
		x *= x; // x = x^2
		x *= x; // x = x^4
	} else if (i == 3) {
		const T tmp(x);
		x *= x; // x = x^2
		x *= tmp; // x = x^2
	} else if (i == 2) {
		x *= x;
	}
	return x;
}

template<class T>
T pow2(T&& x) {
	x *= x;
	return x;
}

template<class T>
T min(const T& a, const T& b) {
	return T::min(a,b);
}

template<class T>
typename std::enable_if<!std::is_same<T,double>::value,T>::type
min(const T& a, double b)
{
	return T::min(a, T::constant(b));
}

template<class T>
T max(const T& a, const T& b) {
	return T::max(a,b);
}

template<class T>
typename std::enable_if<!std::is_same<T,double>::value,T>::type
max(const T& a, double b) {
	return T::max(a, T::constant(b));
}

// Declare double templated specializations:
template<>
double min(const double& a, const double& b);
template<>
double max(const double& a, const double& b);

// Constants and variables:
template<typename real>
autodouble<5,real> constant5(real x)
{
	return autodouble<5,real>::constant(x);
}

template<typename real>
autodouble<4,real> constant4(real x)
{
	return autodouble<4,real>::constant(x);
}

template<dim_t i, typename real>
autodouble<5,real> variable5(real x) {
	return autodouble<5,real>::template variable<i>(x);
}

template<dim_t i, typename real>
autodouble<4,real> variable4(real x) {
	return autodouble<4,real>::template variable<i>(x);
}

// Some math functions:
template<typename T>
T deg2rad(const T& deg) {
	constexpr long double d2r = PI / 180.0;
	return deg * d2r;
}

template<typename T>
T rad2deg(const T& rad) {
	constexpr long double r2d = 180.0 / PI;
	return rad * r2d;
}


#endif
