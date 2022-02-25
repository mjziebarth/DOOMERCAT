/*
 * Automatically differentiating double.
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

#include <cmath>
#include <array>
#include <stdexcept>
#include <functional>
#include <type_traits>

#include <iostream>

#ifndef AUTODOUBLE_H
#define AUTODOUBLE_H

typedef unsigned short dim_t;

template<dim_t d>
class autodouble {
public:
	autodouble(double x, std::array<double,d> dx) : x(x), deriv(dx)
	{};

	static autodouble constant(double x);

	template<dim_t i>
	static autodouble variable(double x);

	double value() const;

	double derivative(dim_t dimension) const;

	autodouble operator-() const;

	void invert_sign();

	autodouble operator+(const autodouble<d>& other) const;
	autodouble operator-(const autodouble<d>& other) const;
	autodouble operator*(const autodouble<d>& other) const;
	autodouble operator/(const autodouble<d>& other) const;

	autodouble operator+(const double c) const;
	autodouble operator-(const double c) const;
	autodouble operator*(const double c) const;
	autodouble operator/(const double c) const;

	autodouble& operator+=(const autodouble& other);
	autodouble& operator-=(const autodouble& other);
	autodouble& operator*=(const autodouble& other);
	autodouble& operator+=(double c);
	autodouble& operator*=(double c);
	autodouble& operator-=(double c);

	bool operator>=(const autodouble& other) const;
	bool operator<=(const autodouble& other) const;
	bool operator>=(double) const;
	bool operator<=(double) const;
	bool operator>(double) const;
	bool operator<(double) const;

	static autodouble div(const double c, const autodouble<d>& x);
	static autodouble minus(const double c, const autodouble<d>& x);

	static autodouble exp(const autodouble& x);
	static autodouble exp(autodouble&& x);
	static autodouble log(const autodouble& x);
	static autodouble log_move(autodouble&& x);
	static autodouble sqrt(const autodouble& x);
	static autodouble sqrt(autodouble&& x);
	static autodouble pow(const autodouble& x, const autodouble& a);
	static autodouble pow(const autodouble& x, const double a);
	static autodouble pow(const double x, const autodouble& a);
	static autodouble pow(autodouble&& x, const double a);

	static autodouble abs(autodouble&& x);

	static autodouble sin(const autodouble& x);
	static autodouble sin(autodouble&& x);
	static autodouble cos(const autodouble& x);
	static autodouble cos(autodouble&& x);
	static autodouble tan(const autodouble& x);
	static autodouble asin(const autodouble& x);
	static autodouble acos(const autodouble& x);
	static autodouble atan(const autodouble& x);
	static autodouble atan(autodouble&& x);
	static autodouble atan2(const autodouble& y, const autodouble& x);
	static autodouble atan2(autodouble&& y, const autodouble& x);
	static autodouble atanh(const autodouble& x);
	static autodouble atanh_move(autodouble&& x);

	static autodouble min(const autodouble& x, const autodouble& y);
	static autodouble max(const autodouble& x, const autodouble& y);
	//static autodouble max(autodouble&& x, autodouble&& y);

	static autodouble sum(const std::vector<autodouble>&);

private:
	double x;
	std::array<double,d> deriv;

	autodouble(double x) : x(x)
	{
		for (dim_t i=0; i<d; ++i){
			deriv[i] = 0.0;
		}
	};

	static autodouble nan();
};


/*
 * Create a variable:
 */
template<dim_t d>
autodouble<d> autodouble<d>::constant(double x)
{
	// First initialize the value and set all derivatives to 0:
	return autodouble<d>(x);
}

template<dim_t d>
template<dim_t i>
autodouble<d> autodouble<d>::variable(double x)
{
	static_assert(i < d, "Out-of-bounds variable access");
	// First initialize empty autodouble:
	autodouble<d> ad(x);

	// Seed the correct derivative:
	ad.deriv[i] = 1.0;

	return ad;
}

template<dim_t d>
autodouble<d> autodouble<d>::nan()
{
	// First initialize empty autodouble:
	constexpr double nan = std::nan("");
	autodouble<d> ad(nan);

	// Seed the correct derivative:
	for (dim_t i=0; i<d; ++i)
		ad.deriv[i] = nan;

	return ad;
}



/*
 * Value access:
 */
template<dim_t d>
double autodouble<d>::derivative(dim_t dimension) const
{
	if (dimension > d)
		throw std::runtime_error("Try to access out-of-bound derivative");

	return deriv[dimension];
}


/*
 * Operators:
 */
template<dim_t d>
double autodouble<d>::value() const
{
	return x;
}

template<dim_t d>
autodouble<d> autodouble<d>::operator-() const
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = -deriv[i];
	return autodouble(-x, deriv_new);
}

template<dim_t d>
void autodouble<d>::invert_sign()
{
	x = -x;
	for (dim_t i=0; i<d; ++i)
		deriv[i] = -deriv[i];
}


/*
 * Addition:
 */

template<dim_t d>
autodouble<d> autodouble<d>::operator+(const autodouble<d>& other) const
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = deriv[i] + other.deriv[i];

	return autodouble(x + other.x, deriv_new);
}

/*template<dim_t d>
autodouble<d> operator+(const autodouble<d>& x, const autodouble<d>& y)
{
	autodouble<d> z(x);
	z += y;
	return z;
}*/

template<dim_t d>
autodouble<d> operator+(autodouble<d>&& x, const autodouble<d>& y)
{
	x += y;
	return x;
}

template<dim_t d>
autodouble<d> operator+(const autodouble<d>& x, autodouble<d>&& y)
{
	y += x;
	return y;
}

template<dim_t d>
autodouble<d> operator+(autodouble<d>&& x, autodouble<d>&& y)
{
	x += y;
	return x;
}



/*
 * Subtraction:
 */


template<dim_t d>
autodouble<d> autodouble<d>::operator-(const autodouble<d>& other) const
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = deriv[i] - other.deriv[i];

	return autodouble(x - other.x, deriv_new);
}

/*template<dim_t d>
autodouble<d> operator-(const autodouble<d>& x, const autodouble<d>& y)
{
	autodouble<d> z(x);
	z -= y;
	return z;
}*/

template<dim_t d>
autodouble<d> operator-(autodouble<d>&& x, const autodouble<d>& y)
{
	x -= y;
	return x;
}

template<dim_t d>
autodouble<d> operator-(autodouble<d>&& x, autodouble<d>&& y)
{
	x -= y;
	return x;
}

template<dim_t d>
autodouble<d> operator-(const autodouble<d>& x, autodouble<d>&& y)
{
	/* x-y = x + (-y) = (-y) + x */
	y.invert_sign();
	y += x;
	return y;
}


/*
 * Multiplication:
 */

template<dim_t d>
autodouble<d> autodouble<d>::operator*(const autodouble<d>& other) const
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = other.x * deriv[i] + x * other.deriv[i];

	return autodouble(x * other.x, deriv_new);
}

/*template<dim_t d>
autodouble<d> operator*(const autodouble<d>& x, const autodouble<d>& y)
{
	autodouble<d> z(x);
	z *= y;
	return z;
}*/

template<dim_t d>
autodouble<d> operator*(autodouble<d>&& x, const autodouble<d>& y)
{
	x *= y;
	return x;
}

template<dim_t d>
autodouble<d> operator*(const autodouble<d>& x, autodouble<d>&& y)
{
	y *= x;
	return y;
}

template<dim_t d>
autodouble<d> operator*(autodouble<d>&& x, autodouble<d>&& y)
{
	x *= y;
	return x;
}




template<dim_t d>
autodouble<d> autodouble<d>::operator/(const autodouble<d>& other) const
{
	std::array<double,d> deriv_new;
	const double denom = 1.0 / (other.x * other.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = denom * (other.x * deriv[i] - x * other.deriv[i]);

	return autodouble(x / other.x, deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::operator*(const double c) const
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = c * deriv[i];

	return autodouble(c*x, deriv_new);
}


template<dim_t d>
autodouble<d> autodouble<d>::operator+(const double c) const
{
	return autodouble(x+c, deriv);
}

template<dim_t d>
autodouble<d> autodouble<d>::operator-(const double c) const
{
	return autodouble(x-c, deriv);
}

template<dim_t d>
autodouble<d> autodouble<d>::operator/(const double c) const
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = deriv[i] / c;

	return autodouble(x/c, deriv_new);
}


/*
 * Inplace operators:
 */

template<dim_t d>
autodouble<d>& autodouble<d>::operator+=(const autodouble<d>& other)
{
	x += other.x;
	for (dim_t i=0; i<d; ++i)
		deriv[i] += other.deriv[i];

	return *this;
}

template<dim_t d>
autodouble<d>& autodouble<d>::operator+=(double c)
{
	x += c;
	return *this;
}


template<dim_t d>
autodouble<d>& autodouble<d>::operator-=(const autodouble<d>& other)
{
	x -= other.x;
	for (dim_t i=0; i<d; ++i)
		deriv[i] -= other.deriv[i];

	return *this;
}

template<dim_t d>
autodouble<d>& autodouble<d>::operator-=(double c)
{
	x -= c;
	return *this;
}

template<dim_t d>
autodouble<d>& autodouble<d>::operator*=(const autodouble<d>& other)
{
	for (dim_t i=0; i<d; ++i)
		deriv[i] = other.x * deriv[i] + x * other.deriv[i];

	x *= other.x;

	return *this;
}

template<dim_t d>
autodouble<d>& autodouble<d>::operator*=(double c)
{
	x *= c;
	for (dim_t i=0; i<d; ++i)
		deriv[i] *= c;

	return *this;
}


/*
 * Comparison operators:
 */
template<dim_t d>
bool autodouble<d>::operator>=(const autodouble& other) const
{
	return x >= other.x;
}

template<dim_t d>
bool autodouble<d>::operator<=(const autodouble& other) const
{
	return x <= other.x;
}

template<dim_t d>
bool autodouble<d>::operator>=(double y) const
{
	return x >= y;
}

template<dim_t d>
bool autodouble<d>::operator>(double y) const
{
	return x > y;
}

template<dim_t d>
bool autodouble<d>::operator<=(double y) const
{
	return x <= y;
}

template<dim_t d>
bool autodouble<d>::operator<(double y) const
{
	return x < y;
}






/*
 * Other operators:
 */
template<dim_t d>
autodouble<d> autodouble<d>::div(const double c, const autodouble<d>& x)
{

	std::array<double,d> deriv_new;
	const double f = - c / (x.x * x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble(c/x.x, deriv_new);
}

template<dim_t d>
autodouble<d> operator/(const double c, const autodouble<d>& x)
{
	return autodouble<d>::div(c, x);
}

template<dim_t d>
autodouble<d> operator*(const double c, const autodouble<d>& x)
{
	// Call autodouble's method
	return x * c;
}

template<dim_t d>
autodouble<d> operator*(const double c, autodouble<d>&& x)
{
	x *= c;
	return x;
}

template<dim_t d>
autodouble<d> operator+(const double c, const autodouble<d>& x)
{
	// Call autodouble's method
	return x + c;
}

template<dim_t d>
autodouble<d> operator+(const double c, autodouble<d>&& x)
{
	// Call autodouble's method
	x += c;
	return x;
}


template<dim_t d>
autodouble<d> autodouble<d>::minus(const double c, const autodouble<d>& x)
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = -x.deriv[i];

	return autodouble(c - x.x, deriv_new);
}

template<dim_t d>
autodouble<d> operator-(const double c, const autodouble<d>& x)
{
	// Call autodouble's method
	return autodouble<d>::minus(c, x);
}







/*
 * Trigonometric functions:
 */

template<dim_t d>
autodouble<d> autodouble<d>::sin(const autodouble& x)
{
	std::array<double,d> deriv_new;
	const double cos_x = std::cos(x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = cos_x * x.deriv[i];

	return autodouble(std::sin(x.x), deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::sin(autodouble&& x)
{
	const double cos_x = std::cos(x.x);
	x.x = std::sin(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= cos_x;

	return x;
}



template<dim_t d>
autodouble<d> autodouble<d>::cos(const autodouble& x)
{
	std::array<double,d> deriv_new;
	const double sin_x = std::sin(x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = - sin_x * x.deriv[i];

	return autodouble(std::cos(x.x), deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::cos(autodouble&& x)
{
	const double n_sin_x = -std::sin(x.x);
	x.x = std::cos(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= n_sin_x;

	return x;
}

template<dim_t d>
autodouble<d> autodouble<d>::tan(const autodouble& x)
{
	std::array<double,d> deriv_new;
	const double tan_x = std::tan(x.x);
	const double f = 1.0 + tan_x * tan_x;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d>(tan_x, deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::atan(const autodouble& x)
{
	/* d/dx atan(x) = 1/(1+x^2) */
	const double f = 1.0 / (1.0 + x.x * x.x);
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d>(std::atan(x.x), deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::atan(autodouble&& x)
{
	/* d/dx atan(x) = 1/(1+x^2) */
	const double f = 1.0 / (1.0 + x.x * x.x);
	x.x = std::atan(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] = f * x.deriv[i];

	return x;
}

template<dim_t d>
autodouble<d> autodouble<d>::atan2(const autodouble& y, const autodouble& x)
{
	/* d/dx atan(z) = 1/(1+z^2) */
	const double z = y.x / x.x;
	const double f = 1.0 / (1.0 + z * z);
	const double ix = 1.0 / x.x;
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * ix * (y.deriv[i] - y.x * ix * x.deriv[i]);

	return autodouble<d>(std::atan2(y.x, x.x), deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::atan2(autodouble&& y, const autodouble& x)
{
	/* d/dx atan(z) = 1/(1+z^2) */
	const double yx = y.x;
	const double z = yx / x.x;
	const double f = 1.0 / (1.0 + z * z);
	const double ix = 1.0 / x.x;
	y.x = std::atan2(y.x, x.x);
	for (dim_t i=0; i<d; ++i)
		y.deriv[i] = f * ix * (y.deriv[i] - yx * ix * x.deriv[i]);

	return y;
}

template<dim_t d>
autodouble<d> autodouble<d>::atanh(const autodouble& x)
{
	/* Domain check: */
	if (std::abs(x.x) >= 1.0)
		return autodouble<d>::nan();

	/* d/dx atanh(x) = 1/(1 - x^2) */
	const double f = 1.0 / (1.0 - x.x * x.x);
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d>(std::atanh(x.x), deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::atanh_move(autodouble&& x)
{
	/* Domain check: */
	if (std::abs(x.x) >= 1.0){
		x.x = std::nan("");
		for (dim_t i=0; i<d; ++i)
			x.deriv[i] = std::nan("");
	} else {
		/* d/dx atanh(x) = 1/(1 - x^2) */
		const double f = 1.0 / (1.0 - x.x * x.x);
		x.x = std::atanh(x.x);
		for (dim_t i=0; i<d; ++i)
			x.deriv[i] *= f;
	}

	return x;
}

template<dim_t d>
autodouble<d> autodouble<d>::asin(const autodouble& x)
{
	/* d/dx asin(x) = 1/sqrt(1-x^2) */
	const double f = 1.0 / std::sqrt(1.0 - x.x * x.x);
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d>(std::asin(x.x), deriv_new);
}

/*
 * Other functions
 */


template<dim_t d>
autodouble<d> autodouble<d>::exp(const autodouble& x)
{
	std::array<double,d> deriv_new;
	const double exp_x = std::exp(x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = exp_x * x.deriv[i];

	return autodouble(exp_x, deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::exp(autodouble&& x)
{
	x.x = std::exp(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= x.x;

	return x;
}


template<dim_t d>
autodouble<d> autodouble<d>::log(const autodouble& x)
{
	std::array<double,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = x.deriv[i] / x.x;

	return autodouble(std::log(x.x), deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::log_move(autodouble<d>&& x)
{
	const double ix = 1.0 / x.x;
	x.x = std::log(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= ix;

	return x;
}


template<dim_t d>
autodouble<d> autodouble<d>::sqrt(const autodouble& x)
{
	std::array<double,d> deriv_new;
	const double sqrt = std::sqrt(x.x);
	const double f = 0.5 / sqrt;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble(sqrt, deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::sqrt(autodouble&& x)
{
	x.x = std::sqrt(x.x);
	const double f = 0.5 / x.x;
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= f;

	return x;
}


template<dim_t d>
autodouble<d> autodouble<d>::pow(const autodouble& x, const autodouble& a)
{
	return autodouble<d>::exp(autodouble<d>::log(x) * a);
}


template<dim_t d>
autodouble<d> autodouble<d>::pow(const autodouble& x, const double a)
{
	std::array<double,d> deriv_new;
	const double f = a * std::pow(x.x, a-1.0);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble(std::pow(x.x, a), deriv_new);
}

template<dim_t d>
autodouble<d> autodouble<d>::pow(const double x, const autodouble& a)
{
	std::array<double,d> deriv_new;
	const double f0 = std::pow(x, a.x);
	const double f1 = std::log(x) * f0;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f1 * a.deriv[i];

	return autodouble(f0, deriv_new);
}


template<dim_t d>
autodouble<d> autodouble<d>::pow(autodouble&& x, const double a)
{
	double f = std::pow(x.x, a-1.0);
	x.x *= f; // x.x := x.x*pow(x.x, a-1) = pow(x.x,a)
	f *= a;
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= f;

	return x;
}


template<dim_t d>
autodouble<d> autodouble<d>::abs(autodouble&& x)
{
	if (x.x > 0)
		return x;
	if (x.x == 0){
		for (dim_t i=0; i<d; ++i)
			x.deriv[i] = 0.0;
	} else {
		x.x = -x.x;
		for (dim_t i=0; i<d; ++i)
			x.deriv[i] = -x.deriv[i];
	}
	return x;
}


template<dim_t d>
autodouble<d> autodouble<d>::min(const autodouble<d>& x, const autodouble<d>& y)
{
	if (x.x <= y.x)
		return x;
	return y;
}

template<dim_t d>
autodouble<d> autodouble<d>::max(const autodouble<d>& x, const autodouble<d>& y)
{
	if (x.x >= y.x)
		return x;
	return y;
}


// Make sure that Kahan summation is not killed by re-association:
#pragma GCC optimize("-fno-associative-math")
template<dim_t d>
autodouble<d> autodouble<d>::sum(const std::vector<autodouble<d>>& x)
{
	struct kahan_summand_t {
		long double sum = 0.0;
		long double comp = 0.0;
	};
	kahan_summand_t Sx;
	std::array<kahan_summand_t,d> Sdx;
	for (const autodouble<d>& xi : x){
		long double add = xi.x - Sx.comp;
		long double res = Sx.sum + add;
		Sx.comp = (res - Sx.sum) - add;
		Sx.sum = res;
		for (dim_t j=0; j<d; ++j){
			long double add = xi.deriv[j] - Sdx[j].comp;
			long double res = Sdx[j].sum + add;
			Sdx[j].comp = (res - Sdx[j].sum) - add;
			Sdx[j].sum = res;
		}
	}

	std::array<double,d> dx;
	for (dim_t j=0; j<d; ++j){
		dx[j] = static_cast<double>(Sdx[j].sum);
	}
	return autodouble<d>(static_cast<double>(Sx.sum), dx);
}


#endif // ifndef AUTODOUBLE
