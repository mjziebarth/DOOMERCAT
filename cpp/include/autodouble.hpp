/*
 * Automatically differentiating double.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
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

#include <../include/sum.hpp>

#include <cmath>
#include <array>
#include <vector>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <stdint.h>

#ifndef AUTODOUBLE_H
#define AUTODOUBLE_H

typedef uint_fast8_t dim_t;

template<dim_t d, typename real>
class autodouble {
public:
	typedef real real_t;

	template<typename T2 = double>
	constexpr autodouble(
	    real x,
	    std::enable_if_t<!std::is_same_v<real,T2>, std::array<double,d>> dx
	) : x(x)
	{
		for (dim_t i=0; i<d; ++i)
			deriv[i] = dx[i];
	}

	constexpr autodouble(real x, std::array<real,d> dx) : x(x), deriv(dx)
	{}

	constexpr autodouble() : x(0.0)
	{
		for (dim_t i=0; i<d; ++i)
			deriv[i] = 0.0;
	}

	constexpr static autodouble constant(real x);

	template<dim_t i>
	constexpr static autodouble variable(real x);

	real value() const;

	real derivative(dim_t dimension) const;

	autodouble operator-() const;

	void invert_sign();

	autodouble operator+(const autodouble& other) const;
	autodouble operator-(const autodouble& other) const;
	autodouble operator*(const autodouble& other) const;
	autodouble operator/(const autodouble& other) const;

	autodouble operator+(real c) const;
	autodouble operator-(real c) const;
	autodouble operator*(real c) const;
	autodouble operator/(real c) const;

	autodouble& operator+=(const autodouble& other);
	autodouble& operator-=(const autodouble& other);
	autodouble& operator*=(const autodouble& other);
	autodouble& operator+=(real c);
	autodouble& operator*=(real c);
	autodouble& operator-=(real c);

	bool operator>=(const autodouble& other) const;
	bool operator<=(const autodouble& other) const;
	bool operator>(const autodouble& other) const;
	bool operator<(const autodouble& other) const;
	bool operator>=(real) const;
	bool operator<=(real) const;
	bool operator>(real) const;
	bool operator<(real) const;
	bool operator==(real) const;

	static autodouble div(real c, const autodouble& x);
	static autodouble minus(real c, const autodouble& x);
	static autodouble minus(const autodouble& x, autodouble&& y);

	static autodouble exp(const autodouble& x);
	static autodouble exp(autodouble&& x);
	static autodouble log(const autodouble& x);
	static autodouble log_move(autodouble&& x);
	static autodouble sqrt(const autodouble& x);
	static autodouble sqrt(autodouble&& x);
	static autodouble pow(const autodouble& x, const autodouble& a);
	static autodouble pow(const autodouble& x, const real a);
	static autodouble pow(real x, const autodouble& a);
	static autodouble pow(autodouble&& x, real a);

	static autodouble abs(autodouble&& x);
	static long floor(autodouble&& x);

	template<typename T>
	static autodouble fmod(const autodouble& a, const T& b);

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
	static autodouble kahan_sum(const std::vector<autodouble>&);
	static autodouble recursive_sum(const std::vector<autodouble>&);

private:
	real x;
	std::array<real,d> deriv;

	autodouble(real x) : x(x)
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
template<dim_t d, typename real>
constexpr autodouble<d,real> autodouble<d,real>::constant(real x)
{
	// First initialize the value and set all derivatives to 0:
	return autodouble<d,real>(x);
}

template<dim_t d, typename real>
template<dim_t i>
constexpr autodouble<d,real> autodouble<d,real>::variable(real x)
{
	static_assert(i < d, "Out-of-bounds variable access");
	// First initialize empty autodouble:
	autodouble ad(x);

	// Seed the correct derivative:
	ad.deriv[i] = 1.0;

	return ad;
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::nan()
{
	// First initialize empty autodouble:
	constexpr real nan = std::nan("");
	autodouble<d,real> ad(nan);

	// Seed the correct derivative:
	for (dim_t i=0; i<d; ++i)
		ad.deriv[i] = nan;

	return ad;
}



/*
 * Value access:
 */
template<dim_t d,typename real>
real autodouble<d,real>::derivative(dim_t dimension) const
{
	if (dimension > d)
		throw std::runtime_error("Try to access out-of-bound derivative");

	return deriv[dimension];
}


/*
 * Operators:
 */
template<dim_t d, typename real>
real autodouble<d,real>::value() const
{
	return x;
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::operator-() const
{
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = -deriv[i];
	return autodouble(-x, deriv_new);
}

template<dim_t d, typename real>
void autodouble<d,real>::invert_sign()
{
	x = -x;
	for (dim_t i=0; i<d; ++i)
		deriv[i] = -deriv[i];
}


/*
 * Addition:
 */

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::operator+(const autodouble<d,real>& other) const
{
	std::array<real,d> deriv_new;
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

template<dim_t d, typename real>
autodouble<d,real>
operator+(autodouble<d,real>&& x, const autodouble<d,real>& y)
{
	x += y;
	return x;
}

template<dim_t d, typename real>
autodouble<d,real>
operator+(const autodouble<d,real>& x, autodouble<d,real>&& y)
{
	y += x;
	return y;
}

template<dim_t d, typename real>
autodouble<d,real>
operator+(autodouble<d,real>&& x, autodouble<d,real>&& y)
{
	x += y;
	return x;
}



/*
 * Subtraction:
 */


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::operator-(const autodouble<d,real>& other) const
{
	std::array<real,d> deriv_new;
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

template<dim_t d, typename real>
autodouble<d,real>
operator-(autodouble<d,real>&& x, const autodouble<d,real>& y)
{
	x -= y;
	return x;
}

template<dim_t d, typename real>
autodouble<d,real> operator-(autodouble<d,real>&& x, autodouble<d,real>&& y)
{
	x -= y;
	return x;
}

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::minus(const autodouble<d,real>& x, autodouble<d,real>&& y)
{
	/* x-y = x + (-y) = (-y) + x */
	y.x = x.x - y.x;
	for (dim_t i=0; i<d; ++i)
		y.deriv[i] = x.deriv[i] - y.deriv[i];
	return y;
}

template<dim_t d, typename real>
autodouble<d,real>
operator-(const autodouble<d,real>& x, autodouble<d,real>&& y)
{
	return autodouble<d,real>::minus(x, std::move(y));
}

/*
 * Multiplication:
 */

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::operator*(const autodouble<d,real>& other) const
{
	std::array<real,d> deriv_new;
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

template<dim_t d, typename real>
autodouble<d,real>
operator*(autodouble<d,real>&& x, const autodouble<d,real>& y)
{
	x *= y;
	return x;
}

template<dim_t d, typename real>
autodouble<d,real>
operator*(const autodouble<d,real>& x, autodouble<d,real>&& y)
{
	y *= x;
	return y;
}

template<dim_t d, typename real>
autodouble<d,real>
operator*(autodouble<d,real>&& x, autodouble<d,real>&& y)
{
	x *= y;
	return x;
}




template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::operator/(const autodouble<d,real>& other) const
{
	std::array<real,d> deriv_new;
	const real denom = 1.0 / (other.x * other.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = denom * (other.x * deriv[i] - x * other.deriv[i]);

	return autodouble(x / other.x, deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::operator*(real c) const
{
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = c * deriv[i];

	return autodouble(c*x, deriv_new);
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::operator+(real c) const
{
	return autodouble(x+c, deriv);
}

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::operator-(real c) const
{
	return autodouble(x-c, deriv);
}

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::operator/(real c) const
{
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = deriv[i] / c;

	return autodouble(x/c, deriv_new);
}


/*
 * Inplace operators:
 */

template<dim_t d, typename real>
autodouble<d,real>&
autodouble<d,real>::operator+=(const autodouble<d,real>& other)
{
	x += other.x;
	for (dim_t i=0; i<d; ++i)
		deriv[i] += other.deriv[i];

	return *this;
}

template<dim_t d, typename real>
autodouble<d,real>& autodouble<d,real>::operator+=(real c)
{
	x += c;
	return *this;
}


template<dim_t d, typename real>
autodouble<d,real>&
autodouble<d,real>::operator-=(const autodouble<d,real>& other)
{
	x -= other.x;
	for (dim_t i=0; i<d; ++i)
		deriv[i] -= other.deriv[i];

	return *this;
}

template<dim_t d, typename real>
autodouble<d,real>&
autodouble<d,real>::operator-=(real c)
{
	x -= c;
	return *this;
}

template<dim_t d, typename real>
autodouble<d,real>&
autodouble<d,real>::operator*=(const autodouble<d,real>& other)
{
	for (dim_t i=0; i<d; ++i)
		deriv[i] = other.x * deriv[i] + x * other.deriv[i];

	x *= other.x;

	return *this;
}

template<dim_t d, typename real>
autodouble<d,real>&
autodouble<d,real>::operator*=(real c)
{
	x *= c;
	for (dim_t i=0; i<d; ++i)
		deriv[i] *= c;

	return *this;
}


/*
 * Comparison operators:
 */
template<dim_t d, typename real>
bool autodouble<d,real>::operator>=(const autodouble& other) const
{
	return x >= other.x;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator<=(const autodouble& other) const
{
	return x <= other.x;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator>(const autodouble& other) const
{
	return x > other.x;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator<(const autodouble& other) const
{
	return x < other.x;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator>=(real y) const
{
	return x >= y;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator>(real y) const
{
	return x > y;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator<=(real y) const
{
	return x <= y;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator<(real y) const
{
	return x < y;
}

template<dim_t d, typename real>
bool autodouble<d,real>::operator==(real y) const
{
	return x == y;
}






/*
 * Other operators:
 */
template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::div(real c, const autodouble<d,real>& x)
{

	std::array<real,d> deriv_new;
	const real f = - c / (x.x * x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble(c/x.x, deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> operator/(real c, const autodouble<d,real>& x)
{
	return autodouble<d,real>::div(c, x);
}

template<dim_t d, typename real>
autodouble<d,real> operator*(real c, const autodouble<d,real>& x)
{
	// Call autodouble's method
	return x * c;
}

template<dim_t d, typename real>
autodouble<d,real> operator*(real c, autodouble<d,real>&& x)
{
	x *= c;
	return x;
}

template<dim_t d, typename real>
autodouble<d,real> operator+(real c, const autodouble<d,real>& x)
{
	// Call autodouble's method
	return x + c;
}

template<dim_t d, typename real>
autodouble<d,real> operator+(real c, autodouble<d,real>&& x)
{
	// Call autodouble's method
	x += c;
	return x;
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::minus(real c, const autodouble& x)
{
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = -x.deriv[i];

	return autodouble(c - x.x, deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> operator-(real c, const autodouble<d,real>& x)
{
	// Call autodouble's method
	return autodouble<d,real>::minus(c, x);
}







/*
 * Trigonometric functions:
 */

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::sin(const autodouble& x)
{
	std::array<real,d> deriv_new;
	const real cos_x = std::cos(x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = cos_x * x.deriv[i];

	return autodouble(std::sin(x.x), deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::sin(autodouble&& x)
{
	const real cos_x = std::cos(x.x);
	x.x = std::sin(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= cos_x;

	return x;
}



template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::cos(const autodouble& x)
{
	std::array<real,d> deriv_new;
	const real sin_x = std::sin(x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = - sin_x * x.deriv[i];

	return autodouble(std::cos(x.x), deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::cos(autodouble&& x)
{
	const real n_sin_x = -std::sin(x.x);
	x.x = std::cos(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= n_sin_x;

	return x;
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::tan(const autodouble& x)
{
	std::array<real,d> deriv_new;
	const real tan_x = std::tan(x.x);
	const real f = 1.0 + tan_x * tan_x;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d,real>(tan_x, deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::atan(const autodouble& x)
{
	/* d/dx atan(x) = 1/(1+x^2) */
	const real f = 1.0 / (1.0 + x.x * x.x);
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d,real>(std::atan(x.x), deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::atan(autodouble&& x)
{
	/* d/dx atan(x) = 1/(1+x^2) */
	const real f = 1.0 / (1.0 + x.x * x.x);
	x.x = std::atan(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] = f * x.deriv[i];

	return x;
}

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::atan2(const autodouble& y, const autodouble& x)
{
	/* d/dx atan(z) = 1/(1+z^2) */
	const real z = y.x / x.x;
	const real f = 1.0 / (1.0 + z * z);
	const real ix = 1.0 / x.x;
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * ix * (y.deriv[i] - y.x * ix * x.deriv[i]);

	return autodouble<d,real>(std::atan2(y.x, x.x), deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::atan2(autodouble&& y, const autodouble& x)
{
	/* d/dx atan(z) = 1/(1+z^2) */
	const real yx = y.x;
	const real z = yx / x.x;
	const real f = 1.0 / (1.0 + z * z);
	const real ix = 1.0 / x.x;
	y.x = std::atan2(y.x, x.x);
	for (dim_t i=0; i<d; ++i)
		y.deriv[i] = f * ix * (y.deriv[i] - yx * ix * x.deriv[i]);

	return y;
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::atanh(const autodouble& x)
{
	/* Domain check: */
	if (std::abs(x.x) >= 1.0)
		return autodouble<d,real>::nan();

	/* d/dx atanh(x) = 1/(1 - x^2) */
	const real f = 1.0 / (1.0 - x.x * x.x);
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d,real>(std::atanh(x.x), deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::atanh_move(autodouble&& x)
{
	/* Domain check: */
	if (std::abs(x.x) >= 1.0){
		x.x = std::nan("");
		for (dim_t i=0; i<d; ++i)
			x.deriv[i] = std::nan("");
	} else {
		/* d/dx atanh(x) = 1/(1 - x^2) */
		const real f = 1.0 / (1.0 - x.x * x.x);
		x.x = std::atanh(x.x);
		for (dim_t i=0; i<d; ++i)
			x.deriv[i] *= f;
	}

	return x;
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::asin(const autodouble& x)
{
	/* d/dx asin(x) = 1/sqrt(1-x^2) */
	const real f = std::min<real>(1e8, 1.0 / std::sqrt(1.0 - x.x * x.x));
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble<d,real>(std::asin(x.x), deriv_new);
}

/*
 * Other functions
 */


template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::exp(const autodouble& x)
{
	std::array<real,d> deriv_new;
	const real exp_x = std::exp(x.x);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = exp_x * x.deriv[i];

	return autodouble(exp_x, deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::exp(autodouble&& x)
{
	x.x = std::exp(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= x.x;

	return x;
}


template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::log(const autodouble& x)
{
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = x.deriv[i] / x.x;

	return autodouble(std::log(x.x), deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::log_move(autodouble&& x)
{
	const real ix = 1.0 / x.x;
	x.x = std::log(x.x);
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= ix;

	return x;
}


template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::sqrt(const autodouble& x)
{
	std::array<real,d> deriv_new;
	const real sqrt = std::sqrt(x.x);
	const real f = 0.5 / sqrt;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble(sqrt, deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::sqrt(autodouble&& x)
{
	x.x = std::sqrt(x.x);
	const real f = 0.5 / x.x;
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= f;

	return x;
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::pow(const autodouble& x, const autodouble& a)
{
	real x_pow_a = std::pow(x.x, a.x);
	real c0 = a.x * std::pow(x.x, a.x-1.0);
	real c1 = std::log(x.x) * x_pow_a;
	std::array<real,d> deriv_new;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = c0 * x.deriv[i] + c1 * a.deriv[i];
	return autodouble(x_pow_a, deriv_new);
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::pow(const autodouble& x, real a)
{
	std::array<real,d> deriv_new;
	const real f = a * std::pow(x.x, a-1.0);
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f * x.deriv[i];

	return autodouble(std::pow(x.x, a), deriv_new);
}

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::pow(real x, const autodouble& a)
{
	std::array<real,d> deriv_new;
	const real f0 = std::pow(x, a.x);
	const real f1 = std::log(x) * f0;
	for (dim_t i=0; i<d; ++i)
		deriv_new[i] = f1 * a.deriv[i];

	return autodouble(f0, deriv_new);
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::pow(autodouble&& x, real a)
{
	real f = std::pow(x.x, a-1.0);
	x.x *= f; // x.x := x.x*pow(x.x, a-1) = pow(x.x,a)
	f *= a;
	for (dim_t i=0; i<d; ++i)
		x.deriv[i] *= f;

	return x;
}


template<dim_t d, typename real>
autodouble<d,real> autodouble<d,real>::abs(autodouble&& x)
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


template<dim_t d, typename real>
long autodouble<d,real>::floor(autodouble&& x)
{
	return std::floor(x.x);
}


template<dim_t d, typename real>
template<typename T>
autodouble<d,real> autodouble<d,real>::fmod(const autodouble& a, const T& b)
{
	return autodouble(std::fmod(a.x, b), a.deriv);
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::min(const autodouble& x,const autodouble& y)
{
	if (x.x <= y.x)
		return x;
	return y;
}

template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::max(const autodouble& x, const autodouble& y)
{
	if (x.x >= y.x)
		return x;
	return y;
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::recursive_sum(const std::vector<autodouble<d,real>>& x)
{
	return doomercat::recursive_sum<autodouble<d,real>>(x);
}


template<dim_t d, typename real>
autodouble<d,real>
autodouble<d,real>::sum(const std::vector<autodouble<d,real>>& x)
{
	return recursive_sum(x);
}


#endif // ifndef AUTODOUBLE
