/*
 * Arithmetics definitions for automatically differentiating double.
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

#include <utility>

#ifndef DOOMERCAT_ARITHMETIC_H
#define DOOMERCAT_ARITHMETIC_H

template<typename T>
class Arithmetic {
public:
	static T sqrt(const T&);
	static T sqrt(T&&);
	static T cos(const T&);
	static T cos(T&&);
	static T sin(const T&);
	static T sin(T&&);
	static T pow(const T&, int);
	static T pow(T&&, int);
	static T pow(T&&, double);
	static T pow2(const T&);
	static T pow2(T&&);
	template<typename T2>
	static T pow(const T2&, const T&);
	static T asin(T&&);
	static T tan(const T&);
	static T atan2(T&&, const T&);
	static T log(const T&);
	static T log(T&&);
	static T abs(T&&);

	static T max(T&&, T&&);

	static T constant(double);
};

/* Declare the specializations for double: */
template<>
double Arithmetic<double>::sqrt(const double&);

template<>
double Arithmetic<double>::sqrt(double&&);

template<>
double Arithmetic<double>::cos(const double&);

template<>
double Arithmetic<double>::cos(double&&);

template<>
double Arithmetic<double>::sin(const double&);

template<>
double Arithmetic<double>::sin(double&&);

template<>
double Arithmetic<double>::atan2(double&&, const double&);

template<>
double Arithmetic<double>::log(double&&);

template<>
double Arithmetic<double>::log(const double&);

template<>
template<>
double Arithmetic<double>::pow(const double&, const double&);

template<>
double Arithmetic<double>::pow(double&&, int);

template<>
double Arithmetic<double>::pow(double&&, double);

template<>
double Arithmetic<double>::asin(double&&);

template<>
double Arithmetic<double>::tan(const double&);

template<>
double Arithmetic<double>::abs(double&&);



template<typename T>
T Arithmetic<T>::sqrt(const T& t)
{
	return T::sqrt(t);
}

template<typename T>
T Arithmetic<T>::sqrt(T&& t)
{
	return T::sqrt(std::move(t));
}

template<typename T>
T Arithmetic<T>::cos(const T& t)
{
	return T::cos(t);
}

template<typename T>
T Arithmetic<T>::cos(T&& t)
{
	return T::cos(std::move(t));
}

template<typename T>
T Arithmetic<T>::sin(const T& t)
{
	return T::sin(t);
}

template<typename T>
T Arithmetic<T>::sin(T&& t)
{
	return T::sin(std::move(t));
}

template<typename T>
T Arithmetic<T>::pow(const T& t, int n)
{
	return T::pow(t, n);
}

template<typename T>
T Arithmetic<T>::pow(T&& t, int n)
{
	return T::pow(std::move(t), n);
}

template<typename T>
T Arithmetic<T>::pow(T&& t, double b)
{
	return T::pow(std::move(t), b);
}

template<typename T>
template<typename T2>
T Arithmetic<T>::pow(const T2& t, const T& n)
{
	return T::pow(t, n);
}

template<typename T>
T Arithmetic<T>::pow2(const T& t)
{
	return t * t;
}

template<typename T>
T Arithmetic<T>::pow2(T&& t)
{
	t *= t;
	return t;
}

template<typename T>
T Arithmetic<T>::asin(T&& t)
{
	return T::asin(std::move(t));
}

template<typename T>
T Arithmetic<T>::tan(const T& t)
{
	return T::tan(t);
}
template<typename T>
T Arithmetic<T>::atan2(T&& y, const T& x)
{
	return T::atan2(std::move(y), x);
}

template<typename T>
T Arithmetic<T>::log(const T& t)
{
	return T::log(t);
}

template<typename T>
T Arithmetic<T>::log(T&& t)
{
	return T::log(std::move(t));
}

template<typename T>
T Arithmetic<T>::abs(T&& t)
{
	return T::abs(std::move(t));
}



template<typename T>
T Arithmetic<T>::max(T&& l, T&& r)
{
	if (l >= r)
		return l;
	return r;
}

#endif
