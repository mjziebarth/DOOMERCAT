/*
 * Arithmetics definition for automatically differentiating double.
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
#include <../include/arithmetic.hpp>
#include <../include/functions.hpp>

#include <cmath>

template<>
double Arithmetic<double>::sqrt(const double& x)
{
	return std::sqrt(x);
}

template<>
double Arithmetic<double>::sqrt(double&& x)
{
	return std::sqrt(x);
}

template<>
double Arithmetic<double>::sin(const double& d)
{
	return std::sin(d);
}

template<>
double Arithmetic<double>::sin(double&& d)
{
	return std::sin(d);
}

template<>
double Arithmetic<double>::cos(const double& d)
{
	return std::cos(d);
}

template<>
double Arithmetic<double>::cos(double&& d)
{
	return std::cos(d);
}

template<>
double Arithmetic<double>::tan(const double& d)
{
	return std::tan(d);
}

template<>
double Arithmetic<double>::log(double&& d)
{
	return std::log(d);
}

template<>
template<>
double Arithmetic<double>::pow(const double& a, const double& b)
{
	return std::pow(a,b);
}

template<>
double Arithmetic<double>::pow(double&& a, double b)
{
	return std::pow(a,b);
}

template<>
double Arithmetic<double>::pow(double&& a, int n)
{
	return std::pow(a,n);
}

template<>
double Arithmetic<double>::pow(const double& a, int n)
{
	return std::pow(a,n);
}

template<>
double Arithmetic<double>::atan2(double&& y, const double& x)
{
	return std::atan2(y,x);
}

template<>
double Arithmetic<double>::asin(double&& d)
{
	return std::asin(d);
}

template<>
double Arithmetic<double>::constant(double x)
{
	return x;
}

template<>
double Arithmetic<double>::abs(double&& x)
{
	return std::abs(x);
}

template<>
double Arithmetic<double>::log(const double& x)
{
	return std::log(x);
}




template<>
real5v Arithmetic<real5v>::constant(double x)
{
	return constant5(x);
}

template<>
real4v Arithmetic<real4v>::constant(double x)
{
	return constant4(x);
}
