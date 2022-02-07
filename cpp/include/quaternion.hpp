/*
 * Quaternion class for data-driven optimizaion of the Laborde
 * oblique Mercator projection.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
 *          Sebastian von Specht
 *
 * Copyright (C) 2019-2021 Deutsches GeoForschungsZentrum Potsdam,
 *                         Sebastian von Specht
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

#include <array>
#include <../include/arithmetic.hpp>

#ifndef DOOMERCAT_QUATERNION_H
#define DOOMERCAT_QUATERNION_H

template<typename T>
class Quaternion;

template<typename T>
class ImaginaryQuaternion {
	friend Quaternion<T>;
public:
	ImaginaryQuaternion(const T& i, const T& j, const T& k)
		: _i(i), _j(j), _k(k)
	{};

	T norm() const;

	const T& i() const;
	const T& j() const;
	const T& k() const;

	Quaternion<T> operator*(const Quaternion<T>&) const;
//	Quaternion operator*(const T&) const;
//	Quaternion operator/(const T&) const;

/*	void operator/=(const T&);

	template<typename T2, typename = std::enable_if_t<std::is_same<T2,double>::value>>
	Quaternion operator*(const Quaternion<T2>&) const;

	template<typename T2, typename = std::enable_if_t<!std::is_same<T2,double>::value>>
	Quaternion<T2> operator*(const Quaternion<T2>&) const;

	Quaternion conj() const;

	std::array<T,3> imag() const;

	Quaternion& operator*=(const Quaternion& r);
	*/

private:
	T _i,_j,_k;
};

template<typename T>
Quaternion<T> ImaginaryQuaternion<T>::operator*(const Quaternion<T>& p) const
{
	return Quaternion<T>(
	            std::move(- (_i * p.i()) - (_j * p.j()) - (_k * p.k())),
	            std::move(  (_i * p.r()) + (_j * p.k()) - (_k * p.j())),
	            std::move(- (_i * p.k()) + (_j * p.r()) + (_k * p.i())),
	            std::move(  (_i * p.j()) - (_j * p.i()) + (_k * p.r())));
}




template<typename T>
class Quaternion {
	friend ImaginaryQuaternion<T>;
public:
	Quaternion(const T& r, const T& i, const T& j, const T& k)
		: _r(r), _i(i), _j(j), _k(k)
	{};
	Quaternion(T&& r, T&& i, T&& j, T&& k)
		: _r(std::move(r)), _i(std::move(i)), _j(std::move(j)), _k(std::move(k))
	{};

	T norm() const;

	const T& r() const;
	const T& i() const;
	const T& j() const;
	const T& k() const;

	Quaternion operator*(const Quaternion&) const;
	Quaternion operator*(const ImaginaryQuaternion<T>&) const;
	Quaternion operator*(const T&) const;
	Quaternion operator/(const T&) const;

	Quaternion& operator/=(const T&);

	template<typename T2, typename = std::enable_if_t<!std::is_same<T2,double>::value>>
	Quaternion<T2> operator*(const Quaternion<T2>&) const;

	Quaternion conj() const;

	std::array<T,3> imag() const;

	Quaternion& operator*=(const Quaternion& r);

private:
	T _r,_i,_j,_k;
};


template<typename T>
T Quaternion<T>::norm() const
{
	return Arithmetic<T>::sqrt(_r*_r + _i*_i + _j*_j + _k*_k);
}


template<typename T>
const T& Quaternion<T>::r() const
{
	return _r;
}


template<typename T>
const T& Quaternion<T>::i() const
{
	return _i;
}


template<typename T>
const T& Quaternion<T>::j() const
{
	return _j;
}


template<typename T>
const T& Quaternion<T>::k() const
{
	return _k;
}


template<typename T>
Quaternion<T> Quaternion<T>::conj() const
{
	return Quaternion(_r, -_i, -_j, -_k);
}


template<typename T>
std::array<T,3> Quaternion<T>::imag() const
{
	return std::array<T,3>({_i, _j, _k});
}

template<typename T>
Quaternion<T> Quaternion<T>::operator*(const Quaternion<T>& p) const
{
	/* As long as we can multiply T and T2, we can formulate Quaternion
	 * multiplication: */
	return Quaternion(
	      std::move((_r * p._r) - (_i * p._i) - (_j * p._j) - (_k * p._k)),
	      std::move((_r * p._i) + (_i * p._r) + (_j * p._k) - (_k * p._j)),
	      std::move((_r * p._j) - (_i * p._k) + (_j * p._r) + (_k * p._i)),
	      std::move((_r * p._k) + (_i * p._j) - (_j * p._i) + (_k * p._r)));
}


template<>
template<typename T2, typename>// = std::enable_if_t<!std::is_same<T2,double>::value>>
Quaternion<T2> Quaternion<double>::operator*(const Quaternion<T2>& p) const
{
	static_assert(!std::is_same<T2,double>::value, "Instantiate wrong "
	                                               "Quaternion multiplication");
	/* As long as we can multiply T and T2, we can formulate Quaternion
	 * multiplication: */
	return Quaternion<T2>(
	    std::move((_r * p.r()) - (_i * p.i()) - (_j * p.j()) - (_k * p.k())),
	    std::move((_r * p.i()) + (_i * p.r()) + (_j * p.k()) - (_k * p.j())),
	    std::move((_r * p.j()) - (_i * p.k()) + (_j * p.r()) + (_k * p.i())),
	    std::move((_r * p.k()) + (_i * p.j()) - (_j * p.i()) + (_k * p.r())));
}

template<typename T>
Quaternion<T>& Quaternion<T>::operator*=(const Quaternion& p)
{
	const T r = _r;
	_r = (r * p._r) - (_i * p._i) - (_j * p._j) - (_k * p._k);
	const T i = _i;
	_i = (r * p._i) + (i * p._r) + (_j * p._k) - (_k * p._j);
	const T j = _j;
	_j = (r * p._j) - (i * p._k) + (j * p._r) + (_k * p._i);
	_k = (r * p._k) + (i * p._j) - (j * p._i) + (_k * p._r);
	return *this;
}


template<typename T>
Quaternion<T> Quaternion<T>::operator*(const ImaginaryQuaternion<T>& p) const
{
	return Quaternion(std::move(- (_i * p._i) - (_j * p._j) - (_k * p._k)),
	                  std::move(  (_r * p._i) + (_j * p._k) - (_k * p._j)),
	                  std::move(  (_r * p._j) - (_i * p._k) + (_k * p._i)),
	                  std::move(  (_r * p._k) + (_i * p._j) - (_j * p._i)));
}


template<typename T>
Quaternion<T> Quaternion<T>::operator*(const T& t) const
{
	return Quaternion(std::move(_r * t),
	                  std::move(_i * t),
	                  std::move(_j * t),
	                  std::move(_k * t));
}


template<typename T>
Quaternion<T> Quaternion<T>::operator/(const T& t) const
{
	return Quaternion(std::move(_r/t),
	                  std::move(_i/t),
	                  std::move(_j/t),
	                  std::move(_k/t));
}

template<typename T>
Quaternion<T>& Quaternion<T>::operator/=(const T& val)
{
	_r /= val;
	_i /= val;
	_j /= val;
	_k /= val;
	return *this;
}


template<typename T>
Quaternion<T> operator/(Quaternion<T>&& q, const T& t)
{
	q /= t;
	return q;
}


/*
 * Quaternion rotation:
 */

template<typename T>
Quaternion<T> qrot(Quaternion<T>&& q0, ImaginaryQuaternion<T>&& p,
                   const Quaternion<T>& q1)
{
	Quaternion<T> tmp(p * q1);
	q0 *= tmp;
	return q0;
}

template<typename T, typename T2>
Quaternion<T> qrot(const Quaternion<T>& q0, const Quaternion<T2> p,
                   const Quaternion<T>& q1)
{
	return q0 * (p * q1);
}

#endif
