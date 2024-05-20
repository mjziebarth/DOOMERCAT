/*
 * Choosing a linear algebra implementation.
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


#include <cmath>
#include <array>
#include <cstddef>
#include <stdexcept>

#include <iostream>
#include <chrono>
#include <thread>

#define NDEBUG
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>

#ifndef DOOMERCAT_LINALG_HPP
#define DOOMERCAT_LINALG_HPP

/* Choose here the library to use:
 * At this position, a custom implementation could be provided by
 * including a different header, which defines the *linalg_t* struct
 * accordingly, setting another DEFINE, and compiling.
 */
#define USE_CUSTOM


#ifdef USE_CUSTOM

template<size_t d, typename real>
class RowVector;

template<size_t d, typename real>
class ColVector;

/*
 *  The matrix class.
 */
template<size_t d, typename real>
class SqMatrix {
	friend RowVector<d,real>;
	friend ColVector<d,real>;

public:
	SqMatrix() {
		M.fill(0.0);
	};
	SqMatrix(SqMatrix&&) = default;
	explicit SqMatrix(const SqMatrix&) = default;
	SqMatrix& operator=(SqMatrix&&) = default;

	SqMatrix operator*(const SqMatrix& other) &&;
	SqMatrix operator+(const SqMatrix& other) const;
	SqMatrix operator-(const SqMatrix& other) &&;
	SqMatrix operator-(SqMatrix&& other) const;

	SqMatrix operator*(double c) &&;

	SqMatrix operator-() &&;

	real operator()(size_t i, size_t j) const;
	real& operator()(size_t i, size_t j);

	ColVector<d,real> operator*(const ColVector<d,real>&) const;
	RowVector<d,real> lmul(const RowVector<d,real>&) const;

	static SqMatrix identity();

private:
	std::array<real,d*d> M;
	real at(size_t i, size_t j) const;
	real& at(size_t i, size_t j);
};



template<size_t d, typename real>
SqMatrix<d,real>
SqMatrix<d,real>::operator*(const SqMatrix<d,real>& other) &&
{
	std::array<real,d> ri;
	for (size_t i=0; i<d; ++i){
		/* Save r[i,j] temporarily for this specific i: */
		for (size_t j=0; j<d; ++j){
			ri[j] = M[i*d + j];
			M[i*d + j] = 0.0;
		}
		/* Now overwrite r[i,j] for this specific i: */
		for (size_t k=0; k<d; ++k){
			for (size_t j=0; j<d; ++j){
				M[i*d + j] += ri[k] * other.M[d*k + j];
			}
		}
	}
	return std::move(*this);
	// SqMatrix<d,real> R;
	// for (size_t i=0; i<d; ++i)
	// 	for (size_t k=0; k<d; ++k)
	// 		for (size_t j=0; j<d; ++j)
	// 			R.at(i,j) += at(i,k) * other.at(k,j);
	// return R;
}



template<size_t d, typename real>
SqMatrix<d,real>
SqMatrix<d,real>::operator+(const SqMatrix<d,real>& other) const
{
	SqMatrix<d,real> R;
	for (size_t i=0; i<d*d; ++i)
		R.M[i] = M[i] + other.M[i];
	return R;
}

template<size_t d, typename real>
SqMatrix<d,real>
SqMatrix<d,real>::operator-(const SqMatrix<d,real>& other) &&
{
	for (size_t i=0; i<d*d; ++i)
		M[i] -= other.M[i];
	return std::move(*this);
}

template<size_t d, typename real>
SqMatrix<d,real>
SqMatrix<d,real>::operator-(SqMatrix<d,real>&& other) const
{
	for (size_t i=0; i<d*d; ++i)
		other.M[i] = M[i] - other.M[i];
	return std::move(other);
}

template<size_t d, typename real>
SqMatrix<d,real> SqMatrix<d,real>::operator-() &&
{
	for (size_t i=0; i<d*d; ++i)
		M[i] = -M[i];
	return *this;
}

template<size_t d, typename real>
ColVector<d,real> SqMatrix<d,real>::operator*(const ColVector<d,real>& v) const
{
	ColVector<d,real> r;
	for (size_t i=0; i<d; ++i){
		double ri = 0.0;
		for (size_t j=0; j<d; ++j)
			ri += M[i*d + j] * v.x[j];
		r[i] = ri;
	}
	return r;
}

template<size_t d, typename real>
RowVector<d,real> SqMatrix<d,real>::lmul(const RowVector<d,real>& v) const
{
	RowVector<d,real> r;
	for (size_t j=0; j<d; ++j){
		double rj = 0.0;
		for (size_t i=0; i<d; ++i)
			rj += v.x[i] * M[i*d + j];
		r[j] = rj;
	}
	return r;
}

template<size_t d, typename real>
RowVector<d,real> operator*(const RowVector<d,real>& v,
                            const SqMatrix<d,real>& M)
{
	return M.lmul(v);
}


template<size_t d, typename real>
SqMatrix<d,real>
SqMatrix<d,real>::operator*(double c) &&
{
	for (size_t i=0; i<d*d; ++i)
		M[i] *= c;

	return std::move(*this);
}

template<size_t d, typename real>
SqMatrix<d,real>
operator*(double c, const SqMatrix<d,real>& A)
{
	return SqMatrix<d,real>(A) * c;
}

template<size_t d, typename real>
SqMatrix<d,real> operator*(real c, SqMatrix<d,real>&& m)
{
	return std::move(m).operator*(c);
}




template<size_t d, typename real>
real SqMatrix<d,real>::operator()(size_t i, size_t j) const
{
	if (i >= d)
		throw std::runtime_error("Out-of-bounds acces in SqMatrix.");
	if (j >= d)
		throw std::runtime_error("Out-of-bounds acces in SqMatrix.");
	return at(i,j);
}

template<size_t d, typename real>
real& SqMatrix<d,real>::operator()(size_t i, size_t j)
{
	if (i >= d)
		throw std::runtime_error("Out-of-bounds acces in SqMatrix.");
	if (j >= d)
		throw std::runtime_error("Out-of-bounds acces in SqMatrix.");
	return at(i,j);
}

template<size_t d, typename real>
real SqMatrix<d,real>::at(size_t i, size_t j) const
{
	return M[d*i + j];
}

template<size_t d, typename real>
real& SqMatrix<d,real>::at(size_t i, size_t j)
{
	return M[d*i + j];
}

template<size_t d, typename real>
SqMatrix<d,real> SqMatrix<d,real>::identity()
{
	SqMatrix I;
	for (size_t i=0; i<d; ++i)
		I.M[d*i + i] = 1.0;
	return I;
}




/*
 * The vector classes.
 */

template<size_t d, typename real>
class RowVector
{
	friend class SqMatrix<d,real>;
	friend class ColVector<d,real>;

public:
	RowVector() {};
	RowVector(RowVector&&) = default;

	real& operator[](size_t i);
	real operator[](size_t i) const;

	ColVector<d,real> transpose() const;

	RowVector<d,real> operator*(double c) const;
	real operator*(const ColVector<d,real>& col) const;

private:
	std::array<real,d> x;
};

template<size_t d, typename real>
real RowVector<d,real>::operator*(const ColVector<d,real>& other) const
{
	real r=0.0;
	for (size_t i=0; i<d; ++i)
		r += x[i] * other.x[i];
	return r;
}

template<size_t d, typename real>
real& RowVector<d,real>::operator[](size_t i)
{
	if (i >= d)
		throw std::runtime_error("Index access out of bounds in RowVector.");
	return x[i];
}

template<size_t d, typename real>
real RowVector<d,real>::operator[](size_t i) const
{
	if (i >= d)
		throw std::runtime_error("Index access out of bounds in RowVector.");
	return x[i];
}

template<size_t d, typename real>
ColVector<d,real> RowVector<d,real>::transpose() const
{
	ColVector<d,real> v;
	for (size_t i=0; i<d;++i)
		v.x[i] = x[i];
	return v;
}


template<size_t d, typename real>
RowVector<d,real> RowVector<d,real>::operator*(double c) const
{
	RowVector<d,real> r;
	for (size_t i=0; i<d; ++i)
		r.x[i] = x[i] * c;
	return r;
}







template<size_t d, typename real>
class ColVector
{
	friend class SqMatrix<d,real>;
	friend class RowVector<d,real>;
public:
	ColVector() {};
	ColVector(ColVector&&) = default;
	ColVector(const ColVector&) = default;
	ColVector(std::array<real,d>&& x) : x(x) {};

	ColVector& operator=(ColVector&&) = default;
	ColVector& operator=(const ColVector&) = default;

	void fill_from_array(const std::array<real,d>&);

	void fill_array(std::array<real,d>& x) const;

	real& operator[](size_t i);
	real operator[](size_t i) const;

	SqMatrix<d,real> operator*(const RowVector<d,real>& r) const;
	ColVector<d,real> operator*(double c) const;
	ColVector operator+(const ColVector& other) const;
	ColVector operator-(const ColVector& other) const;
	ColVector& operator/=(double c);
	ColVector& operator-=(const ColVector& other);

	ColVector cross(const ColVector& other) const;

	ColVector operator-() &&;

	RowVector<d,real> transpose() const;

	real dot(const ColVector& other) const;

	real norm() const;

	bool operator==(const ColVector&) const;

private:
	std::array<real,d> x;
};


template<size_t d, typename real>
real ColVector<d,real>::operator[](size_t i) const
{
	if (i >= d)
		throw std::runtime_error("Index access out of bounds in ColVector.");
	return x[i];
}

template<size_t d, typename real>
real& ColVector<d,real>::operator[](size_t i)
{
	if (i >= d)
		throw std::runtime_error("Index access out of bounds in ColVector.");
	return x[i];
}

template<size_t d, typename real>
ColVector<d,real>
ColVector<d,real>::operator*(double c) const
{
	ColVector<d,real> r;
	for (size_t i=0; i<d; ++i)
		r.x[i] = c * x[i];
	return r;
}

template<size_t d, typename real>
SqMatrix<d,real>
ColVector<d,real>::operator*(const RowVector<d,real>& other) const
{
	SqMatrix<d,real> R;
	for (size_t i=0; i<d; ++i)
		for (size_t j=0; j<d; ++j)
			R.M[d*i+j] = x[i] * other.x[j];
	return R;
}

template<size_t d, typename real>
ColVector<d,real>
ColVector<d,real>::operator+(const ColVector<d,real>& other) const
{
	ColVector<d,real> r;
	for (size_t i=0; i<d; ++i)
		r.x[i] = x[i] + other.x[i];
	return r;
}

template<size_t d, typename real>
ColVector<d,real>
ColVector<d,real>::operator-(const ColVector<d,real>& other) const
{
	ColVector<d,real> r;
	for (size_t i=0; i<d; ++i)
		r.x[i] = x[i] - other.x[i];
	return r;
}

template<size_t d, typename real>
ColVector<d,real> ColVector<d,real>::operator-() &&
{
	for (size_t i=0; i<d; ++i)
		x[i] = -x[i];
	return std::move(*this);
}

template<size_t d, typename real>
ColVector<d,real>& ColVector<d,real>::operator/=(double c)
{
	for (size_t i=0; i<d; ++i)
		x[i] /= c;

	return *this;
}

template<size_t d, typename real>
ColVector<d,real>& ColVector<d,real>::operator-=(const ColVector& other)
{
	for (size_t i=0; i<d; ++i)
		x[i] -= other.x[i];

	return *this;
}

template<size_t d, typename real>
bool ColVector<d,real>::operator==(const ColVector<d,real>& v) const
{
	for (size_t i=0; i<d; ++i){
		if (x[i] != v.x[i])
			return false;
	}
	return true;
}


template<size_t d, typename real>
RowVector<d,real> ColVector<d,real>::transpose() const
{
	RowVector<d,real> v;
	for (size_t i=0; i<d;++i)
		v.x[i] = x[i];
	return v;
}

template<size_t d, typename real>
real ColVector<d,real>::dot(const ColVector<d,real>& other) const
{
	real dt(x[0] * other.x[0]);
	for (size_t i=1; i<d; ++i)
		dt += x[i] * other.x[i];
	return dt;
}

template<size_t d, typename real>
real ColVector<d,real>::norm() const
{
	double nrm2 = 0.0;
	for (size_t i=0; i<d; ++i)
		nrm2 += x[i] * x[i];
	return std::sqrt(nrm2);
}

template<size_t d, typename real>
void ColVector<d,real>::fill_from_array(const std::array<real,d>& y)
{
	for (size_t i=0; i<d; ++i)
		x[i] = y[i];
}

template<size_t d, typename real>
void ColVector<d,real>::fill_array(std::array<real,d>& y) const
{
	for (size_t i=0; i<d; ++i)
		y[i] = x[i];
}



/*
 * Other operators:
 */

template<size_t d, typename real>
ColVector<d,real> operator*(double c, ColVector<d,real>&& v)
{
	return std::move(v*c);
}

template<size_t d, typename real>
ColVector<d,real> operator*(double c, const ColVector<d,real>& v)
{
	return v*c;
}


/*
 * The linear algebra definition.
 */


template<size_t d, typename real>
struct linalg_t {
	constexpr static size_t ndim = d;
	typedef real real_t;
	typedef SqMatrix<d,real> matrix_dxd_t;
	typedef ColVector<d,real> column_vectord_t;
	typedef RowVector<d,real> row_vectord_t;

	typedef std::array<real,d> point_t;
	typedef std::array<real,d> grad_t;

	static matrix_dxd_t create_dxd_matrix()
	{
		return matrix_dxd_t();
	}

	static void init_column_vectord(column_vectord_t& v,
	                                const std::array<real,d>& data)
	{
		v.fill_from_array(data);
	}

	static void fill_array(std::array<real,d>& data,
	                       const column_vectord_t& v)
	{
		v.fill_array(data);
	}

	static row_vectord_t transpose(const column_vectord_t& v)
	{
		return v.transpose();
	}

	static real dot(const column_vectord_t& v0,
	                const column_vectord_t& v1)
	{
		return v0.dot(v1);
	}

	static real norm(const column_vectord_t& v)
	{
		return v.norm();
	}

	static matrix_dxd_t identity()
	{
		return matrix_dxd_t::identity();
	}

	static column_vectord_t solve(const matrix_dxd_t& A,
	                              const column_vectord_t& b)
	{
		namespace ublas = boost::numeric::ublas;
		typedef size_t index_t;

		/* The result will be stored here: */
		column_vectord_t res;

		ublas::matrix<real> Au(d,d);
		for (index_t i=0; i<d; ++i){
			for (index_t j=0; j<d; ++j){
				Au(i,j) = A(i,j);
			}
		}

		ublas::vector<real> bu(d);
		for (index_t i=0; i<d; ++i)
			bu(i) = b[i];


		/* Perform the matrix inversion using LU factorization and
		 * substitution */
		ublas::permutation_matrix<index_t> permutations(d);
		if (ublas::lu_factorize(Au, permutations)){
			std::cerr << "lu_factorize failed :-(\n" << std::flush;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			return res;
		}

		ublas::lu_substitute(Au, permutations, bu);

		/* Reassign: */
		for (index_t i=0; i<d; ++i)
			res[i] = bu(i);

		return res;
	}

};

#endif

#endif