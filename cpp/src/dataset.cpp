/*
 * Data set wrapper
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
 *          Sebastian von Specht
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
 *                    Sebastian von Specht
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

#include <../include/dataset.hpp>
#include <cmath>
#include <stdexcept>
#include <iostream>

using doomercat::SimpleDataSet;
using doomercat::DataSetWithHeight;
using doomercat::WeightedDataSet;
using doomercat::WeightedDataSetWithHeight;

static long double compute_k_e(double h, double a, double f)
{
	/* The elevation scale factor: */
	return a / (a + h);
}

SimpleDataSet::SimpleDataSet(const size_t N, const double* lon,
                             const double* lat)
    : DataSet<false,false>(N)
{
	for (size_t i=0; i<N; ++i){
		data[i].lambda = deg2rad(std::fmod((lon[i]+180.0), 360.) - 180.0);
	}
	for (size_t i=0; i<N; ++i){
		data[i].phi = deg2rad(lat[i]);
	}
}

DataSetWithHeight::DataSetWithHeight(const size_t N, const double* lon,
                                     const double* lat, const double* h,
                                     double a, double f)
    : DataSet<true,false>(N)
{
	for (size_t i=0; i<N; ++i){
		data[i].lambda = deg2rad(std::fmod((lon[i]+180.0), 360.) - 180.0);
	}
	for (size_t i=0; i<N; ++i){
		data[i].phi = deg2rad(lat[i]);
	}
	for (size_t i=0; i<N; ++i){
		data[i].k_e = compute_k_e(h[i], a, f);
	}
}

WeightedDataSet::WeightedDataSet(const size_t N, const double* lon,
                                 const double* lat, const double* w)
    : DataSet<false,true>(N)
{
	for (size_t i=0; i<N; ++i){
		data[i].lambda = deg2rad(std::fmod((lon[i]+180.0), 360.) - 180.0);
	}
	for (size_t i=0; i<N; ++i){
		data[i].phi = deg2rad(lat[i]);
	}
	for (size_t i=0; i<N; ++i){
		data[i].w = w[i];
	}
}

WeightedDataSetWithHeight::WeightedDataSetWithHeight(const size_t N,
                                 const double* lon, const double* lat,
                                 const double* h, const double* w,
                                 double a, double f)
    : DataSet<true,true>(N)
{
	for (size_t i=0; i<N; ++i){
		data[i].lambda = deg2rad(std::fmod((lon[i]+180.0), 360.) - 180.0);
	}
	for (size_t i=0; i<N; ++i){
		data[i].phi = deg2rad(lat[i]);
	}
	for (size_t i=0; i<N; ++i){
		data[i].k_e = compute_k_e(h[i], a, f);
	}
	for (size_t i=0; i<N; ++i){
		data[i].w = w[i];
	}
}

double WeightedDataSet::w(size_t i) const
{
	return data[i].w;
}

double DataSetWithHeight::k_e(size_t i) const
{
	return data[i].k_e;
}

double WeightedDataSetWithHeight::w(size_t i) const
{
	return data[i].w;
}

double WeightedDataSetWithHeight::k_e(size_t i) const
{
	return data[i].k_e;
}
