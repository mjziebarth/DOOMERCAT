/*
 * Data set wrapper
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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
#include <stdexcept>

using doomercat::DataSet;


DataSet::DataSet(const size_t N, const double* lon, const double* lat,
                 const double* w)
    : data(N)
{
	for (size_t i=0; i<N; ++i){
		data[i].lambda = deg2rad(std::fmod((lon[i]+180.0), 360.) - 180.0);
	}
	for (size_t i=0; i<N; ++i){
		data[i].phi = deg2rad(lat[i]);
	}
	if (w){
		for (size_t i=0; i<N; ++i){
			data[i].w = w[i];
		}
	} else {
		for (size_t i=0; i<N; ++i){
			data[i].w = 1.0;
		}
	}
}

size_t DataSet::size() const
{
	return data.size();
}

double DataSet::w(size_t i) const
{
	if (i >= data.size())
		throw std::runtime_error("Out-of-bounds in DataSet weight access.");

	return data[i].w;
}
