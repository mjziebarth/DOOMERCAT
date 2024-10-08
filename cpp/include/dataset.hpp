/*
 * Data set wrapper
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
 *               2024 Technical University of Munich
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

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <../include/functions.hpp>

#ifndef DOOMERCAT_DATASET_H
#define DOOMERCAT_DATASET_H

namespace doomercat {

/*
 * The data entries: (un-)weighted, with(out) height information:
 */

template<bool has_height, bool has_weight>
struct data_entry_t;

template<>
struct data_entry_t<true,true> {
	double lambda; //Longitude
	double phi; // Latitude
	double k_e; // Elevation scale k_e
	double w; // weight
};

template<>
struct data_entry_t<true,false> {
	double lambda;
	double phi;
	double k_e;
};

template<>
struct data_entry_t<false,true> {
	double lambda;
	double phi;
	double w;
};

template<>
struct data_entry_t<false,false> {
	double lambda;
	double phi;
};


/*
 * The common functionality:
 */

template<bool has_height, bool has_weight>
class DataSet {
/*
 * Class representing a data set in optimization.
 */

public:
	DataSet(size_t N, double W) : data(N), W(W)
	{};

	size_t size() const {
		return data.size();
	};

	double lambda(size_t i) const {
		if (i >= data.size())
			throw std::runtime_error("Out-of-bounds in DataSet lambda access.");
		return data[i].lambda;
	};

	double phi(size_t i) const {
		if (i >= data.size())
			throw std::runtime_error("Out-of-bounds in DataSet phi access.");
		return data[i].phi;
	};

	/* This function sums up all weights: */
	double summed_weight() const
	{
		return W;
	}

protected:
	typedef data_entry_t<has_height,has_weight> entry_t;

	std::vector<entry_t> data;

	double W;
};


/*
 * Height and weight functionality:
 */
class WeightedDataSetWithHeight : public DataSet<true,true> {
public:
	WeightedDataSetWithHeight(const size_t N, const double* lon,
	                          const double* lat, const double* h,
	                          const double* w, double a, double f);

	double w(size_t i) const;
	double k_e(size_t i) const;
};

class DataSetWithHeight : public DataSet<true,false> {
public:
	DataSetWithHeight(const size_t N, const double* lon, const double* lat,
	                  const double* h, double a, double f);

	constexpr double w(size_t i) const {
		return 1.0;
	};

	double k_e(size_t i) const;
};

class WeightedDataSet : public DataSet<false,true> {
public:
	WeightedDataSet(const size_t N, const double* lon, const double* lat,
	                const double* w);

	double w(size_t i) const;

	constexpr double k_e(size_t i) const {
		return 1.0;
	};
};

class SimpleDataSet : public DataSet<false,false> {
public:
	SimpleDataSet(const size_t N, const double* lon, const double* lat);

	constexpr double w(size_t i) const {
		return 1.0;
	};

	constexpr double k_e(size_t i) const {
		return 1.0;
	};
};


}

#endif
