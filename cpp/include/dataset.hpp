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

#include <cstddef>
#include <vector>
#include <../include/functions.hpp>

#ifndef DOOMERCAT_DATASET_H
#define DOOMERCAT_DATASET_H

namespace doomercat {

class LabordeProjectedDataSet;

class DataSet {
/*
 * Class representing a data set in optimization.
 */
friend LabordeProjectedDataSet;

public:
	DataSet(const size_t N, const double* lon, const double* lat,
	        const double* w = nullptr);

	size_t size() const;

	double lambda(size_t i) const;
	double phi(size_t i) const;
	double w(size_t i) const;

protected:
	struct entry_t {
		double lambda;
		double phi;
		double w;
	};

	std::vector<entry_t> data;

};

}

#endif
