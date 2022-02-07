/*
 * Laborde-projected data.
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

#include <vector>
#include <memory>

#include <../include/dataset.hpp>
#include <../include/labordecylinder.hpp>

#ifndef DOOMERCAT_LABORDEPROJECTEDDATASET_H
#define DOOMERCAT_LABORDEPROJECTEDDATASET_H

namespace doomercat {

class LabordeProjectedDataSet {
/*
 * A DataSet instance that has been projected as defined by
 *a Laborde cylinder.
 */
public:
	LabordeProjectedDataSet(std::shared_ptr<const DataSet> data,
	                        std::shared_ptr<const LabordeCylinder> cylinder);


	size_t size() const;

	double w(size_t i) const;

	const vec3_5v& uvw(size_t i) const;

	const LabordeCylinder& cylinder() const;

private:
	std::shared_ptr<const DataSet> data;
	std::shared_ptr<const LabordeCylinder> cyl;
	std::vector<vec3_5v> _uvw;

	void compute_uvw(const std::vector<DataSet::entry_t>& lola,
	                 const LabordeCylinder& cyl);
};

}

#endif
