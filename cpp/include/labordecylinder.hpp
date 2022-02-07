/*
 * Convenience methods for the rotating cylinder of the Laborde oblique Mercator
 * projection.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de),
 *          Sebastian von Specht
 *
 * Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam,
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

#include <../include/types.hpp>
#include <../include/quaternion.hpp>
#include <memory>

#ifndef LABORDECYLINDER_H
#define LABORDECYLINDER_H

namespace doomercat {

class LabordeCylinder {
public:
	LabordeCylinder(double qr, double qi, double qj, double qk, double k0,
	                double f);

	LabordeCylinder(const Quaternion<real5v>& q, const real5v& k0,
	                const double f);

	vec3_5v axis() const;

	real5v azimuth() const;

	const real5v& k0() const;

	const Quaternion<real5v>& rotation_quaternion() const;

	static std::shared_ptr<LabordeCylinder>
	   from_axis_lon_lat(double lon, double lat, double k0, double f);

	static std::shared_ptr<LabordeCylinder>
	   from_parameters(double qr, double qi, double qj, double qk, double k0,
	                   double f);

	const real5v& B() const;
	const real5v& C() const;
	double e() const;
	double f() const;

private:
	Quaternion<real5v> q;
	real5v _k0;
	double _f;
	double e2;
	double _e;
	vec3_5v _axis;
	real5v fies;
	real5v vize;
	real5v _B;
	real5v _C;

	static vec3_5v compute_axis(const Quaternion<real5v>& q);
};

}

#endif
