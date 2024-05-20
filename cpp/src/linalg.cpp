/*
 * Choosing a linear algebra implementation.
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

#include <../include/linalg.hpp>
#include <../include/functions.hpp>

#ifdef USE_CUSTOM

template<>
ColVector<3,double>
ColVector<3,double>::cross(const ColVector<3,double>& other) const
{
	const double xnew = x[1] * other.x[2] - x[2] * other.x[1];
	const double ynew = x[2] * other.x[0] - x[0] * other.x[2];
	const double znew = x[0] * other.x[1] - x[1] * other.x[0];
	return ColVector<3, double>({xnew,ynew,znew});
}

template<>
ColVector<3, long double>
ColVector<3, long double>::cross(const ColVector<3, long double>& other) const
{
	const double xnew = x[1] * other.x[2] - x[2] * other.x[1];
	const double ynew = x[2] * other.x[0] - x[0] * other.x[2];
	const double znew = x[0] * other.x[1] - x[1] * other.x[0];
	return ColVector<3, long double>({xnew,ynew,znew});
}

#endif
