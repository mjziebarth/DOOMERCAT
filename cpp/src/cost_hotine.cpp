/*
 * Cost function based on the Hotine oblique Mercator projection.
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

#include <../include/cost_hotine.hpp>
#include <vector>
#include <algorithm>

using doomercat::DataSet;
using doomercat::HotineObliqueMercator;
using doomercat::CostHotine;
using doomercat::CostFunctionHotine;
using doomercat::CostFunctionHotineInf;


template<>
long double
CostFunctionHotine<long double>::sum(const std::vector<long double>& x)
{
	return recursive_sum<long double>(x);
}

template<>
CostHotine<double>::operator double() const
{
	return cost;
}

template<>
CostHotine<long double>::operator long double() const
{
	return cost;
}





/***************************************************************************
 *                          Cost with p=infty                              *
 ***************************************************************************/
