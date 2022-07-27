/*
 * Type definitions for common automatically differentiating doubles.
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

#include <../include/autodouble.hpp>

#ifndef AUTODOUBLE_TYPES_H
#define AUTODOUBLE_TYPES_H

typedef autodouble<1> real1v;
typedef std::array<real1v,3> vec3_1v;

typedef autodouble<2> real2v;
typedef std::array<real2v,3> vec3_2v;

typedef autodouble<3> real3v;
typedef std::array<real3v,3> vec3_3v;

typedef autodouble<4> real4v;
typedef std::array<real4v,3> vec3_4v;

typedef autodouble<5> real5v;
typedef std::array<real5v,3> vec3_5v;

#endif
