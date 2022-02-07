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

#include <cmath>
#include <iostream>

#include <../include/labordecylinder.hpp>
#include <../include/functions.hpp>

using doomercat::LabordeCylinder;

/*
 *  Functions:
 */

real5v compute_vize(const real5v& fies, const double e2)
{
	if (e2 == 0.0)
		return fies;

	const double e4 = e2*e2;
	const double e6 = e4*e2;

	const real5v D = pow(sin(fies),2);
	const real5v D2 = D*D;
	const real5v D3 = D2*D;
	const real5v D4 = D3*D;
	const real5v D5 = D4*D;
	const real5v D6 = D5*D;
	const real5v D7 = D6*D;

	/* The full equation:
	 * A = (1 - e2) / (e2 * pow(sin(fies),2));
	 * y = (2+A)/2 - sqrt(pow((2+A)/2,2) - 1/e2);
	 */
	const real5v y(min(D + (D3 - 2*D2 + D) * e2
	                      + (2*D5 - 6*D4 + 7*D3 - 4*D2 + D) * e4 \
	                      + (5*D7 - 20 * D6 + 35 * D5 - 32 * D4 + 18*D3
	                         - 6 * D2 + D) * e6,
	                   1.0));

	if (fies.value() >= 0.0)
		return asin(sqrt(y));
	else
		return -asin(sqrt(y));
}


real5v Gd_inv(const real5v& x) {
	return atanh(sin(x));
}


static real5v log_1mes_1pes(const real5v& vize, double e)
{
	const real5v sv(sin(vize));
	return log((1.0 - e * sv) / (1.0 + e * sv));
}


/*
 *   The class implementation:
 */


LabordeCylinder::LabordeCylinder(const Quaternion<real5v>& q_, const real5v& k0,
                                 const double f_)
    : q(q_), _k0(k0), _f(f_), e2(f_*(2.0 - f_)), _e(std::sqrt(e2)),
      _axis(compute_axis(q)), fies(-atan(_axis[0] / _axis[2])),
      vize(compute_vize(fies, e2)),
      _B(sqrt(1.0 + (e2 / (1.0 - e2)) * pow(cos(vize), 4))),
      _C(Gd_inv(fies) - _B*(Gd_inv(vize) + 0.5 * _e * log_1mes_1pes(vize, _e)))
{
}


LabordeCylinder::LabordeCylinder(double qr, double qi, double qj, double qk,
                                 double k0, double f)
    : LabordeCylinder(Quaternion<real5v>(variable5<0>(qr), variable5<1>(qi),
                                         variable5<2>(qj), variable5<3>(qk)),
                      variable5<4>(k0), f)
{
}

vec3_5v LabordeCylinder::compute_axis(const Quaternion<real5v>& q)
{
	const Quaternion<double> qcyl(0,0,0,1);

	// Norm the rotation quaternion:
	const Quaternion<real5v> qn = q / q.norm();

	// Rotate the cylinder and return its axis:
	return qrot(qn, qcyl, qn.conj()).imag();
}


real5v LabordeCylinder::azimuth() const
{
	/* We need the sine of the latitude of the cylinder axis,
	 * which is the cylinder axis z component:
	 */
	const real5v sin_phi_cyl = _axis[2];

	/* Furthermore, we need the cosine of the central coordinate
	 * latitude:
	 */
	const real5v cos_fies = cos(fies);

	/* Now, we can compute the azimuth of the cylinder equator at the
	 * central coordinate using spherical trigonometry on the Laborde
	 * sphere:
	 */
	real5v azimuth = asin(max(min(sin_phi_cyl / cos_fies, 1.0), -1.0));

	/* The spherical geometry used does not consider the correct
	 * sign of the azimuth. Thus, we may have to multiply by -1.
	 * This can be decided by considering the cross product
	 * of the cylinder axis and the central axis:
	 */
	if (_axis[1].value() > 0)
		return -azimuth;

	return azimuth;
}

const real5v& LabordeCylinder::k0() const
{
	return _k0;
}

const real5v& LabordeCylinder::B() const
{
	return _B;
}

const real5v& LabordeCylinder::C() const
{
	return _C;
}

double LabordeCylinder::e() const
{
	return _e;
}

double LabordeCylinder::f() const
{
	return _f;
}

const Quaternion<real5v>& LabordeCylinder::rotation_quaternion() const
{
	return q;
}

vec3_5v LabordeCylinder::axis() const
{
	return _axis;
}


std::shared_ptr<LabordeCylinder>
LabordeCylinder::from_axis_lon_lat(double lon, double lat, double k0, double f)
{
	/* Computes the cylinder whose axis points to a specific point given
	 * in longitude and latitude.
	 *
	 * Both lon and lat are expected in arcdegrees.
	 */
	lon = deg2rad(lon);
	lat = deg2rad(lat);

	/* Step one: create the rotation axis that rotates an axis through
	 * the North pole to (lon, lat):
	 */
	// vec3_5v rot_ax = vec3_5v({-sin(lon), cos(lon), 0.0});

	/* The angle then depends only on latitude: */
	const double rot_angle = 0.5*PI - lat;

	/* Now we can create the rotation quaternion: */
	const double sra = std::sin(0.5*rot_angle);

	// variable5<n>(x) initializes for n<4 the variable q_n = x
	Quaternion<real5v> q(variable5<0>(std::cos(0.5*rot_angle)),
	                     variable5<1>(-sra * std::sin(lon)),
	                     variable5<2>(sra * std::cos(lon)),
	                     variable5<3>(0.0));

	return std::make_shared<LabordeCylinder>(q, variable5<4>(k0), f);
}

std::shared_ptr<LabordeCylinder>
LabordeCylinder::from_parameters(double qr, double qi, double qj, double qk,
                                 double k0, double f)
{
	return std::make_shared<LabordeCylinder>(qr, qi, qj, qk, k0, f);
}
