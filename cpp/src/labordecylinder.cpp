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
#include <../include/linalg.hpp>


using doomercat::LabordeCylinder;

/*
 *  Functions:
 */

real5v lambda_difference(real5v&& a, const real5v& b)
{
	/* Returns the difference in latitude between a and b within the range
	 * of 90 arcdegrees. */
	const double aval = a.value();
	const double bval = b.value();
	if (aval > bval){
		const double alt0 = aval - bval;
		const double alt1 = bval + 2*PI - aval;
		if (alt0 <= alt1)
			return a - b;
		return (2*PI - a) + b;
	} else {
		const double alt0 = bval - aval;
		const double alt1 = aval + 2*PI - bval;
		if (alt0 <= alt1)
			return (-a) + b;
		return (a + 2*PI) - b;
	}
}

double lambda_difference(double a, const double b)
{
	/* Returns the difference in latitude between a and b within the range
	 * of 90 arcdegrees. */
	if (a > b){
		const double alt0 = a - b;
		const double alt1 = b + 2*PI - a;
		return std::min(alt0,alt1);
	} else {
		const double alt0 = b - a;
		const double alt1 = a + 2*PI - b;
		return std::min(alt0,alt1);
	}
}



real5v compute_fies(const vec3_5v& ax, const real5v& lambda_c)
{
	return -atan(cos(lambda_difference(atan2(ax[1],ax[0]), lambda_c))
	             * sqrt(1.0 - ax[2]*ax[2]) / ax[2]);
}

double compute_fies(const ColVector<3, double>& ax, const double lambda_c)
{
	return -std::atan(std::cos(lambda_difference(std::atan2(ax[1],ax[0]),
	                                             lambda_c))
	             * std::sqrt(1.0 - ax[2]*ax[2]) / ax[2]);
}



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
	return log(std::move((1.0 - e * sv) / (1.0 + e * sv)));
}

static void print_lola(double x, double y, double z)
{
	const double nrm = std::sqrt(x*x + y*y + z*z);
	const double lon = rad2deg(std::atan2(y,x));
	const double lat = rad2deg(std::asin(z/nrm));
	std::cout << "lon=" << lon << ", lat=" << lat;
}


static Quaternion<real5v>
quaternion_from_axis_and_lonc(double lon_cyl, double lat_cyl, double lonc)
{
	/* Compute an initial guess of the quaternion from a cylinder axis
	 * and the location of the central point (given through lonc). */
	typedef ColVector<3, double> vec3;

	/* Cylinder axis: */
	const double lambda_cyl = deg2rad(lon_cyl);
	const double phi_cyl = deg2rad(lat_cyl);
	const double lambda_c = deg2rad(lonc);
	const vec3 ax_cyl({std::cos(lambda_cyl) * std::cos(phi_cyl),
	                   std::sin(lambda_cyl) * std::cos(phi_cyl),
	                   std::sin(phi_cyl)});

	/* Compute fies: */
	const double fies = compute_fies(ax_cyl, lambda_c);

	/* Axis pointing to (lonc,fies): */
	const vec3 ax_center({std::cos(lambda_c) * std::cos(fies),
	                      std::sin(lambda_c) * std::cos(fies),
	                      std::sin(fies)});

	/* Now the rotation angle to rotate to the cylinder from (0,0,1) to
	 * its position (lon_cyl, lat_cyl): */
	const double ra_cyl = 0.5*PI - phi_cyl;
	const double sra = std::sin(0.5*ra_cyl);
	const Quaternion<double> q0( std::cos(0.5*ra_cyl),
	                            -sra * std::sin(lambda_cyl),
	                             sra * std::cos(lambda_cyl),
	                             0.0);

	/* Now quaternion to rotate, before, the cylinder around its symmetry axis
	 * to rotate (lonc, fies) back to (1,0,0): */
	ImaginaryQuaternion<double> central(std::cos(lambda_c) * std::cos(fies),
	                                    std::sin(lambda_c) * std::cos(fies),
	                                    std::sin(fies));

	/* Rotate back: */
	ImaginaryQuaternion<double> central_back(central.rotate(q0));

	/* Now we can easily read the rotation angle: */
	const double rot_angle_2 = std::atan2(central_back.j(), central_back.i());

	/* Compute the quaternion as double: */
	const Quaternion<double>
	    res(q0 * Quaternion<double>(std::cos(0.5*rot_angle_2),
	                                0.0, 0.0, std::sin(0.5*rot_angle_2)));

	return Quaternion<real5v>(variable5<0>(res.r()), variable5<1>(res.i()),
	                          variable5<2>(res.j()), variable5<3>(res.k()));
}





/*
 *   The class implementation:
 */


LabordeCylinder::LabordeCylinder(const Quaternion<real5v>& q_, const real5v& k0,
                                 const double f_)
    : q(q_), _k0(k0), _f(f_), e2(f_*(2.0 - f_)), _e(std::sqrt(e2)),
      _axis(compute_axis(q)),
      _central_axis(compute_central(q)),
      _lambda_c(atan2(_central_axis[1], _central_axis[0])),
      fies(compute_fies(_axis, _lambda_c)),
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


LabordeCylinder::LabordeCylinder(double lon_cyl, double lat_cyl, double lonc,
                                 double k0, double f)
    : LabordeCylinder(quaternion_from_axis_and_lonc(lon_cyl, lat_cyl, lonc),
                      variable5<4>(k0), f)
{
}


vec3_5v LabordeCylinder::compute_axis(const Quaternion<real5v>& q)
{
	//const  qcyl(0,0,0,1);

	// Norm the rotation quaternion:
	const Quaternion<real5v> qn = q / q.norm();

	// Rotate the cylinder and return its axis:
	// axis = qrot(qn, Quaternion<double>(0,0,0,1), qn.conj()).imag()
	const vec3_5v axis(ImaginaryQuaternion(0,0,1).rotate(qn.conj()).imag());
	const real5v norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] +
	                          axis[2]*axis[2]);
	return vec3_5v({axis[0] / norm, axis[1] / norm, axis[2] / norm});
}

vec3_5v LabordeCylinder::compute_central(const Quaternion<real5v>& q)
{
	const Quaternion<double> qc(0,1,0,0);

	// Norm the rotation quaternion:
	const Quaternion<real5v> qn = q / q.norm();

	// Rotate the cylinder and return its axis:
	return qrot(qn, qc, qn.conj()).imag();
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
		return -rad2deg(azimuth);

	return rad2deg(azimuth);
}

real5v LabordeCylinder::lat_0() const
{
	return rad2deg(vize);
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

const real5v& LabordeCylinder::lambda_c() const
{
	return _lambda_c;
}

real5v LabordeCylinder::lonc() const
{
	return rad2deg(_lambda_c);
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


double LabordeCylinder::k0_correction() const
{
    const double cv = std::cos(vize.value());
    const double sv = std::sin(vize.value());
    const double d0 = (1.0-_f)*sv;
    const double d1 = (1.0-_f)*d0;
    return std::sqrt((cv*cv + d1*d1) / ((cv*cv + d0*d0)));
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
LabordeCylinder::from_axis_and_central(double lon_cyl, double lat_cyl,
                                       double lonc, double k0, double f)
{
	return std::make_shared<LabordeCylinder>(lon_cyl, lat_cyl, lonc, k0, f);
}


std::shared_ptr<LabordeCylinder>
LabordeCylinder::from_parameters(double qr, double qi, double qj, double qk,
                                 double k0, double f)
{
	return std::make_shared<LabordeCylinder>(qr, qi, qj, qk, k0, f);
}
