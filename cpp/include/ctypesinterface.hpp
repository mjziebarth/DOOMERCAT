/*
 * Methods suitable for interfacing with Ctypes.
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

using std::size_t;

extern "C" {
int compute_cost(const size_t N, const double* lon, const double* lat,
                 const double* w, double lon_cyl, double lat_cyl, double k0,
                 double f, unsigned int pnorm, double k0_ap, double sigma_k0,
                 double* result);

int compute_cost_and_derivatives(const size_t N, const double* lon,
        const double* lat, const double* w, double lon_cyl, double lat_cyl,
        double lonc, double k0, double f, unsigned int pnorm, double k0_ap,
        double sigma_k0, double* result);

int perform_billo_gradient_descent(const size_t N, const double* lon,
        const double* lat, double f,
        unsigned int pnorm, double k0_ap, double sigma_k0,
        double lon0, double lat0, const size_t Nmax, double* result);

int perform_bfgs(const size_t N, const double* lon, const double* lat,
                 const double* w, double f, unsigned int pnorm, double k0_ap,
                 double sigma_k0, double lon_cyl0, double lat_cyl0,
                 double lonc0, double k00, const size_t Nmax, double* result,
                 unsigned int* n_steps, double* final_cylinder);


int perform_adam(const size_t N, const double* lon, const double* lat,
                 const double* w, double f, unsigned int pnorm, double k0_ap,
                 double sigma_k0, double lon_cyl0, double lat_cyl0,
                 double lonc0, double k00, const size_t Nmax, double* result,
                 unsigned int* n_steps, double* final_cylinder);

int laborde_project(size_t N, const double* lon, const double* lat,
                    double qr, double qi, double qj, double qk, double k0,
                    double f, double a, double* xy);

int compute_cost_k0_iter(const size_t N, const double* lon,
        const double* lat, const double* w, const size_t M,
        const double* k0, double qr, double qi, double qj, double qk,
        double f, unsigned int pnorm, double k0_ap, double sigma_k0,
        double* cost_result);

}
