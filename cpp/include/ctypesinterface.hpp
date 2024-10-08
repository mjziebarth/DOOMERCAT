/*
 * Methods suitable for interfacing with Ctypes.
 * This file is part of the DOOMERCAT python module.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
 *               2024 Technische Universität München
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

int compute_cost_hotine_batch(const size_t N, const double* lon,
        const double* lat, const double* h, const double* w, const size_t M,
        const double* lonc, const double* lat0, const double* alpha,
        const double* k0, double a, double f, double pnorm, double k0_ap,
        double sigma_k0, unsigned short logarithmic,
        unsigned short wrap_plane, unsigned short precision,
        double* result);

int compute_cost_gradient_hotine_batch(const size_t N, const double* lon,
        const double* lat, const double* h, const double* w, const size_t M,
        const double* lonc, const double* lat0, const double* alpha,
        const double* k0, double a, double f, double pnorm, double k0_ap,
        double sigma_k0, unsigned short logarithmic,
        unsigned short wrap_plane,
        double* result);

int compute_k_hotine(const size_t N, const double* lon,
        const double* lat, const double* w,
        double lonc, double lat0, double alpha, double k0, double f,
        double* result);

int hotine_project_uv(const size_t N, const double* lon,
        const double* lat, double lonc, double lat0, double alpha,
        double k0, double f, double* result);

int hotine_project(const size_t N, const double* lon,
        const double* lat, double lonc, double lat0, double alpha,
        double k0, double gamma, double f, double* result);

int hotine_inverse(const size_t N, const double* x,
        const double* y, double lonc, double lat0, double alpha,
        double k0, double gamma, double f, double* result);

int hotine_damped_bfgs(
    const size_t N, const double* lon, const double* lat,
    const double* h, const double* w, double a, double f,
    double pnorm, double k0_ap, double sigma_k0, double lonc_0,
    double lat_0_0, double alpha_0, double k_0_0, unsigned int Nmax,
    double epsilon, double* result,
    unsigned int* n_steps, uint64_t* n_fun_eval
);

int hotine_backtrack_GD(
    const size_t N, const double* lon, const double* lat,
    const double* h, const double* w, double a, double f,
    double pnorm, double k0_ap, double sigma_k0, double lonc_0,
    double lat_0_0, double alpha_0, double k_0_0, unsigned int Nmax,
    double epsilon, double* result,
    unsigned int* n_steps,uint64_t* n_fun_eval);

int hotine_parameters_debug(double lonc, double lat0, double alpha,
                            double k0, double f, double* result);

}
