/*
 * Optimization routines.
 */

#include <memory>
#include <../include/labordecylinder.hpp>
#include <../include/projecteddata.hpp>
#include <../include/cost.hpp>
#include <../include/cost_hotine.hpp>
#include <../include/hotine.hpp>

#ifndef DOOMERCAT_OPTIMIZE_H
#define DOOMERCAT_OPTIMIZE_H

namespace doomercat {

enum class return_state_t {
	CONVERGED=0,
	MAX_ITERATIONS=1,
	ERROR=2
};

struct result_t {
	double cost;
	LabordeCylinder cylinder;
	return_state_t state;
};

struct hotine_result_t {
	double cost;
	double lonc;
	double lat_0;
	double alpha;
	double k0;
	double grad_lonc;
	double grad_lat0;
	double grad_alpha;
	double grad_k0;
	return_state_t state;
	unsigned int algorithm_state;
};


std::vector<result_t>
billo_gradient_descent(std::shared_ptr<const DataSet> data,
                       std::shared_ptr<const LabordeCylinder> cyl0,
                       const CostFunction& fun, size_t Nmax);

std::vector<result_t>
bfgs_optimize(std::shared_ptr<const DataSet> data,
              std::shared_ptr<const LabordeCylinder> cyl0,
              const CostFunction& cost_function,
              const size_t Nmax, bool transform_k0=false);

std::vector<result_t>
adam_optimize(std::shared_ptr<const DataSet> data,
              std::shared_ptr<const LabordeCylinder> cyl0,
              const CostFunction& cost_function,
              const size_t Nmax);

std::vector<hotine_result_t>
bfgs_optimize_hotine(const DataSet& data, const double lonc0,
                     const double lat_00, const double alpha0,
                     const double k00, const double f,
                     const unsigned int pnorm, const double k0_ap,
                     const double sigma_k0,
                     const size_t Nmax, bool transform_k0);

}
#endif
