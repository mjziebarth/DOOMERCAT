/*
 * Optimization routines.
 */

#include <memory>
#include <../include/labordecylinder.hpp>
#include <../include/projecteddata.hpp>
#include <../include/cost.hpp>

#ifndef DOOMERCAT_OPTIMIZE_H
#define DOOMERCAT_OPTIMIZE_H

namespace doomercat {

struct result_t {
	double cost;
	LabordeCylinder cylinder;
};

std::vector<result_t>
billo_gradient_descent(std::shared_ptr<const DataSet> data,
                       std::shared_ptr<const LabordeCylinder> cyl0,
                       const CostFunction& fun, size_t Nmax);

std::vector<result_t>
bfgs_optimize(std::shared_ptr<const DataSet> data,
              std::shared_ptr<const LabordeCylinder> cyl0,
              const CostFunction& cost_function,
              const size_t Nmax);

}
#endif
