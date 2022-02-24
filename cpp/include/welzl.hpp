/*
 * Welzl (1991)'s algorithm for determining the smallest enclosing
 * circle.
 */

#ifndef DOOMERCAT_WELZL_HPP
#define DOOMERCAT_WELZL_HPP

#include <vector>

namespace doomercat {

struct xy_t {
	double x;
	double y;
};

struct disk_t {
	double x;
	double y;
	double r2;
};

disk_t welzl1991(const std::vector<xy_t>&, double eps=1e-12);

};

#endif
