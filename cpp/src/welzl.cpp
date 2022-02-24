



#include <../include/welzl.hpp>
#include <stdexcept>
#include <random>
#include <algorithm>


using doomercat::xy_t;
using doomercat::disk_t;
using doomercat::welzl1991;

static disk_t minidisk_2(const xy_t& x0, const xy_t& x1, double eps)
{
	/* Computes the minidisk between two points. */
	double dx = x0.x - x1.x;
	double dy = x0.y - x1.y;
	/* r = 0.5 * sqrt(dx*dx + dy*dy)
	 * -> r2 = 0.25 * (dx*dx + dy*dy) */
	double xc = 0.5*(x0.x + x1.x);
	double yc = 0.5*(x0.y + x1.y);
	return {xc, yc, 0.25*(dx*dx + dy*dy) * (1.0 + eps)};
}

static disk_t minidisk_3(const xy_t& p1, const xy_t& p2, const xy_t& p3,
                         double eps)
{
	/* Follows Ratliff, Hunter N. (2019): Cartesian formulas for
	 * curvature, circumradius, and circumcenter for any three
	 * two-dimensional points.
	 * https://doi.org/10.5281/zenodo.2556424
	 */
	double A = 0.5 * (  p1.x*p2.y - p2.x*p1.y + p2.x*p3.y - p3.x*p2.y
	                  + p3.x*p1.y - p1.x*p3.y);
	double x = -( (p3.y-p1.y)*(p2.y-p1.y)*(p3.y-p2.y)
	             - (p2.x*p2.x - p1.x*p1.x)*(p3.y-p2.y)
	             + (p3.x*p3.x - p2.x*p2.x)*(p2.y-p1.y)) / (4*A);
	double y = - (p2.x - p1.x) * x / (p2.y - p1.y)
	           + (p2.x*p2.x - p1.x*p1.x + p2.y*p2.y - p1.y*p1.y)
	             / (2*(p2.y - p1.y));
	double r2 = (  (x-p1.x)*(x-p1.x) + (y-p1.y)*(y-p1.y)
	             + (x-p2.x)*(x-p2.x) + (y-p2.y)*(y-p2.y)
	             + (x-p3.x)*(x-p3.x) + (y-p3.y)*(y-p3.y)) / 3;
	return {x, y, r2 * (1.0 + eps)};
}


static bool in_minidisk(const xy_t& x, const disk_t& D)
{
	double dx = x.x - D.x;
	double dy = x.y - D.y;
	return dx*dx + dy*dy <= D.r2;
}


static disk_t b_md(const std::vector<xy_t>& R, double eps)
{
	/* Compute the minidisk for the cases where it is trivially
	 * possible. */
	if (R.size() == 1)
		return {R[0].x, R[0].y, eps};
	else if (R.size() == 2)
		return minidisk_2(R[0], R[1], eps);
	else if (R.size() == 3){
		/* propose the three options which are defined by
		 * only two points: */
		disk_t D = minidisk_2(R[0], R[1], eps);
		if (in_minidisk(R[2], D))
			return D;
		D = minidisk_2(R[0], R[2], eps);
		if (in_minidisk(R[1], D))
			return D;
		D = minidisk_2(R[1], R[2], eps);
		if (in_minidisk(R[0],D))
			return D;
		/* Now the circumcircle option: */
		return minidisk_3(R[0], R[1], R[2], eps);
	}
	throw std::runtime_error("R.size() > 3 or R.size() == 0.");
}

typedef typename std::vector<xy_t>::const_iterator iter_t;

static disk_t b_minidisk(const iter_t& P0, iter_t P1,
                         std::vector<xy_t>& R, double eps)
{
	/* Welzl's algorithm subroutine to compute the minimum
	 * enclosing disk subject to a set of points on the circle. */
	if (std::distance(P0,P1) == 0 || R.size() == 3)
		return b_md(R, eps);

	/* Pop last element of P: */
	--P1;
	const xy_t p = *P1;
	disk_t D(b_minidisk(P0, P1, R, eps));
	if (!in_minidisk(p, D)){
		R.push_back(p);
		D = b_minidisk(P0, P1, R, eps);
		R.pop_back();
	}
	return D;
}


static disk_t minidisk(const iter_t& P0, iter_t P1, double eps)
{
	if (distance(P0,P1) == 0)
		return {0.0, 0.0, -1.0};
	/* Remove the last point of P: */
	--P1;
	const xy_t p = *P1;
	disk_t D(minidisk(P0, P1, eps));

	if (!in_minidisk(p, D)){
		std::vector<xy_t> R;
		R.push_back(p);
		D = b_minidisk(P0, P1, R, eps);
	}
	return D;
}


disk_t doomercat::welzl1991(const std::vector<xy_t>& xy, double eps)
{
	if (xy.size() == 0)
		return {0.0, 0.0, -1.0};

	/* Shuffle the data: */
	std::vector<xy_t> xy2(xy);
	std::shuffle(xy2.begin(), xy2.end(), std::default_random_engine());

	/* Call algorithm: */
	return minidisk(xy2.begin(), xy2.end(), eps);
}
