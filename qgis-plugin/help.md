## Data-Optimized Oblique MERCATor
### Parameters
The **cost function exponent** determines how
the residual distance between the points on
the ellipsoid and the projected points on the
cylinder are weighted in the cost function
which is minimized. A value of 2 leads to a
simple least-squares solution of the
residuals. A large value, up to a maximum of
99, leads to a cost function that is more
similar to a maximum absolute deviation. In
other words, a value of 2 leads to a
least-squares compromise in projection error
while the largest values resemble a
minimization of the maximum distortion.

The **minimum k0 constraint** (k0min) gives a
soft lower bound on the scale k0 at the
oblique equator. This prevents the algorithm
from converging to small-circle solutions,
that is, solutions in which the data points
are located on small circle intersection of
the cylinder with the ellipsoid, far off the
oblique equator. The cost function is
augmented by a half-sided quadratic potential
for k0 < k0min **with bandwidth** as
given.

If **select homogeneous** is checked, a point
selection algorithm is performed before the
optimization starts. The selection algorithm
successively removes one point of the closest
point pair until only the provided number of
data points remains. This should lead to a
fairly homogeneous data set, a considerable
speedup for large data sets (such as large
raster layers), and can have beneficial
effects on the cost function if large point
clouds lead to undesirable results.

The **ellipsoid** row specifies the ellipsoid
according to which geographic coordinates
will internally be projected to 3D Euclidean
space.

Next, the tabs switch between mechanisms
to input the point set used in the
optimization. **Selection** collects the
points currently selected and allows for a
simple manual selection. **Shapefile** allows
loading from those. Either *all points* or
only the *centroids of geometries* can be fed
to the optimization. **Weighted raster**
allows using all points of a raster layer
weighted by the absolute value a selectable
raster band. **Weighted points** provides the
same mechanism for vector layers.

After optimization, the **projection string**
line is filled with the PROJ string
representation of the resulting projection. A
user CRS can be created from this string
using the **save** button and, afterwards,
this CRS can be **apply**-ed to the current
project. Unchecking the **orient North** box
adds the necessary flags that cause PROJ to
return the raw *u*,*v* coordinates in the
oblique Mercator projection. That is, the
oblique Mercator equator is horizontal in the
resulting projection. If the box is checked,
an affine transformation is performed that
ensures that North is upwards at the
projection center.
