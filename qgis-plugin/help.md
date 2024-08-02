## Data-Optimized Oblique MERCATor
The Data-Optimized Oblique MERCATor (DOOMERCAT)
plugin optimizes the parameters of an oblique
Mercator projection based on a spatial data set.
It minimizes the distortion at the points of
the data set according to a cost function that
effectively determines how the distortion at a
single point should be balanced with the
distortion of the other points.

For detailed information see the accompanying
paper [1].

### Parameters
The **cost function exponent** determines how
the values of map distortion at the data points
are weighted in the cost function
which is minimized. A value of 2 leads to a
simple least-squares solution of the
residuals. A large value, up to a maximum of
99, leads to a cost function that is more
similar to a maximum absolute deviation. In
other words, a value of 2 leads to a
least-squares compromise in projection error
while the largest values resemble a
minimization of the maximum distortion.
If the **is infinite** box is checked, the
exponent is positive infinity. This means that
the maximum absolute distortion across all data
points is minimized.

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

The **ellipsoid** row specifies the ellipsoid
according to which geographic coordinates
will internally be projected to 3D Euclidean
space.

The **optimizer** row specifies the internal
backend used to optimize the parameters of the
projection. The default, `'Python'`, uses a
Levenberg-Marquardt or Adam solver, depending
on the *p*-norm. The alternative, `'C++'`, is
available if the plugin has been delivered with
compiled code suitable for this machine. It
uses a BFGS solver. Both optimizers aim to
minimize the same cost function and should not
differ much in the resulting quality. If a
difficulty occurs in one of the solvers, using
the alternative might lead to a better solution.

Next, the tabs switch between mechanisms
to input the point set used in the
optimization. **Selection** collects the
points currently selected and allows for a
simple manual selection. **Weighted raster**
allows using all points of a raster layer
weighted by the absolute value a selectable
raster band. **Weighted points** provides the
same mechanism for vector layers.

The next row, **Use data height**, toggles
whether the height of the data points is
corrected for in the cost function. Points
located above or below the reference ellipsoid,
for instance in the deep sea or on mountain
tops, are effectively located on an ellipsoid
of different scale that corresponds to another
map scale. If this difference is large, this
elevation-induced scale change can lead to
errors in map scale that are comparable to the
oblique Mercator distortion. Toggling the check
box will apply a local scale correction at each
data point to balance both sources of
distortion. The height above the reference
ellipsoid will be read from the *z* coordinate
of vector layer points. Note that this check
box does not check a prior whether valid
*z* coordinates can be read from the layer;
failure to do so will show an error message once
the optimization starts.

The **orient north** row allows to specify one
point on the map which is rotated so that up
is north. The point can be the center of the
data set, the center of the optimized projection
(which may lie outside the data set), or a
manually chosen point.

After optimization, the **projection string**
line is filled with the PROJ string
representation of the resulting projection. A
user CRS can be created from this string
using the **save** button and, afterwards,
this CRS can be **apply**-ed to the current
project. Unchecking the **orient north** box
adds the necessary flags that cause PROJ to
return the raw *u*,*v* coordinates in the
oblique Mercator projection. That is, the
oblique Mercator equator is horizontal in the
resulting projection. If the box is checked,
an affine transformation is performed that
ensures that North is upwards at the
projection center.

[1] von Specht, Ziebarth, Veh (in prep.)
