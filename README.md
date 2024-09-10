# DOOMERCAT
The Data-Optimized Oblique MERCATor (DOOMERCAT) module implements an
optimization routine to determine the parameters of an oblique
Mercator projection based on a spatial data set. The optimization
minimizes the distortion at the points of the data set.

This repository contains both the ```doomercat``` python module and
the DOOMERCAT plugin for QGIS. To install the ```doomercat``` python module,
execute ```pip install .``` (or your python installation command
of choice) inside the repository root folder. The QGIS plugin
can be built by executing the ```generate-plugin.sh``` script
with a bash shell.

## Mathematical background
Minimizing the distortion for the data set is realized by minimizing
a cost function that penalizes deviations of the local scale factor
from one. The norm parameter ```pnorm``` adjusts how strongly extreme
residuals are weighted, with low values minimizing average distortion
and high values minimizing maximum distortion. Details about the
optimization method are documented in the accompanying paper [1].

## The ```doomercat``` python module
The module provides

 - the ```HotineObliqueMercator``` class
 - the ```OptimizeError``` class
 - ... and more.

The ```HotineObliqueMercator``` class bundles the module's
functionality. It can be constructed from a set of longitude
and latitude coordinates ```lon``` and ```lat``` by calling
```python
HOM = HotineObliqueMercator(lon=lon, lat=lat, pnorm=2)
```
which optimizes the parameters ```lonc```, ```lat_0```, ```alpha```,
and ```k_0``` of the Hotine oblique Mercator projection for
the data set using a quadratic cost function and initializes itself
accordingly. Internally, the class constructor calls either the
```grad``` method from the ```doomercat.hotine``` submodule or the
```bfgs_optimize``` method from the ```doomercat.cppextensions```
submodule for the optimization. Afterwards,
```python
HOM.proj4_string()
```
can be called to construct a PROJ projection string using the
```omerc``` projection.

If the optimization fails, e.g. by producing illegal values for the parameters,
an instance of ```OptimizeError``` is raised. From it,
the parameter results of the failed optimization can be obtained.

### Call Signatures
```python
HotineObliqueMercator(
    # Specify the data (optionally with height and weight):
    lon=None,
    lat=None,
    h=None,
    weight=None,
    # Cost function exponent (pnorm) and k_0 prior:
    pnorm=2,
    k0_ap=0.98,
    sigma_k0=0.002,
    # Datum:
    ellipsoid=None,
    f=None,
    a=None,
    # Technical algorithm costraints:
    Nmax=1000,
    Nmax_pre_adamax=50,
    backend='Python',
    bfgs_epsilon=1e-3,
    # Further arguments:
    compute_enclosing_sphere=False,
    fisher_bingham_use_weight=False
)
```

## The DOOMERCAT QGIS plugin
The python code is available as a QGIS plugin. The plugin can be built on
linux by executing the ```generate-plugin.sh``` shell script. This will
create the file ```build/DOOMERCAT.zip``` that can be installed in the QGIS
plugin menu.

The plugin is developed for QGIS version 3.38.1.

## License
The ```doomercat``` package is licensed under the European Public License (EUPL)
version 1.2 or later. The DOOMERCAT QGIS plugin, both the code contained in
the ```qgis-plugin``` folder and the combined (with the python plugin)
derivative work created by the ```generate-plugin.sh```, is licensed under the
GNU General Public License (GPL), version 3 or (at your option) later.

## References
[1] von Specht, Ziebarth (in prep.)


## Changelog
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [2.0.4] - 2024-09-09
#### Changed
- Various small import-related improvements for the QGIS plugin.

### [2.0.3] - 2024-09-09
#### Changed
- Fix the Schnellschuss and actually fall back to `Any`.

### [2.0.2] - 2024-09-09
#### Changed
- Update typing to rely on `numpy.typing.NDArray` in a central position and
  fall back to `Any`.


### [2.0.1] - 2024-09-05
#### Changed
- Modified the `autodouble::sqrt` methods to default to zero derivative at
  $x=0$. This fixes some difficulties for the special starting values with
  $\phi_0=\lambda_c=0$.

### [2.0.0] - 2024-09-04
#### Changed
- Remove the `proot` parameter. The cost function has been reworked and now
  uses the *p*th root by default.

### [1.0.3] - 2024-09-02
#### Added
- Add version to `qgis-plugin/metadata.txt`.

### [1.0.2] - 2024-09-02
#### Added
- Add source code archive to QGIS plugin.

### [1.0.1] - 2024-09-02
#### Added
- Added workflow steps to build the source distribution and a pure Python package.

#### Changed
- QGIS Plugin: Change some type info to be compatible with Python versions before 3.10
- QGIS Plugin: Fallback `WktVariant` for QGIS versions before 3.36
- Changed workflow triggers: build full release only on tag. On pull requests and
  pushes, do only a minimal test build on Ubuntu/Windows.

### [1.0.0] - 2024-08-30
#### Added
- First release; begin of the versioning.