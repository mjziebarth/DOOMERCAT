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
HotineObliqueMercator(lon=None, lat=None, weight=None, pnorm=2, k0_ap=0.98,
                      sigma_k0=0.002, ellipsoid=None, f=None, a=None,
                      lonc0=None, lat_00=None, alpha0=None, k00=None, lonc=None,
                      lat_0=None, alpha=None, k0=None, Nmax=1000, proot=False,
                      logger=None, backend='C++', fisher_bingham_use_weight=False,
                      compute_enclosing_sphere=False, bfgs_epsilon=1e-3)
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


### Unreleased
#### Changed
- QGIS Plugin: Change some type info to be compatible with Python versions before 3.10
- QGIS Plugin: Fallback `WktVariant` for QGIS versions before 3.36

### [1.0.0] - 2024-08-30
#### Added
- First release; begin of the versioning.