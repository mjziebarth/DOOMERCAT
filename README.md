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
a cost function that penalizes residuals between original coordinates in
three-dimensional space and projected coordinates on the oblique
cylinder. The norm parameter ```pnorm``` adjusts how strongly extreme
residuals are weighted, with low values minimizing average distortion
and high values minimizing maximum distortion. Details about the
optimization method are documented in the accompanying paper [1].

## The ```doomercat``` python module
The module provides

 - the ```LabordeObliqueMercator``` class
 - the ```levenberg_marquardt``` function
 - the ```points_from_shapefile``` function
 - the ```OptimizeError``` class

The ```LabordeObliqueMercator``` class bundles the module's
functionality. It can be constructed from a set of longitude
and latitude coordinates ```lon``` and ```lat``` by calling
```python
LOM = LabordeObliqueMercator(lon=lon, lat=lat, pnorm=2)
```
which optimizes the parameters ```lonc```, ```lat_0```, ```alpha```,
and ```k_0``` of the Laborde oblique Mercator projection for
the data set using a quadratic cost function and initializes itself
accordingly. Internally, the class constructor calls the
```levenberg_marquardt``` method for the optimization. Afterwards,
```python
LOM.proj4_string()
```
can be called to construct a PROJ projection string using the
```omerc``` projection.

If the optimization fails, e.g. by producing illegal values for the parameters,
an instance of ```OptimizeError``` is raised. From it,
the parameter results of the failed optimization can be obtained.

### Call Signatures
```python
LabordeObliqueMercator(lon=None, lat=None, weight=None, pnorm=2, k0_ap=0.98,
                       sigma_k0=0.02, initial_lambda=10, nu=0.99,
                       lbda_min=1e-10, lbda_max=1e10, ellipsoid='WGS84',
                       use='all', f=None, a=None, lonc=None, lat_0=None,
                       alpha=None, k0=None, logger=None)
```

## The DOOMERCAT QGIS plugin
The python code is available as a QGIS plugin. The plugin can be built on
linux by executing the ```generate-plugin.sh``` shell script. This will
create the file ```build/DOOMERCAT.zip``` that can be installed in the QGIS
plugin menu.

The plugin is developed for QGIS version 3.20.0.

## License
The ```doomercat``` package is licensed under the European Public License (EUPL)
version 1.2 or later. The DOOMERCAT QGIS plugin, both the code contained in
the ```qgis-plugin``` folder and the combined (with the python plugin)
derivative work created by the ```generate-plugin.sh```, is licensed under the
GNU General Public License (GPL), version 3 or (at your option) later.

## References
[1] von Specht, Ziebarth, and Veh (in prep.)
