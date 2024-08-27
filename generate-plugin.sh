# QGIS plugin generation script. Run this script in a linux shell
# to generate the compressed QGIS zip file.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2021 Deutsches GeoForschungsZentrum Potsdam
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
set -e

# Create the plugin directory:
MODULE=build/doomercat_plugin/module/doomercat/
mkdir -p $MODULE

# Generate the help files:
python build-help.py

# Copy the DOOMERCAT module:
cp doomercat/__init__.py doomercat/_typing.py doomercat/config.py \
   doomercat/cppextensions.py doomercat/defs.py doomercat/enclosingsphere.py \
   doomercat/fibonacci.py doomercat/geometry.py doomercat/hom.py \
   doomercat/hotine.py doomercat/hotineproject.py doomercat/initial.py \
   doomercat/lomerror.py doomercat/messages.py doomercat/shapefile.py \
   $MODULE
#mv build/doomercat_plugin/__init__.py build/doomercat_plugin/doomercat.py

# Copy the C++ libraries:
if [ "$1" = "--portable" ]; then
    cp  doomercat/_cppextensions.so doomercat/_cppextensions.dll \
        $MODULE
    if [ -f $MODULE/_cppextensions_native.so ]; then
        rm $MODULE/_cppextensions_native.so
    fi
else
    # Native-only linux plugin:
    cp doomercat/_cppextensions_native.so $MODULE
    if [ -f $MODULE/_cppextensions.so ]; then
        rm $MODULE/_cppextensions.so
    fi
    if [ -f $MODULE/_cppextensions.dll ]; then
        rm $MODULE/_cppextensions.dll
    fi
fi

# Copy the relevant python and config files for the plugin:
cp qgis-plugin/__init__.py qgis-plugin/doomercatplugin.py \
   qgis-plugin/metadata.txt qgis-plugin/dialogs.py qgis-plugin/worker.py \
   qgis-plugin/graph.py qgis-plugin/qgisproject.py qgis-plugin/LICENSE \
   qgis-plugin/README.md qgis-plugin/process.py qgis-plugin/messages.py \
   qgis-plugin/moduleloader.py qgis-plugin/pointvalidator.py\
   build/doomercat_plugin

# Generate the resources.py:
pyrcc5 -o build/doomercat_plugin/resources.py qgis-plugin/resources.qrc

# Remove (if exists) existing zip file!
if [ -f build/DOOMERCAT.zip ]; then
    rm build/DOOMERCAT.zip
fi

# Generate the compressed plugin file:
cd build
zip -r DOOMERCAT.zip doomercat_plugin/

echo Plugin ZIP file created. Can be found under build/DOOMERCAT.zip.
