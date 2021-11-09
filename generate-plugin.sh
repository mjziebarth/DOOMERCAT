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

# TODO Do some basic tests:

# Create the plugin directory:
mkdir -p build/doomercat_plugin

# Generate the help files:
python build-help.py

# Copy the DOOMERCAT module:
cp doomercat/__init__.py doomercat/defs.py doomercat/laborde.py \
   doomercat/lom.py doomercat/lombase.py doomercat/optimize.py \
   doomercat/quaternion.py doomercat/shapefile.py doomercat/lomerror.py \
   build/doomercat_plugin
mv build/doomercat_plugin/__init__.py build/doomercat_plugin/doomercat.py

# Copy the relevant python and config files for the plugin:
cp qgis-plugin/__init__.py qgis-plugin/doomercatplugin.py \
   qgis-plugin/metadata.txt qgis-plugin/dialogs.py qgis-plugin/worker.py \
   qgis-plugin/graph.py qgis-plugin/qgisproject.py qgis-plugin/LICENSE \
   qgis-plugin/README.md \
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
