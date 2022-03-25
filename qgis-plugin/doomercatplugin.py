# Data-Optimized Oblique MERCATor QGis plugin code.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam
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

# Adapted from QGIS "Developing Python Plugins" documentation

from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import *
from qgis.PyQt.QtSvg import *
from qgis.PyQt.QtCore import Qt, pyqtSlot, QThreadPool, QVariant
from qgis.core import *

# initialize Qt resources from file resources.py
from . import resources
from .dialogs import SaveProjectionDialog, ProgressDialog
from .worker import OptimizationWorker
from .qgisproject import project
from .help import help_html
from .messages import info
from .moduleloader import HAS_CPPEXTENSIONS, _ellipsoids, points_from_shapefile

import numpy as np
import os.path

# See if the shapefile module is loaded:
try:
	import shapefile
	_has_shapefile = True
except:
	_has_shapefile = False


# Some GUI strings:
_ORIENT_NORTH_PROJ_CENTER = "projection center"
_ORIENT_NORTH_DATA_CENTER = "data center"
_ORIENT_NORTH_CUSTOM = "custom"
_ALGORITHM_CPP = "C++"
_ALGORITHM_PY = "Python"


class DOOMERCATPlugin:

    def __init__(self, iface):
        # save reference to the QGIS interface
        self.iface = iface
        self._hom = None
        self._res_crs = None
        self._srsid = None
        self._cursor = None
        self._weighted_raster_layers = []
        self._weighted_vector_layers = []
        self._icon_path = ":/plugins/doomercat/icon"
        self._crs_geo = QgsCoordinateReferenceSystem("EPSG:4326")

    def initGui(self):
        # create action that will start plugin configuration
        self.action = QAction(QIcon(self._icon_path + ".png"),
                              "Data-Optimized Oblique MERCATor",
                              self.iface.mainWindow())
        self.action.setObjectName("doomAction")
        self.action.setWhatsThis("Obtain a data-optimized oblique Mercator "
                                 "projection.")
#		self.action.setStatusTip("This is status tip")
        self.action.triggered.connect(self.run)

        # add toolbar button and menu item
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&DOOMERCAT", self.action)

        # Create the configuration dialog:
        self.dialog = QDialog()

        # Projection Name Dialog:
        self.nameDialog = SaveProjectionDialog(self.dialog)
        self.nameDialog.accepted.connect(self.saveExecuted)

        # Progress dialog:
        self.progressDialog = ProgressDialog(self.dialog)

        # Populate the configuration dialog:
        self.tbHelp = QTextBrowser(self.dialog)
        self.leEllipsoidA = QLineEdit(self.dialog)
        self.leEllipsoidA.setEnabled(False)
        self.leEllipsoidA.setMaximumWidth(75)
        self.leEllipsoidInverseFlattening = QLineEdit(self.dialog)
        self.leEllipsoidInverseFlattening.setEnabled(False)
        self.leEllipsoidInverseFlattening.setMaximumWidth(60)
        self.leResult = QLineEdit(self.dialog)
        self.leResult.setReadOnly(True)
        self.leResult.setEnabled(False)
        self.sbExponent= QDoubleSpinBox(self.dialog)
        self.sbExponent.setMinimum(2.0)
        self.cb_k0 = QCheckBox(self.dialog)
        self.cb_k0.setCheckState(Qt.Checked)
        self.cbAlgorithm = QComboBox(self.dialog)
        self.cbAlgorithm.addItem(_ALGORITHM_PY)
        if HAS_CPPEXTENSIONS:
            self.cbAlgorithm.addItem(_ALGORITHM_CPP)
        self.cbEllipsoid = QComboBox(self.dialog)
        self.cbEllipsoid.addItem('WGS84')
        self.cbEllipsoid.addItem('GRS80')
        self.cbEllipsoid.addItem('IERS2003')
        self.cbEllipsoid.addItem('custom')
        self.cbEllipsoid.currentTextChanged.connect(self.cbEllipsoidChanged)
        self.cbEllipsoidChanged('WGS84')
        self.cbOrientNorth = QCheckBox(self.dialog)
        self.cbOrientNorth.setCheckState(Qt.Checked)
        self.cbOrientNorth.stateChanged.connect(self.orientNorthClicked)
        self.cbOrientCenter = QComboBox(self.dialog)
        self.cbOrientCenter.addItem(_ORIENT_NORTH_DATA_CENTER)
        self.cbOrientCenter.addItem(_ORIENT_NORTH_PROJ_CENTER)
        self.cbOrientCenter.addItem(_ORIENT_NORTH_CUSTOM)
        self.cbInfinityNorm = QCheckBox(self.dialog)
        self.cbInfinityNorm.setCheckState(Qt.Unchecked)
        self.sb_k0= QDoubleSpinBox(self.dialog)
        self.sb_k0.setMinimum(0.0)
        self.sb_k0.setValue(0.95)
        self.sb_k0.setDecimals(3)
        self.sb_k0.setSingleStep(0.005)
        self.sb_k0_std= QDoubleSpinBox(self.dialog)
        self.sb_k0_std.setMinimum(0.001)
        self.sb_k0_std.setValue(0.01)
        self.sb_k0_std.setDecimals(3)
        self.sb_k0_std.setSingleStep(0.001)
        self.tabLayout = QTabWidget(self.dialog)
        self.tabLayout.setMinimumWidth(480)
        dialog_layout = QGridLayout(self.dialog)
        row = 0
        self.svg_widget = QSvgWidget(self._icon_path + ".svg")
        self.svg_widget.customContextMenuRequested.connect(self.switchIcon)
        self.svg_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        dialog_layout.addWidget(self.svg_widget, row, 2, 3, 2,
                                alignment=Qt.AlignCenter | Qt.AlignRight)
        dialog_layout.addWidget(QLabel("Cost function exponent:", self.dialog),
                                row,0)
        dialog_layout.addWidget(self.sbExponent, row, 1)
        row += 1
        dialog_layout.addWidget(QLabel("is infinite:", self.dialog),
                                row, 0,
                                alignment = Qt.AlignRight | Qt.AlignCenter)
        dialog_layout.addWidget(self.cbInfinityNorm, row, 1)
        self.cbInfinityNorm.stateChanged.connect(self.infinityNormCheckboxClick)
        row += 1
        dialog_layout.addWidget(QLabel("Minimum k0 constraint:", self.dialog),
                                row,0)
        dialog_layout.addWidget(self.sb_k0, row, 1)
        dialog_layout.addWidget(self.cb_k0, row, 2)
        self.cb_k0.stateChanged.connect(self.k0ConstraintCheckboxClicked)
        row += 1
        label = QLabel("with bandwidth", self.dialog)
        label.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        dialog_layout.addWidget(label, row,0)
        dialog_layout.addWidget(self.sb_k0_std, row, 1)

        # Selecting the ellipsoid to use for the optimization:
        row += 1
        dialog_layout.addWidget(QLabel("Ellipsoid (a in km):",
                                self.dialog), row, 0)
        dialog_layout.addWidget(self.cbEllipsoid, row, 1)
        hbox = QWidget(self.dialog)
        dialog_layout.addWidget(hbox, row, 2, 1, 2)
        hbox_layout = QHBoxLayout(hbox)
        hbox_layout.setSpacing(0)
        hbox.setLayout(hbox_layout)
        hbox_layout.addWidget(QLabel("a:"))
        hbox_layout.addWidget(self.leEllipsoidA)
        hbox_layout.addWidget(QLabel("1/f:"))
        hbox_layout.addWidget(self.leEllipsoidInverseFlattening)

        # Algorithm selection:
        row += 1
        dialog_layout.addWidget(QLabel("Optimizer:",
                                self.dialog), row, 0)
        dialog_layout.addWidget(self.cbAlgorithm, row, 1)



        # The tab layout widget that switches between the data sources:
        row += 1
        tab0 = QWidget(self.dialog)
        self.lblSelection = QLabel(tab0)
        self.lblSelection.setText("Optimize data for the currently selected "
                                  "features of this project.")
        tab0_layout = QGridLayout(tab0)
        tab0_layout.addWidget(self.lblSelection, 0, 0)
        self.tabLayout.addTab(tab0, "Selection")
        tab1 = QWidget(self.dialog)
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setSpacing(0)
        tab1_widget1 = QWidget(tab1)
        tab1_widget1.setContentsMargins(0,0,0,0)
        tab1_layout1 = QHBoxLayout(tab1_widget1)
        tab1_layout.addWidget(tab1_widget1)
        tab1_layout1.addWidget(QLabel("File:"))
        self.leShapefilePath = QLineEdit(tab1)
        tab1_layout1.addWidget(self.leShapefilePath)
        self.btnShapefileFinder = QPushButton("...", tab1)
        self.btnShapefileFinder.setMaximumWidth(20)
        self.btnShapefileFinder.clicked.connect(self.shapefileOpenClicked)
        tab1_layout1.addWidget(self.btnShapefileFinder)
        tab1_widget2 = QWidget(tab1)
        tab1_widget2.setContentsMargins(0,0,0,0)
        tab1_layout2 = QHBoxLayout(tab1_widget2)
        tab1_layout.addWidget(tab1_widget2)
        # Radio buttons to switch between all points selection and reduction to
        # centroids.
        self.bgShapefilePoints = QButtonGroup(self.dialog)
        allPts = QRadioButton("All points", tab1_widget2)
        allPts.setChecked(True)
        tab1_layout2.addWidget(allPts)
        self.bgShapefilePoints.addButton(allPts)
        centroids = QRadioButton("Centroids of geometries", tab1_widget2)
        tab1_layout2.addWidget(centroids)
        self.bgShapefilePoints.addButton(centroids)

        self.tabLayout.addTab(tab1, "Shapefile")
        self.tabLayout.setTabToolTip(0, "Optimize for selected points from "
                                        "this project.")
        self.tabLayout.setTabToolTip(1, "Optimize for points from a shapefile.")
        if not _has_shapefile:
            # If shapefile library cannot be imported, disable the shapefile
            # tab:
            self.tabLayout.setTabEnabled(1, False)
            self.tabLayout.setTabToolTip(1, "Reading points from shapefiles "
                                            "requires the PyShp library. "
                                            "It can be installed using pip.")

        #
        # The "Weighted Raster" tab
        #
        tab2 = QWidget(self.dialog)
        self.tabLayout.addTab(tab2, "Weighted Raster")
        tab2_layout = QGridLayout(tab2)
        tab2_layout.setColumnStretch(1,1)
        tab2_layout.addWidget(QLabel("Layer:"), 0, 0, 1, 1)
        self.cbWeightedRasterLayer = QComboBox(tab2)
        self.cbWeightedRasterLayer.currentIndexChanged\
             .connect(self.cbWeightedLayerChanged)
        tab2_layout.addWidget(self.cbWeightedRasterLayer, 0, 1, 1, 1)
        tab2_layout.addWidget(QLabel("Weighting band:"), 1, 0, 1, 1)
        self.cbWeightingBand = QComboBox(tab2)
        tab2_layout.addWidget(self.cbWeightingBand, 1, 1, 1, 1)

        #
        # The "Weighted Points" tab
        #
        tab3 = QWidget(self.dialog)
        self.tabLayout.addTab(tab3, "Weighted Points")
        tab3_layout = QGridLayout(tab3)
        tab3_layout.setColumnStretch(0,0)
        tab3_layout.setColumnStretch(1,1)
        tab3_layout.addWidget(QLabel("Layer:"), 0, 0, 1, 1)
        self.cbWeightedVectorLayer = QComboBox(tab3)
        self.cbWeightedVectorLayer.currentIndexChanged\
              .connect(self.cbWeightedLayerChanged)
        tab3_layout.addWidget(self.cbWeightedVectorLayer, 0, 1, 1, 1)
        tab3_layout.addWidget(QLabel("Weighting attribute:"), 1, 0, 1, 1)
        self.cbWeightingAttribute = QComboBox(tab3)
        tab3_layout.addWidget(self.cbWeightingAttribute, 1, 1, 1, 1)


        dialog_layout.addWidget(self.tabLayout, row, 0, 1, 4)

        # A widget that eats additional space:
        row += 1
        dialog_layout.addWidget(QWidget(self.dialog), row, 0, 4, 1)
        dialog_layout.setRowStretch(row, 1)

        # Output of projection strings:
        row += 1
        dialog_layout.addWidget(QLabel("Orient North"), row, 0, 1, 1)
        dialog_layout.addWidget(self.cbOrientCenter, row, 1, 1, 1)
        dialog_layout.addWidget(self.cbOrientNorth, row, 2, 1, 1)
        self.cbOrientNorth.stateChanged \
                          .connect(self.orientNorthCheckboxClicked)
        row += 1
        dialog_layout.addWidget(QLabel("Projection string:", self.dialog),
                                row, 0, 1, 3)
        row += 1
        dialog_layout.addWidget(self.leResult, row, 0, 1, 5)
        row += 1

        # Only set up the buttons here since otherwise, their focus might be
        # covered by something else:
        self.btnOptimize = QPushButton("&Optimize",self.dialog)
        self.btnOptimize.clicked.connect(self.optimizeClicked)
        self.btnOptimize.setEnabled(True)
        self.btnSave = QPushButton("&Save", self.dialog)
        self.btnSave.clicked.connect(self.nameDialog.exec)
        self.btnSave.setEnabled(True)
        self.btnSave.setEnabled(False)
        self.btnApply = QPushButton("&Apply",self.dialog)
        self.btnApply.clicked.connect(self.applyProjection)
        self.btnApply.setEnabled(True)
        self.btnApply.setEnabled(False)

        self.tabLayout.currentChanged.connect(self.checkOptimizeEnable)
        self.leShapefilePath.textChanged.connect(self.checkOptimizeEnable)
        self.iface.mapCanvas().layersChanged.connect(self.checkOptimizeEnable)

        dialog_layout.addWidget(self.btnOptimize, row, 0, 1, 1)
        dialog_layout.addWidget(self.btnApply, row, 3, 1, 1)
        dialog_layout.addWidget(self.btnSave, row, 1, 1, 2)
        self.dialog.setLayout(dialog_layout)
        self.dialog.setWindowTitle("Data-Optimized Oblique MERCATor")

        # Error Dialog:
        self.errorDialog = QErrorMessage(self.dialog)

        # Help dialog:
        dialog_layout.addWidget(self.tbHelp, 0, 4, row-1, 1)
        self.tbHelp.setHtml(help_html)
        self.tbHelp.setMinimumWidth(280)



    def unload(self):
        # remove the plugin menu item and icon
        info("Unloading DOOMERCAT plugin.")
        self.iface.removePluginMenu("&DOOMERCAT", self.action)
        self.iface.removeToolBarIcon(self.action)

    def run(self):
        # create and show a configuration dialog or something similar
        self.dialog.show()

    def shapefileOpenClicked(self):
        """
        This slot is called when the three-dot push button to open
        a shapefile is clicked.
        """
        filename = QFileDialog.getOpenFileName(self.dialog,
                                               "QFileDialog.getOpenFileName()",
                                               "", "Shapefiles (*.shp)")[0]
        self.leShapefilePath.setText(filename)


    def optimizeClicked(self):
        """
        Create an optimized Laborde oblique Mercator on button click.
        Decide here which data to optimize for:
        """
        # Clear previous results:
        self._res_crs = None
        self.btnApply.setEnabled(False)
        self.btnSave.setEnabled(False)
        self.leResult.setText("")
        self.leResult.setEnabled(False)

        # Obtain the ellipsoid parameters:
        ellps = self.cbEllipsoid.currentText()
        try:
            a = float(self.leEllipsoidA.text())
            f = 1.0 / float(self.leEllipsoidInverseFlattening.text())
        except:
            self.errorDialog.showMessage("Could not convert the ellipsoid "
                                         "parameters to numbers.")
            return


        ci = self.tabLayout.currentIndex()
        if ci == 0:
            # Optimize for selection:
            self.optimizeForSelection(a,f,ellps)
        elif ci == 1:
            self.optimizeForShapefile(a,f,ellps)
        elif ci == 2:
            self.optimizeWeightedRaster(a,f,ellps)
        elif ci == 3:
            self.optimizeWeightedVectorLayer(a,f,ellps)


    def optimizeForSelection(self, a, f, ellps):
        """
        Create an optimized Laborde oblique Mercator projection for
        the currently selected features.
        """

        # Get all feature coordinates.
        # Adapted from "Using Vector Layers" example.
        lon = []
        lat = []
        canvas = self.iface.mapCanvas()
        if canvas is None:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
            return

        for layer in canvas.layers():
            if layer is None:
                continue

            # Obtain first some information about the layer:
            crs = layer.crs()

            # Continue depending on layer type:
            if isinstance(layer, QgsRasterLayer):
                # TODO : For now, skip raster layers:
                continue

                # Obtain the data provider which contains the extents of the
                # raster file:
                provider = layer.constDataProvider()
                if hasattr(provider,"ignoreExtents") and \
                    provider.ignoreExtents():
                    continue

                # Obtain extents and generate the pixel coordinates:
                extent = provider.extent()

                # Use all pixels of the raster layer:
                X = np.linspace(extent.xMinimum(), extent.xMaximum(),
                                layer.width())
                Y = np.linspace(extent.yMinimum(), extent.yMaximum(),
                                layer.height())

                coordinates = np.stack(np.meshgrid(X, Y), axis=2)\
                                .reshape((-1,2))


            elif isinstance(layer, QgsVectorLayer):
                features = layer.selectedFeatures()

                if len(features) == 0:
                    # Empty selection: Nothing to be done.
                    continue

                coordinates = []
                for feat in features:
                    # Obtain the coordinates of the feature:
                    geom = feat.geometry()

                    # Get all of the points:
                    vertices = geom.vertices()
                    while vertices.hasNext():
                        p = vertices.next()
                        coordinates += [(p.x(), p.y())]

            # Obtain geographic coordinates using WGS84 CRS:
            transform = QgsCoordinateTransform(crs, self._crs_geo,
                                               QgsProject.instance())
            lonlat = [transform.transform(QgsPointXY(c[0],c[1]))
                      for c in coordinates]
            lon += [np.array([l.x() for l in lonlat])]
            lat += [np.array([l.y() for l in lonlat])]

        # Early exit if no selection:
        if len(lon) == 0:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
            return

        # Merge all coordinates of selected features across the
        # layers:
        lon = np.concatenate(lon)
        lat = np.concatenate(lat)

        # Now check if there are any coordinates.
        # Otherwise, reset:
        if lon.size == 0:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
            return

        # Optimize:
        self.optimizeLonLat(lon, lat, None, a, f, ellps)


    def optimizeForShapefile(self, a, f, ellps):
        """
        Optimize for data from a shapefile.
        """
        # First obtain the method:
        method = 'points' if self.bgShapefilePoints.checkedId() == 0 \
                          else 'centroids'

        # Get the points from the shapefile:
        try:
            lon, lat = points_from_shapefile(self.leShapefilePath.text(),
                                             method=method)
        except Exception as e:
            # Error message:
            self.errorDialog.showMessage("Reading the coordinates from the "
                                         "shapefile did not succeed. The error "
                                         "message given is: '"
                                         + str(e) + "'")
            return

        # Optimize:
        self.optimizeLonLat(lon, lat, None, a, f, ellps)


    def optimizeWeightedRaster(self, a, f, ellps):
        """
        Optimize for raster coordinates weighted by a raster channel.
        """
        # Obtain current raster layer and selected weighting band:
        l_id = self.cbWeightedRasterLayer.currentIndex()
        b_id = self.cbWeightingBand.currentIndex()+1
        layer = self._weighted_raster_layers[l_id]

        # Obtain a raster block:
        provider = layer.dataProvider()
        extent = layer.extent()
        w = layer.width()
        h = layer.height()
        block = provider.block(b_id, extent, w, h)
        data = block.data()

        # Get raster block data type and read block to numpy:
        dtype = int(provider.dataType(b_id))
        qgis2npy = {int(Qgis.Float32) : np.float32,
                    int(Qgis.Float64) : np.float64,
                    int(Qgis.Int16) : np.int16, int(Qgis.Int32) : np.int32,
                    int(Qgis.UInt16) : np.uint16, int(Qgis.Byte) : np.uint8,
                    int(Qgis.UInt32) : np.uint32}
        if dtype not in qgis2npy:
            raise TypeError("Raster layer data type not supported.")
        dtype = qgis2npy[dtype]
        array = np.frombuffer(data, dtype=dtype).astype(float)

        # Get raster coordinates:
        crs = layer.crs()
        dx = extent.width() / w
        dy = extent.height() / h
        x = extent.xMinimum() + (0.5 + np.arange(w)) * dx
        y = extent.yMaximum() - (0.5 + np.arange(h)) * dy
        x,y = np.meshgrid(x,y)
        xf = x.reshape(-1)
        yf = y.reshape(-1)

        # Project to WGS84:
        dest_crs = QgsCoordinateReferenceSystem(4326)
        lon, lat = project(xf, yf, crs, dest_crs)

        # Call the optimization routine:
        self.optimizeLonLat(lon, lat, array, a, f, ellps)


    def optimizeWeightedVectorLayer(self, a, f, ellps):
        """
        Optimize for the selected weighted vector layer.
        """
        # Obtain current vector layer and selected weighting attribute:
        l_id = self.cbWeightedVectorLayer.currentIndex()
        a_id = self.cbWeightingAttribute.currentIndex()
        layer = self._weighted_vector_layers[l_id]
        # The list of attributes displayed contains only numeric attributes:
        a_id = self._weighted_vector_attributes[a_id]

        if layer.featureCount() == 0:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
            return

        xy = []
        weight = []
        for feat in layer.getFeatures():
            if not feat.isValid():
                continue

            # Feature weight:
            w = float(feat[a_id])

            # Obtain the coordinates of the feature:
            geom = feat.geometry()

            # Get all of the points:
            vertices = geom.vertices()
            while vertices.hasNext():
                p = vertices.next()
                xy.append((p.x(), p.y()))
                weight.append(w)

        xy = np.array(xy)
        weight = np.array(weight)

        # Project to geographic coordinate system:
        lon, lat = project(xy[:,0], xy[:,1], layer.crs(), self._crs_geo)

        # Call the optimization routine:
        self.optimizeLonLat(lon, lat, weight, a, f, ellps)


    def optimizeLonLat(self, lon, lat, weight, a_km, f, ellps):
        """
        Optimize for numpy arrays of lon and lat:
        """
        # Set waiting cursor:
        self._cursor = self.dialog.cursor()
        self.dialog.setCursor(Qt.WaitCursor)

        # Disable optimization button:
        self.btnOptimize.setDisabled(True)

        # Obtain optimzied HOM:
        k0_ap = self.sb_k0.value() if self.cb_k0.checkState() == Qt.Checked \
                else 0.0
        pnorm = np.inf if self.cbInfinityNorm.checkState() == Qt.Checked \
                else self.sbExponent.value()
        backend = 'Python' if self.cbAlgorithm.currentText() == _ALGORITHM_PY \
                  else 'C++'
        threadpool = QThreadPool.globalInstance()
        worker = OptimizationWorker(lon, lat, weight=weight, pnorm=pnorm,
                                    k0_ap=k0_ap,
                                    sigma_k0=self.sb_k0_std.value(),
                                    ellipsoid= None if ellps == 'custom'
                                               else ellps,
                                    f = f if ellps == 'custom' else None,
                                    a = 1e3*a_km if ellps == 'custom' else None,
                                    backend = backend)
        worker.signals.result.connect(self.receiveResult)
        worker.signals.error.connect(self.receiveError)
        worker.signals.finished.connect(self.workerFinished)
        worker.signals.progress.connect(self.progressDialog.logCost)
        worker.signals.reducePoints.connect(self.progressDialog
                                            .startReducePoints)
        worker.signals.enterOptimization.connect(self.progressDialog
                                                 .enterOptimization)
        worker.signals.pnorm.connect(self.progressDialog.pnormSet)
        self.progressDialog.rejected.connect(worker.performExit)
        threadpool.start(worker)


    def receiveResult(self, hom):
        """
        Slot to receive results from the optimization worker.
        """
        # Save the result:
        self._hom = hom

        self.adjustProjStr()

        # Enable the save button:
        self.btnSave.setEnabled(True)


    def adjustProjStr(self):
        """
        Computes the Proj string according to all selected options
        and the result of an optimization.
        """
        if self._hom is None:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
        else:
            # Compose the proj str:
            proj_str = self._hom.proj4_string()

            if self.cbOrientNorth.checkState() == Qt.Checked:
                # Choose a point where to orient North=Up:
                current = self.cbOrientCenter.currentText()
                if current == _ORIENT_NORTH_PROJ_CENTER:
                    # Gamma will be chosen automatically by Proj.
                    pass
                elif current == _ORIENT_NORTH_DATA_CENTER:
                    # Use the center of the smallest enclosing sphere.
                    gamma = self._hom.north_gamma(*self._hom.
                                           enclosing_sphere_center())
                    proj_str += " +gamma=%.8f" % (gamma,)
                elif current == _ORIENT_NORTH_CUSTOM:
                    proj_str += " +gamma=TODO"
            else:
                proj_str += " +no_rot +no_off"

            self.leResult.setText(proj_str)
            self.leResult.setEnabled(True)


    def workerFinished(self):
        """
        Thread-finished cleanup.
        """
        if self._cursor is not None:
            self.dialog.setCursor(self._cursor)

        # Enable Optimization:
        self.checkOptimizeEnable()


    def receiveError(self, oe=None):
        """
        Slot called when the optimization failed.
        """
        if oe is not None:
            self.errorDialog.showMessage(str(oe))
        else:
            self.errorDialog.showMessage("An error occurred.")


    def saveExecuted(self):
        """
        Save button clicked and name chosen.
        Generate the CRS and save it as user defined.
        """
        # Generate the CRS (and override previous one):
        if self._res_crs is not None:
            del self._res_crs
            QgsCoordinateReferenceSystem.invalidateCache()
        self._res_crs \
           = QgsCoordinateReferenceSystem("PROJ4:" + self.leResult.text())

        # Rename and save the CRS:
        name = self.nameDialog.text()
        self._srsid = self._res_crs.saveAsUserCrs(name)
        if self._srsid == -1:
            self.errorDialog.showMessage("The resulting coordinate reference "
                                         "system could not be saved as a "
                                         "user CRS.")
            self._srsid = None

        # If the CRS is valid, we can enable the apply button,
        # else we disable it:
        if self._res_crs.isValid():
            self.btnApply.setEnabled(True)
        else:
            self.btnApply.setEnabled(False)

            # Show a warning dialog:
            self.errorDialog.showMessage("The resulting coordinate reference "
                                         "system is not valid.")

        # TODO: More sanity checks on the obtained parameters?
        # (i.e. propose to use UTM or Mercator in case that alpha is
        #  close to 0° or 90°)


    def applyProjection(self):
        """
        Apply the optimized projection to the current layer.
        """
        if self._res_crs is None or not self._res_crs.isValid():
            return

        # Try to set the user saved system:
        if self._srsid is not None:
            crs = QgsCoordinateReferenceSystem.fromSrsId(self._srsid)
            if crs.isValid():
                QgsProject.instance().setCrs(crs)
                return

        # Set the optimized CRS:
        QgsProject.instance().setCrs(self._res_crs)


    def checkOptimizeEnable(self, tab=None):
        """
        This slot checks whether to enable the optimize button
        if something related to it changed.
        """
        if tab is None:
            tab = self.tabLayout.currentIndex()
        if tab == 0:
            self.btnOptimize.setEnabled(True)

        elif tab == 1:
            path = self.leShapefilePath.text()
            self.btnOptimize.setEnabled(len(path) > 0 and os.path.isfile(path))

        elif tab == 2:
            # Obtain all raster bands:
            layers = []
            layer_names = []
            for layer in self.iface.mapCanvas().layers():
                if isinstance(layer, QgsRasterLayer):
                    layers += [layer]
                    layer_names += [str(layer.name())]

            # If no raster bands, clear selection:
            if len(layers) == 0:
                self._weighted_raster_layers = []
                self.cbWeightedRasterLayer.blockSignals(True)
                self.cbWeightedRasterLayer.clear()
                self.cbWeightedRasterLayer.blockSignals(False)
                self.cbWeightingBand.clear()
                self.btnOptimize.setEnabled(False)
                return

            # Check if the current layer is in the list of layers:
            current_index = 0
            ci2 = 0
            self.cbWeightedRasterLayer.blockSignals(True)
            if self.cbWeightedRasterLayer.count() > 0:
                id = self.cbWeightedRasterLayer.currentIndex()
                current_layer = self._weighted_raster_layers[id]
                if current_layer in layers:
                    current_index = layers.index(current_layer)
                    ci2 = self.cbWeightingBand.currentIndex()
                self.cbWeightedRasterLayer.clear()
            self.cbWeightingBand.clear()

            # Set the combobox entries:
            for l in layer_names:
                self.cbWeightedRasterLayer.addItem(l)
            self.cbWeightedRasterLayer.setCurrentIndex(current_index)
            layer = layers[current_index]
            for i in range(layer.bandCount()):
                self.cbWeightingBand.addItem(layer.bandName(i+1))
            self.cbWeightingBand.setCurrentIndex(ci2)
            self._weighted_raster_layers = layers

            self.btnOptimize.setEnabled(True)
            self.cbWeightedRasterLayer.blockSignals(False)

        elif tab == 3:
            # Obtain all vector layers:
            layers = []
            layer_names = []
            for layer in self.iface.mapCanvas().layers():
                if isinstance(layer, QgsVectorLayer):
                    layers += [layer]
                    layer_names += [str(layer.name())]

            # If no raster bands, clear selection:
            if len(layers) == 0:
                self._weighted_vector_layers = []
                self.cbWeightedVectorLayer.blockSignals(True)
                self.cbWeightedVectorLayer.clear()
                self.cbWeightedVectorLayer.blockSignals(False)
                self.cbWeightingAttribute.clear()
                self.btnOptimize.setEnabled(False)
                return

            # Check if the current layer is in the list of layers:
            current_index = 0
            ci2 = 0
            self.cbWeightedVectorLayer.blockSignals(True)
            if self.cbWeightedVectorLayer.count() > 0:
                id = self.cbWeightedVectorLayer.currentIndex()
                current_layer = self._weighted_vector_layers[id]
                if current_layer in layers:
                    current_index = layers.index(current_layer)
                    ci2 = self.cbWeightingAttribute.currentIndex()
                self.cbWeightedVectorLayer.clear()
            self.cbWeightingAttribute.clear()

            # Set the combobox entries:
            for l in layer_names:
                self.cbWeightedVectorLayer.addItem(l)
            self.cbWeightedVectorLayer.setCurrentIndex(current_index)
            layer = layers[current_index]
            self._weighted_vector_attributes = []
            for i,field in enumerate(layer.fields()):
                if field.isNumeric():
                    self._weighted_vector_attributes += [i]
                    self.cbWeightingAttribute.addItem(field.name())

            if len(self._weighted_vector_attributes) == 0:
                # No attribute to select for this layer:
                self.cbWeightingAttribute.clear()
                self.btnOptimize.setEnabled(False)
            else:
                self.cbWeightingAttribute.setCurrentIndex(ci2)
                self.btnOptimize.setEnabled(True)

            self._weighted_vector_layers = layers
            self.cbWeightedVectorLayer.blockSignals(False)


    def cbEllipsoidChanged(self, ename):
        """
        Called when the ellipsoid selection checkbox is changed.
        """
        # Obtain the ellipsoid definition:
        if ename == 'custom':
            # Enable editing:
            self.leEllipsoidA.setEnabled(True)
            self.leEllipsoidInverseFlattening.setEnabled(True)
        else:
            ellipsoid = _ellipsoids[ename]
            # Display a in km instead of m:
            self.leEllipsoidA.setText("%.8f" % (1e-3*ellipsoid[0]))
            self.leEllipsoidA.setEnabled(False)
            self.leEllipsoidA.setCursorPosition(0)
            self.leEllipsoidInverseFlattening.setText("%.8f" % (ellipsoid[1]))
            self.leEllipsoidInverseFlattening.setEnabled(False)
            self.leEllipsoidInverseFlattening.setCursorPosition(0)


    def orientNorthClicked(self, state):
        """
        Clicked when the '+no_rot +no_off' argument should be added (or
        removed).
        """
        if self._hom is not None:
            self.adjustProjStr()


    def cbWeightedLayerChanged(self, item):
        """
        Shortcut eating the argument.
        """
        self.checkOptimizeEnable()


    def k0ConstraintCheckboxClicked(self, state):
        """
        This slot is called when the constraint on k_0 should
        be enabled or disabled.
        """
        disabled = (state == 0)
        self.sb_k0.setDisabled(disabled)
        self.sb_k0_std.setDisabled(disabled)


    def infinityNormCheckboxClick(self, state):
        """
        This slot is called when the infinity norm should be selected
        or deselected.
        """
        infinite = (state != 0)
        self.sbExponent.setDisabled(infinite)


    def orientNorthCheckboxClicked(self, state):
        """
        This slot is called when the north orientation should
        be enabled or disabled.
        """
        disabled = (state == 0)
        self.cbOrientCenter.setDisabled(disabled)
        if self._hom is not None:
            self.adjustProjStr()


    def switchIcon(self, *args):
        if self._icon_path == ":/plugins/doomercat/icon":
            self._icon_path = ":/plugins/doomercat/icon2"
        else:
            self._icon_path = ":/plugins/doomercat/icon"

        self.action.setIcon(QIcon(self._icon_path+".png"))
        self.svg_widget.load(self._icon_path+".svg")
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
