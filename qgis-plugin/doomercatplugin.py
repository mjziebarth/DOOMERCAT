# Data-Optimized Oblique MERCATor QGis plugin code.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam,
#               2024 Technical University of Munich
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
from .moduleloader import HAS_CPPEXTENSIONS, _ellipsoids
from .pointvalidator import CoordinateValidator
from .wkt import WktCRS

import numpy as np
import os.path
from numpy.typing import NDArray


# Some GUI strings:
_ORIENT_NORTH_PROJ_CENTER = "projection center"
_ORIENT_NORTH_DATA_CENTER = "data center"
_ORIENT_NORTH_CUSTOM = "custom"
_ALGORITHM_CPP = "C++"
_ALGORITHM_PY = "Python"

_WKT_PARSER_NOTE = "<br><br>Note: DOOMERCAT's WKT parser is built partially " \
    "but not fully to OGC 18-010r11 standard. If you think that " \
    "the parser's assessment is incorrect, you may well be " \
    "correct. As a short-term fix, you might adjust the data's " \
    "CRS WKT to something the parser can handle. You might " \
    "wish to consider a bug report to improve the standard " \
    "coverage of the parser."


class DOOMERCATPlugin:

    wkt: "str | None"
    optimization_input: "tuple[\
        WktCRS, NDArray[np.double], NDArray[np.double]\
    ] | None"

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
        self.leResult = QLineEdit(self.dialog)
        self.leResult.setReadOnly(True)
        self.leResult.setEnabled(False)
        self.teWktResult = QTextEdit(self.dialog)
        self.teWktResult.setReadOnly(True)
        self.teWktResult.setEnabled(False)
        self.wkt = None
        self.optimization_input = None
        self.leCentralPoint = QLineEdit(self.dialog)
        self.leCentralPoint.setEnabled(False)
        self.leCentralPoint.setValidator(CoordinateValidator())
        self.leBFGSEpsilon = QLineEdit(self.dialog)
        self.leBFGSEpsilon.setEnabled(False)
        self.leBFGSEpsilon.setValidator(
            QDoubleValidator(0.0, 1e5, 1, self.dialog)
        )
        self.leBFGSEpsilon.setText("0.0")
        self.sbExponent= QDoubleSpinBox(self.dialog)
        self.sbExponent.setMinimum(2.0)
        self.cb_k0 = QCheckBox(self.dialog)
        self.cb_k0.setCheckState(Qt.Checked)
        self.cbAlgorithm = QComboBox(self.dialog)
        self.cbAlgorithm.addItem(_ALGORITHM_PY)
        if HAS_CPPEXTENSIONS:
            self.cbAlgorithm.addItem(_ALGORITHM_CPP)
        self.cbUseHeight = QCheckBox(self.dialog)
        self.cbUseHeight.setCheckState(Qt.Checked)
        self.cbOrientNorth = QCheckBox(self.dialog)
        self.cbOrientNorth.setCheckState(Qt.Checked)
        self.cbOrientNorth.stateChanged.connect(self.orientNorthClicked)
        self.cbOrientCenter = QComboBox(self.dialog)
        self.cbOrientCenter.addItem(_ORIENT_NORTH_DATA_CENTER)
        self.cbOrientCenter.addItem(_ORIENT_NORTH_PROJ_CENTER)
        self.cbOrientCenter.addItem(_ORIENT_NORTH_CUSTOM)
        self.cbInfinityNorm = QCheckBox(self.dialog)
        self.cbInfinityNorm.setCheckState(Qt.Unchecked)
        self.cbFisherBinghamWeight = QCheckBox(self.dialog)
        self.cbFisherBinghamWeight.setCheckState(Qt.Unchecked)
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
        self.sb_Nmax = QSpinBox(self.dialog)
        self.sb_Nmax.setMinimum(10)
        self.sb_Nmax.setMaximum(1000000)
        self.sb_Nmax.setValue(1000)
        self.sbNmaxPreAdamax = QSpinBox(self.dialog)
        self.sbNmaxPreAdamax.setMinimum(0)
        self.sbNmaxPreAdamax.setMaximum(1000)
        self.sbNmaxPreAdamax.setValue(50)
        self.tabLayout = QTabWidget(self.dialog)
        self.tabLayout.setMinimumWidth(480)
        dialog_layout = QGridLayout(self.dialog)
        row = 0
        self.svg_widget = QSvgWidget(self._icon_path + ".svg")
        self.svg_widget.customContextMenuRequested.connect(self.switchIcon)
        self.svg_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        if hasattr(self.svg_widget.renderer(), "setAspectRatioMode"):
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

        # Algorithm selection:
        row += 1
        dialog_layout.addWidget(QLabel("Optimizer:",
                                self.dialog), row, 0)
        dialog_layout.addWidget(self.cbAlgorithm, row, 1)


        # Some optimization parameters:
        row += 1
        dialog_layout.addWidget(QLabel("Maximum iterations:",
                                self.dialog), row, 0)
        dialog_layout.addWidget(self.sb_Nmax, row, 1)

        row += 1
        dialog_layout.addWidget(QLabel("Fisher-Bingham weighted:",
                                self.dialog), row, 0)
        dialog_layout.addWidget(self.cbFisherBinghamWeight, row, 1)

        row += 1
        dialog_layout.addWidget(QLabel("Adamax preoptim. steps:",
                                self.dialog), row, 0)
        dialog_layout.addWidget(self.sbNmaxPreAdamax, row, 1)

        row += 1
        dialog_layout.addWidget(QLabel("BFGS epsilon:",
                                self.dialog), row, 0)
        dialog_layout.addWidget(self.leBFGSEpsilon, row, 1)



        # The tab layout widget that switches between the data sources:
        row += 1
        tab0 = QWidget(self.dialog)
        self.lblSelection = QLabel(tab0)
        self.lblSelection.setText("Optimize data for the currently selected "
                                  "features of this project.")
        tab0_layout = QGridLayout(tab0)
        tab0_layout.addWidget(self.lblSelection, 0, 0)
        self.tabLayout.addTab(tab0, "Selection")
        self.tabLayout.setTabToolTip(0, "Optimize for selected points from "
                                        "this project.")

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

        # Whether to use the z-information:
        row += 1
        dialog_layout.addWidget(QLabel("Use data height:"), row, 0, 1, 1)
        dialog_layout.addWidget(self.cbUseHeight, row, 1, 1, 1)

        # A widget that eats additional space:
        row += 1
        dialog_layout.addWidget(QWidget(self.dialog), row, 0, 4, 1)
        dialog_layout.setRowStretch(row, 1)

        row += 1
        dialog_layout.addWidget(QLabel("Orient North"), row, 0, 1, 1)
        dialog_layout.addWidget(self.cbOrientCenter, row, 1, 1, 1)
        dialog_layout.addWidget(self.cbOrientNorth, row, 2, 1, 1)
        self.cbOrientNorth.stateChanged \
                          .connect(self.orientNorthCheckboxClicked)
        row += 1
        label_at_point = QLabel("at point (lon,lat)")
        label_at_point.setAlignment(Qt.AlignRight)
        dialog_layout.addWidget(label_at_point, row, 0, 1, 1)
        dialog_layout.addWidget(self.leCentralPoint, row, 1, 1, 2)
        # Output of projection strings:
        row += 1
        dialog_layout.addWidget(QLabel("Projection string:", self.dialog),
                                row, 0, 1, 3)
        row += 1
        dialog_layout.addWidget(self.leResult, row, 0, 1, 5)
        row += 1
        dialog_layout.addWidget(QLabel("Projected CRS WKT:", self.dialog),
                                row, 0, 1, 3)
        row += 1
        dialog_layout.addWidget(self.teWktResult, row, 0, 1, 5)
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
        self.iface.mapCanvas().layersChanged.connect(self.checkOptimizeEnable)
        self.cbOrientCenter.currentIndexChanged.connect(self.orientNorthChanged)
        self.leCentralPoint.textEdited.connect(self.centralPointEdited)
        self.cbAlgorithm.currentIndexChanged.connect(self.cbAlgorithmChanged)

        dialog_layout.addWidget(self.btnOptimize, row, 0, 1, 1)
        dialog_layout.addWidget(self.btnApply, row, 3, 1, 1)
        dialog_layout.addWidget(self.btnSave, row, 1, 1, 2)
        self.dialog.setLayout(dialog_layout)
        self.dialog.setWindowTitle("Data-Optimized Oblique MERCATor")

        # Error Dialog:
        self.errorDialog = QErrorMessage(self.dialog)

        # Help dialog:
        dialog_layout.addWidget(self.tbHelp, 0, 4, row-3, 1)
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
        self.teWktResult.setText("")
        self.teWktResult.setEnabled(False)
        self.wkt = None


        ci = self.tabLayout.currentIndex()
        if ci == 0:
            # Optimize for selection:
            self.optimizeForSelection()
        elif ci == 1:
            self.optimizeWeightedRaster()
        elif ci == 2:
            self.optimizeWeightedVectorLayer()


    def _get_geographic_crs(self,
            crs: QgsCoordinateReferenceSystem,
            layer: QgsMapLayer,
        ) -> WktCRS:
        """
        Convenience function to obtain the WktCRS instance corresponding to
        a QgsCoordinateReferenceSystem instance. This function adds a bit of
        error-catching and reporting around that task.
        """
        try:
            try:
                variant = Qgis.CrsWktVariant.Wkt2_2019
            except AttributeError:
                # The above is only available starting from
                # QGIS 3.36.
                # This version has been tested to work on
                # QGIS 3.32:
                variant = QgsCoordinateReferenceSystem.WktVariant.WKT2_2019
            wkt = crs.toWkt(variant)
        except:
            raise RuntimeError(
                "Failed to obtain WKT from layer " + layer.name()
            )
        try:
            parsed_crs = WktCRS(wkt)
        except (RuntimeError, TypeError) as e:
            msg = e.args[0]
            raise RuntimeError(
                msg + "\n(in layer " + layer.name() + ")"
                + _WKT_PARSER_NOTE
            )

        # Needs to be a geographic CRS (lat-lon):
        if not parsed_crs.is_geographic:
            raise RuntimeError(
                "At least one layer with selected features is not a "
                "geographic CRS.<br><br>Offending layer: " + layer.name() +
                ".<br><br>Offending CRS: " + crs.toWkt()
                + _WKT_PARSER_NOTE
            )

        return parsed_crs



    def optimizeForSelection(self):
        """
        Create an optimized Laborde oblique Mercator projection for
        the currently selected features.
        """

        # Get all feature coordinates.
        # Adapted from "Using Vector Layers" example.
        lon = []
        lat = []
        h = []
        canvas = self.iface.mapCanvas()
        if canvas is None:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
            self.teWktResult.setText("")
            self.teWktResult.setEnabled(False)
            self.wkt = None
            return

        # TODO: Display the ellipsoid in a status message after DOOM was run.
        geo_crs: "QgsCoordinateReferenceSystem | None" = None
        parsed_geo_crs: "WktCRS | None" = None

        def ensure_geo_crs(
                layer: QgsMapLayer
            ) -> tuple[QgsCoordinateReferenceSystem, WktCRS]:
            """
            This function handles setting the geographic reference system
            for all layers (`geo_crs`) as well as ensuring that all layers
            with selected geometries have that same CRS.
            The latter is ensured by return value; if `True` is returned,
            the input CRS `crs` is compatible with `geo_crs` (this may also
            mean that it is the first CRS encountered).
            """
            crs = layer.crs()
            parsed_crs = self._get_geographic_crs(crs, layer)

            # If it's the first geographic CRS, automatic accept and set the
            # geo_crs for the following layers:
            nonlocal geo_crs, parsed_geo_crs
            if geo_crs is None or parsed_geo_crs is None:
                parsed_geo_crs = parsed_crs
                return crs, parsed_crs

            # Otherwise require the same CRS:
            if geo_crs != crs:
                raise RuntimeError(
                    "At least one layer with selected features does not have "
                    "the same geographic CRS as other layers.<br><br>Offending "
                    "layer: " + layer.name() +
                    "<br><br>Offending CRS: " + crs.toWkt() +
                    "<br><br>Other CRS: " + geo_crs.toWkt()
                )

            return geo_crs, parsed_geo_crs

        #
        # Main loop: iterate over all vector layers and check if the selection
        #            is empty.
        #
        layer: "QgsMapLayer | None"
        coordinates: list[tuple[float,float,float]] = []
        for layer in canvas.layers():
            if layer is None:
                continue

            # Handle only vector layers:
            if isinstance(layer, QgsVectorLayer):
                features = layer.selectedFeatures()

                if len(features) == 0:
                    # Empty selection: Nothing to be done.
                    continue

                # Now that we have coordinates, ensure that the CRS is
                # geographic:
                try:
                    geo_crs, parsed_geo_crs = ensure_geo_crs(layer)
                except (RuntimeError, TypeError) as e:
                    self.errorDialog.showMessage(
                        e.args[0]
                    )
                    return

                for feat in features:
                    # Obtain the coordinates of the feature:
                    geom = feat.geometry()

                    # Get all of the points:
                    vertices = geom.vertices()
                    while vertices.hasNext():
                        p = vertices.next()
                        coordinates.append((p.x(), p.y(), p.z()))

        # Early exit if no selection:
        if len(coordinates) == 0:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
            self.teWktResult.setText("")
            self.teWktResult.setEnabled(False)
            self.wkt = None
            return

        # Merge all coordinates of selected features across the
        # layers:
        xyz = np.array(coordinates, dtype=np.double)
        if parsed_geo_crs is None:
            # This is just for keeping the type checker happy.
            # If len(coordinates) > 0, we have also ensured a geo_crs.
            self.errorDialog.showMessage(
                "Impossible error: len(coordinates) > 0 but geo_crs is None."
            )
            return
        if parsed_geo_crs.has_axis_inverted:
            lon = xyz[:,0].copy()
            lat = xyz[:,1].copy()
        else:
            lon = xyz[:,0].copy()
            lat = xyz[:,1].copy()

        if parsed_geo_crs.has_elevation:
            h = xyz[:,2].copy()
        else:
            h = None

        # Optimize:
        self.optimizeLonLat(lon, lat, h, None, parsed_geo_crs)


    def optimizeWeightedRaster(self):
        """
        Optimize for raster coordinates weighted by a raster channel.
        """
        # Obtain current raster layer and selected weighting band:
        l_id = self.cbWeightedRasterLayer.currentIndex()
        b_id = self.cbWeightingBand.currentIndex()+1
        layer = self._weighted_raster_layers[l_id]

        # Ensure that CRS is geographic:
        crs = self._get_geographic_crs(layer.crs(), layer)

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
        dx = extent.width() / w
        dy = extent.height() / h
        x = extent.xMinimum() + (0.5 + np.arange(w)) * dx
        y = extent.yMaximum() - (0.5 + np.arange(h)) * dy
        x,y = np.meshgrid(x,y)
        xf = x.reshape(-1)
        yf = y.reshape(-1)

        # Check the order and assign lat/lon:
        if crs.has_axis_inverted:
            lon = yf
            lat = xf
        else:
            lon = xf
            lat = xf

        # Call the optimization routine:
        self.optimizeLonLat(lon, lat, np.zeros_like(lon), array, crs)


    def optimizeWeightedVectorLayer(self):
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
            self.teWktResult.setText("")
            self.teWktResult.setEnabled(False)
            self.wkt = None
            return

        # Ensure that CRS is geographic:
        crs = self._get_geographic_crs(layer.crs(), layer)

        xyzw: list[tuple[float,float,float,float]] = []
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
                xyzw.append((p.x(), p.y(), p.z(), w))

        xyzw_np: NDArray[np.double] = np.array(xyzw)
        weight = xyzw_np[:,3].copy()
        if crs.has_elevation:
            h = xyzw_np[:,2].copy()
        else:
            h = None

        # Check the order and assign lat/lon:
        if crs.has_axis_inverted:
            lon = xyzw_np[:,1]
            lat = xyzw_np[:,0]
        else:
            lon = xyzw_np[:,0]
            lat = xyzw_np[:,1]


        # Call the optimization routine:
        self.optimizeLonLat(lon, lat, h, weight, crs)


    def optimizeLonLat(self,
            lon: NDArray[np.double],
            lat: NDArray[np.double],
            h: "NDArray[np.double] | None",
            weight: "NDArray[np.double] | None",
            crs: WktCRS
        ):
        """
        Optimize for numpy arrays of lon and lat:
        """
        # Use height only if wanted:
        if self.cbUseHeight.checkState() != Qt.Checked:
            h = None

        # The methods used to obtain elevation from the vertices may return
        # nan. Here we ensure that we catch any such cases.
        # Also early check for infinite height.
        if h is not None:
            if np.any(np.isnan(h)):
                self.errorDialog.showMessage(
                    "At least one data point has elevation `nan'. This "
                    "cannot be handled. Please uncheck `Use data height' or "
                    "fix the data's elevation."
                )
                return
            elif np.any(np.isinf(h)):
                self.errorDialog.showMessage(
                    "At least one data point has infinite elevation. This "
                    "cannot be handled. Please uncheck `Use data height' or "
                    "fix the data's elevation."
                )
                return

        threadpool = QThreadPool.globalInstance()
        if threadpool is None:
            # If no threadpool found, show error message and reenable button:
            self.errorDialog.showMessage("Could not get a thread pool.")
            return

        # Set waiting cursor:
        self._cursor = self.dialog.cursor()
        self.dialog.setCursor(Qt.WaitCursor)

        # Disable optimization button:
        self.btnOptimize.setDisabled(True)

        # Save the input CRS:
        self.optimization_input = (crs, lon, lat)

        # Obtain optimzied HOM:
        k0_ap = self.sb_k0.value() if self.cb_k0.checkState() == Qt.Checked \
                else 0.0
        pnorm = np.inf if self.cbInfinityNorm.checkState() == Qt.Checked \
                else self.sbExponent.value()
        backend = 'Python' if self.cbAlgorithm.currentText() == _ALGORITHM_PY \
                  else 'C++'
        fb_weighted = self.cbFisherBinghamWeight.checkState() == Qt.Checked
        worker = OptimizationWorker(
            lon,
            lat,
            h=h,
            weight=weight,
            pnorm=pnorm,
            k0_ap=k0_ap,
            sigma_k0=self.sb_k0_std.value(),
            ellipsoid= None,
            f = crs.f,
            a = crs.a,
            Nmax=self.sb_Nmax.value(),
            fisher_bingham_use_weight = fb_weighted,
            bfgs_epsilon = float(self.leBFGSEpsilon.text()),
            Nmax_pre_adamax = self.sbNmaxPreAdamax.value(),
            backend = backend
        )
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

        self.adjustProjStr(warn_user_error=True)

        # Enable the save button:
        self.btnSave.setEnabled(True)


    def adjustProjStr(self, warn_user_error=False):
        """
        Computes the Proj string according to all selected options
        and the result of an optimization.
        """
        if self._hom is None:
            self.leResult.setText("")
            self.leResult.setEnabled(False)
            self.teWktResult.setText("")
            self.teWktResult.setEnabled(False)
            self.wkt = None
        else:
            # Compose the proj str:
            proj_str = self._hom.proj4_string()

            reset_palette = True
            gamma = None
            lon_c = lat_c = None
            if self.cbOrientNorth.checkState() == Qt.Checked:
                # Choose a point where to orient North=Up:
                current = self.cbOrientCenter.currentText()
                if current == _ORIENT_NORTH_PROJ_CENTER:
                    # Don't use the auto-assignment feature of gamma from Proj;
                    # be more explicit!
                    # It's also very simple in this case: alpha = gamma since
                    # alpha is relative to north in the projection center.
                    # lon_c, lat_c = self._hom.lonc(), self._hom.lat_0()
                    gamma = self._hom.alpha()
                elif current == _ORIENT_NORTH_DATA_CENTER:
                    # Use the center of the smallest enclosing sphere.
                    lon_c, lat_c = self._hom.enclosing_sphere_center()
                elif current == _ORIENT_NORTH_CUSTOM:
                    # Use the center given by the user:
                    text = self.leCentralPoint.text()
                    validator = self.leCentralPoint.validator()
                    if validator.validate(text, 0)[0] == QValidator.Acceptable:
                        lon_c, lat_c = (float(x) for x in text.split(','))
                    else:
                        self.leCentralPoint.setStyleSheet(
                            "QLineEdit { background : rgb(240,128,128) };"
                        )
                        reset_palette = False

                if lon_c is not None and lat_c is not None:
                    gamma = self._hom.north_gamma(lon_c, lat_c)
                if gamma is not None:
                    proj_str += " +gamma=%.8f" % (gamma,)

            else:
                gamma = 0.0
                proj_str += " +no_rot +no_off"

            if reset_palette:
                self.leCentralPoint.setStyleSheet("")

            self.leResult.setText(proj_str)
            self.leResult.setEnabled(True)

            # Generate the WKT:
            if self.optimization_input is not None and gamma is not None:
                crs, lon, lat = self.optimization_input
                self.wkt = crs.get_projcrs_wkt(
                    self._hom, gamma, "<unnamed DOOMERCAT>", lon, lat
                )
                self.teWktResult.setText(self.wkt)
                self.teWktResult.setEnabled(True)
            else:
                self.teWktResult.setText("")
                self.teWktResult.setEnabled(False)


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
           = QgsCoordinateReferenceSystem(self.wkt)

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

        elif tab == 2:
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

    def orientNorthChanged(self, idx):
        """
        Called when the combo box defining the point of true northing
        changes selection.
        """
        if idx == 2:
            # The point picker line edit should be shown:
            self.leCentralPoint.setEnabled(True)
        else:
            # The point picker line edit should not be shown.
            self.leCentralPoint.setEnabled(False)

        if self._hom is not None:
            self.adjustProjStr()


    def centralPointEdited(self, text):
        """
        This slot is called when the text of the self.leCentralPoint
        line edit is edited by the user.
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


    def cbAlgorithmChanged(self, item):
        """
        This slot is called when the algorithm was changed.
        Depending on which algorithm is selected, different parameters
        become relevant, and the corresponding edits are activated.
        """
        if self.cbAlgorithm.currentText() == _ALGORITHM_PY:
            # Python.
            self.leBFGSEpsilon.setEnabled(False)
            self.sbNmaxPreAdamax.setEnabled(True)
        else:
            # C++
            self.leBFGSEpsilon.setEnabled(True)
            self.sbNmaxPreAdamax.setEnabled(False)


    def switchIcon(self, *args):
        if self._icon_path == ":/plugins/doomercat/icon":
            self._icon_path = ":/plugins/doomercat/icon2"
        else:
            self._icon_path = ":/plugins/doomercat/icon"

        self.action.setIcon(QIcon(self._icon_path+".png"))
        self.svg_widget.load(self._icon_path+".svg")
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
