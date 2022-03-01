# DOOMERCAT plugin dialogs.
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

from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import *
from qgis.PyQt.QtSvg import *
from qgis.PyQt.QtCore import Qt

from .graph import ScalarHistoryGraph


class SaveProjectionDialog(QDialog):
    """
    A simple dialog that queries for a projection name.
    """
    def __init__(self, *args, **kwargs):
        # Parent initialization:
        super().__init__(*args, **kwargs)
        self.setModal(True)
        self.setMinimumWidth(280)
        self.setWindowTitle("Enter User CRS Name")

        # Now populate some widgets:
        layout = QGridLayout(self)

        self.leName = QLineEdit(self)
        self.leName.textEdited.connect(self.textEdited)
        layout.addWidget(self.leName, 0, 0, 1, 2)
        self.btnAccept = QPushButton("&Accept",self)
        self.btnAccept.setEnabled(False)
        self.btnAccept.clicked.connect(self.accept)
        layout.addWidget(self.btnAccept, 1, 0, 1, 1)
        self.btnCancel = QPushButton("&Cancel",self)
        self.btnCancel.clicked.connect(self.reject)
        layout.addWidget(self.btnCancel, 1, 1, 1, 1)

    def showEvent(self, *args, **kwargs):
        """
        Override show to clear the name field.
        """
        self.leName.setText("")
        self.btnAccept.setEnabled(False)
        super().showEvent(*args, **kwargs)

    def textEdited(self, txt):
        """
        Slot that is called when the LineEdit text is
        edited.
        """
        if len(txt) == 0:
	        self.btnAccept.setEnabled(False)
        else:
	        self.btnAccept.setEnabled(True)

    def text(self):
        """
        Returns the line edit text:
        """
        return self.leName.text()


class ProgressDialog(QDialog):
    """
    A progress dialog with a cancel option.
    """
    def __init__(self, *args, **kwargs):
        # Parent initialization:
        super().__init__(*args, **kwargs)
        self.setModal(True)
#        self.setMinimumWidth(480)
#        self.setMinimumHeight(280)
        self.setWindowTitle("Optimization progress")

        # Now populate some widgets:
        layout = QGridLayout(self)
        self._graph = ScalarHistoryGraph()

        layout.addWidget(self._graph, 0, 0, 1, 2)

        self.btnCancel = QPushButton("&Cancel",self)
        self.btnCancel.clicked.connect(self.reject)
        layout.addWidget(self.btnCancel, 1, 1, 1, 1)

    def showEvent(self, *args, **kwargs):
        """
        Override show to clear the name field.
        """
        self._graph.clearHistory()
        self._graph.showSteps(True)

    def startReducePoints(self, N):
        """
        Slot to call when reduction of points starts.
        """
        self._graph.clearHistory()
        self._graph.showSteps(False)
        self._graph.addPoint(N)
        self.setWindowTitle("Reducing number of points...")

    def enterOptimization(self):
        """
        Slot to call when optimization starts.
        """
        self._graph.clearHistory()
        self._graph.showSteps(True)
        self.setWindowTitle("Optimizing Laborde oblique Mercator...")

    def pnormSet(self, pnorm):
        """
        Slot to call when pnorm is changed.
        """
        self._graph.addMarker("p=%.2f" % pnorm)

    def logCost(self, cost):
        self._graph.addPoint(cost)
