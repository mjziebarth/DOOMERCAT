# Dynamic two-dimensional history graphing widget.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2020 Deutsches GeoForschungsZentrum Potsdam
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

from qgis.PyQt.QtGui import QPainterPath, QPen, QPainter, QPalette, QPolygonF
from qgis.PyQt.QtWidgets import QWidget
from qgis.PyQt.QtCore import QPoint, Qt, QPointF, QRect


class ScalarHistoryGraph(QWidget):
    """
    Simple widget to render the time history of a scalar.
    """
    def __init__(self, *args, **kwargs):
        self._hist = []
        self._markers = []
        self._maxY = 0.0
        self._show_steps = True
        super().__init__(*args, **kwargs)
        self.setMinimumWidth(400)
        self.setMinimumHeight(100)

    def addPoint(self, y):
        """
        Adds another point to this graph.
        """
        self._hist += [y]
        self._maxY = max(self._maxY, y)
        self.repaint()

    def addMarker(self, text):
        """
        Adds a marker at the current point.
        """
        self._markers += [(len(self._hist), text)]
        self.repaint()

    def clearHistory(self):
        """
        Clear the history.
        """
        self._hist = []
        self._markers = []
        self._maxY = 0.0
        self.repaint()

    def showSteps(self, show):
        """
        Configure whether to show steps.
        """
        self._show_steps = show

    def paintEvent(self, event):
        """
        Overridden paint event.
        """
        super().paintEvent(event)

        # Early exit if nothing to paint:
        N = len(self._hist)
        if N < 2:
            return

        # Obtain the widget's coordinates:
        rect = self.rect()

        # 1) Create the painter path:
        poly = QPolygonF()
        max_y = self._maxY
        height = rect.height()
        width = rect.width()
        qy = height / max_y
        qx = width / max(N, 200)
        poly.append(QPointF(0, max_y - qy * self._hist[0]))
        for i in range(1, N):
            poly.append(QPointF(qx*i, qy * (max_y - self._hist[i])))

        # Finish the polygon:
        poly.append(QPointF(qx*(N-1), rect.height()))
        poly.append(QPointF(0, rect.height()))

        # Select brushes:
        brush0 = QPalette().window()
        brush1 = QPalette().mid()
        if brush0.color() == brush1.color():
            # In Fusion::Night Mapping, the window() and mid() of the palette
            # are the same, so that the cost function visualization does not
            # actually work.
            # Choose another brush color from the palette in this case:
            brush1 = QPalette().windowText()

        # Paint:
        path = QPainterPath()
        path.addPolygon(poly)
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(rect, brush0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(brush1)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.drawPath(path)

        # 2) Draw informative text:
        painter.setPen(QPen())
        steps_advance = 0
        if self._show_steps:
            painter.drawText(rect, Qt.AlignTop | Qt.AlignRight, str(N) + " steps")
            steps_advance = 1.05*self.fontMetrics().height()

        # 3) Draw markers:
        ilast = None
        for i,text in self._markers[::-1]:
            painter.drawLine(qx*i, 0, qx*i, height)

            # Prevent painting over following text:
            if ilast is not None:
                textwidth = self.fontMetrics().width(text)
                if qx*i + 1.1*textwidth >= qx*ilast:
                    ilast = i
                    continue

            r = QRect(qx*i+1, steps_advance, width-qx*i-1, height-steps_advance)
            painter.drawText(r, Qt.AlignTop | Qt.AlignLeft, text)
            ilast = i

        painter.end()
