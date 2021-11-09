# A worker class to run the optimization algorithm in
# a separate thread.
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

from qgis.PyQt.QtCore import QRunnable, QObject, pyqtSlot, pyqtSignal
from .doomercat import LabordeObliqueMercator, OptimizeError


class OptimizationSignals(QObject):
    """
    Custom optimization signals.
    """
    result = pyqtSignal(LabordeObliqueMercator)
    error = pyqtSignal(Exception)
    progress = pyqtSignal(float)
    reducePoints = pyqtSignal(int)
    enterOptimization = pyqtSignal()
    finished = pyqtSignal()
    pnorm = pyqtSignal(float)


class OptimizationWorker(QRunnable):
    """
    A worker that does a Laborde oblique Mercator
    optimization for a data set.
    """
    # Signals:

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs
        self._pnorm = kwargs["pnorm"]
        self.signals = OptimizationSignals()
        self._exit = False


    def run(self):
        """
        Run this worker.
        """
        try:
            lom = LabordeObliqueMercator(*self._args, **self._kwargs,
                                         logger=self)
            self.signals.result.emit(lom)
        except OptimizeError as err:
            self.signals.error.emit(err)
        except Exception as err:
            self.signals.error.emit(err)
        else:
            pass
        finally:
            self.signals.finished.emit()


    def performExit(self):
        """
        Signals this optimization worker to perform an exit.
        """
        self._exit = True


    def log(self, level, msg):
        """
        Implements a logging interface similar to the python
        logging interface.
        """
        if "cost=" in msg:
            cost = float(msg.split("cost=")[-1])
            self.signals.progress.emit(cost)
        elif "Reducing points to " in msg:
            N = int(msg.split("Reducing points to ")[-1])
            self.signals.reducePoints.emit(N)
        elif "remaining=" in msg:
            N = int(msg.split("remaining=")[-1])
            self.signals.progress.emit(N)
        elif msg == "Enter optimization loop.":
            self.signals.enterOptimization.emit()
        elif "Optimize pnorm=" in msg and self._pnorm > 2:
            p = float(msg.split("pnorm=")[-1])
            self.signals.pnorm.emit(p)


    def exit(self):
        """
        Can be queried in the event loop.
        """
        return self._exit
