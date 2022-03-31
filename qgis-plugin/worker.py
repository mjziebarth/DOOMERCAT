# A worker class to run the optimization algorithm in
# a separate thread.
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

import sys
import platform
import subprocess
from pathlib import Path
from pickle import Pickler, Unpickler
from tempfile import TemporaryDirectory
from qgis.PyQt.QtCore import QRunnable, QObject, pyqtSlot, pyqtSignal
from .messages import info
from .moduleloader import _plugin_dir, HotineObliqueMercator


# Obtain the path to the correct python executable:
def _get_python_exec():
    system = platform.system()
    if system == 'Linux':
        return str(sys.executable)

    elif system == 'Windows':
        # Seems like QGIS on Windows bundles everything together
        # so we should be able to just call python:
        return "python"
    else:
        raise NotImplementedError("Only Linux and Windows implemented.")

def _get_creationflags():
    system = platform.system()
    if system == 'Linux':
        return 0
    elif system == 'Windows':
        # Do not create a window
        return subprocess.CREATE_NO_WINDOW
    else:
        return 0


_python_cmd = _get_python_exec()
_creationflags =_get_creationflags()

# Constructing the command to execute the python file:
def _get_python_file_cmd(path):
    system = platform.system()
    if system == 'Linux':
        return str(path)

    elif system == 'Windows':
        return str(path)

    else:
        raise NotImplementedError("Only Linux and Windows implemented.")



class OptimizationSignals(QObject):
    """
    Custom optimization signals.
    """
    result = pyqtSignal(HotineObliqueMercator)
    error = pyqtSignal(Exception)
    progress = pyqtSignal(float)
    reducePoints = pyqtSignal(int)
    enterOptimization = pyqtSignal()
    finished = pyqtSignal()
    pnorm = pyqtSignal(float)


class OptimizationWorker(QRunnable):
    """
    A worker that performs a Hotine oblique Mercator
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
        # Create a temporary file:
        try:
            with TemporaryDirectory() as tmpdir:
                fpath = Path(tmpdir) / "args.pickle"
                with open(fpath,'wb') as f:
                    Pickler(f).dump((self._args, self._kwargs))

                working_dir = _plugin_dir / "module"
                script_path = _plugin_dir / "process.py"
                cmd = [_python_cmd, _get_python_file_cmd(script_path),
                       str(fpath), str(working_dir)]
                result = subprocess.run(cmd, capture_output=True,
                                        text=True, creationflags=_creationflags)

                # Check whether subprocess failed:
                if result.returncode != 0:
                    raise RuntimeError("Subprocess error:\n"
                                       + str(result.stderr))

                # Load the Hotine oblique Mercator:
                with open(fpath,'rb') as f:
                    HOM = Unpickler(f).load()

            # Return the result:
            self.signals.result.emit(HOM)


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
