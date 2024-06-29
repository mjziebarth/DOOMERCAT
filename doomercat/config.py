# User-defined default configuration for the HotineObliqueMercator class.
# This file is part of the DOOMERCAT python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Malte J. Ziebarth,
#               2024 Technical University of Munich
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

from math import inf
from configparser import ConfigParser

def _generate_default_config():
    """
    A method to generate the default configuration.
    """
    _default = {}
    _default["backend"] = "Python"
    _default["pnorm"] = 2
    _default["k0_ap"] = 0.0
    _default["sigma_k0"] = inf
    _default["ellipsoid"] = None
    _default["a"] = None
    _default["f"] = None
    _default["Nmax"] = 1000
    _default["proot"] = False
    _default["fisher_bingham_use_weight"] = False
    _default["compute_enclosing_sphere"] = True
    _default["bfgs_epsilon"] = 0.0
    _default["Nmax_pre_adamax"] = 100
    _default["return_full_history"] = False
    return _default


try:
    import appdirs

    def _init_file():
        return appdirs.user_config_dir("DOOMERCAT")

    def _load_defaults():
        """
        Load the default parameters.
        """
        # load the init file:
        conf = ConfigParser()
        conf.read(_init_file())
        defaults = {key : val for key,val in conf["default"].items()}

        # Convert parameters from the string-based handling
        # within the configparser module:
        pnorm = defaults["pnorm"]
        defaults["pnorm"] = int(pnorm) if pnorm != "inf" else inf
        defaults["k0_ap"] = float(defaults["k0_ap"])
        defaults["sigma_k0"] = float(defaults["sigma_k0"])
        ellipsoid = defaults["ellipsoid"]
        defaults["ellipsoid"] = ellipsoid if ellipsoid != "None" else None
        a = defaults["a"]
        defaults["a"] = float(a) if a != "None" else None
        f = defaults["f"]
        defaults["f"] = float(f) if f != "None" else None
        defaults["Nmax"] = int(defaults["nmax"])
        del defaults["nmax"]
        defaults["proot"] = bool(defaults["proot"])
        defaults["fisher_bingham_use_weight"] \
           = bool(defaults["fisher_bingham_use_weight"])
        defaults["compute_enclosing_sphere"] \
           = bool(defaults["compute_enclosing_sphere"])
        defaults["bfgs_epsilon"] = float(defaults["bfgs_epsilon"])
        defaults["Nmax_pre_adamax"] = int(defaults["nmax_pre_adamax"])
        del defaults["nmax_pre_adamax"]
        defaults["return_full_history"] = bool(defaults["return_full_history"])
        return defaults

    def save_defaults():
        """
        Save the default keyword argument configuration for the
        HotineObliqueMercator class.
        """
        config = ConfigParser()
        config["default"] = {key : str(val) for key,val in _default.items()}
        with open(_init_file(),'w') as f:
            config.write(f)

    _default = _load_defaults()


except ImportError:
    _default = _generate_default_config()

    def save_defaults():
        """
        Save the defaults. Warning: the `appdirs` package is not
        installed so `save_defaults` does not work.
        """
        raise ImportError("`save_defaults` needs the `appdirs` Python "
                          "package to be installed.")


except KeyError:
    _default = _generate_default_config()
    from warnings import warn
    warn("Warning: Configuration file is not in valid format.")


def change_default(key, value):
    """
    Sets a default configuration value, that is, default keyword
    arguments for the HotineObliqueMercator class. The change can
    be made permanent with a call to `save_defaults()` if the
    `appdirs` package is installed.
    """
    global _default
    _default[key] = value


def reset_defaults():
    """
    Reset the default keyword argument configuration for the
    HotineObliqueMercator class.
    """
    global _default
    _default = _generate_default_config()
    save_defaults()