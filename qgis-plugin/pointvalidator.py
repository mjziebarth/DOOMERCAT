# QValidator that ensures that a line edit contains QGIS "copy point
# coordinates"-formatted coordinate input.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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

from qgis.PyQt.QtGui import QValidator

class CoordinateValidator(QValidator):
    """
    A validator that ensures that text entered into a line
    edit is formatted as a pair of coordinates.
    """
    def validate(self, text, cursor_pos):
        """
        Ensure that the text is of the form
           NUMBER, NUMBER
        """
        # A function that aims to evaluate the validation status of
        # one split, i.e. one decimal number:
        def validate_split(s, dmin, dmax):
            try:
                # Try to convert to a floating point number:
                d = float(s)
                if d < dmin:
                    return 1
                elif d > dmax:
                    return 1
                return 0
            except ValueError:
                if not all(c in set(('0','1','2','3','4','5','6','7','8',
                                     '9','.','-'))
                           for c in s):
                    return 2
                if s.count('.') > 1:
                    return 2
                if s.count('-') > 1:
                    # We use only fixed-point decimal numbers:
                    return 2
                return 1

        # Now split the string at the comma, which denotes the transition
        # between two numbers:
        split = text.split(',')
        if len(split) > 2:
            return (QValidator.Invalid, text, cursor_pos)
        elif len(split) == 2:
            status = 0
            for dmin, dmax, s in zip([-180.0, -90.0], [360.0, 90.0], split):
                status = max(validate_split(s, dmin, dmax), status)
                if status == 2:
                    return (QValidator.Invalid, text, cursor_pos)
            if status == 0:
                return (QValidator.Acceptable, text, cursor_pos)
            else:
                return (QValidator.Intermediate, text, cursor_pos)
        elif len(split) == 1:
            status = validate_split(split[0], -180.0, 360.0)
            if status == 2:
                return (QValidator.Invalid, text, cursor_pos)

        return (QValidator.Intermediate, text, cursor_pos)

