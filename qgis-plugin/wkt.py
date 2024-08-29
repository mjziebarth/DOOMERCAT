# WKT-parsing for geographical coordinate systems.
#
# Author: Malte J. Ziebarth (malte.ziebarth@tum.de)
#
# Copyright (C) 2024 Technical University of Munich
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

import numpy as np
from numpy._core.defchararray import isalpha, isspace

def _wkt_level(wkt: str) -> list[int]:
    """
    Parses the bracket-level of each character. This counts opening
    and closing brackets---lifting and lowering the level,
    respectively---while accounting for quotes (which may contain brackets
    that are not part of the WKT syntax).
    """
    l = 0
    level = []
    in_quote = False
    for i,c in enumerate(wkt):
        li = l
        if c == '[':
            if not in_quote:
                l += 1
        elif c == ']':
            if not in_quote:
                l -= 1
                li = l
        elif c == '"':
            in_quote = not in_quote
        level.append(li)

    return level


class WktNode:
    """
    A node of the WKT.
    """
    tag: str
    content: "list[str | int | float | WktNode]"
    wkt: str
    def __init__(self,
            tag: str,
            content: "list[str | int | float | WktNode]",
            wkt: str
        ):
        self.tag = tag
        self.content = content
        self.wkt = wkt

    def __repr__(self):
        return self.tag


def find_node_end(wkt: str, level: list[int], i: int):
    """
    Determine the end of the node starting at position i in wkt.
    """
    # Position of the bracket:
    ibr = wkt.find('[', i+1)

    # Level inside the bracket:
    l = level[ibr+1]

    # From the first char after the bracket, find the first position
    # that goes back to the original level:
    i1 = level.index(l-1, ibr+1)

    return i1


def alpha_end(wkt: str, i: int) -> int:
    """
    End of an alpha-valued string.
    """
    n = len(wkt)
    while i < n and wkt[i].isalpha():
        i += 1
    return i



def parse_wkt_recursively(
        wkt: str,
        level: list[int],
        parent: str,
        wkt_dict: dict[str, list[WktNode]]
    ) -> WktNode:
    """
    This function parses a WKT recursively. Note: this function is expecting
    a non-whitespace version of the WKT. Also, it is expecting the part of
    the WKT *except* for the closing bracket of the root node (e.g. GEOGCRS)
    """
    # Determine the keyword of this node:
    itag = wkt.find('[', 1)
    tag = wkt[:itag]

    # Determine the key of this node:
    if len(parent) == 0:
        key = tag
    else:
        key = parent + "." + tag
    parsed = []
    i = itag+1
    n = len(wkt)
    while i < n:
        c = wkt[i]
        # Identify what comes next:
        if c.isspace() or c == ',':
            i += 1
            continue
        elif c == '"':
            # string.
            i0 = i
            i1 = wkt.index('"', i0+1)
            # Handle double quotes:
            while i1+1 < n and wkt[i1+1] == '"':
                i1 = wkt.index('"', i1+2)
            parsed.append(wkt[i0+1:i1])
            i = i1+1
        elif c.isdigit() or c == '-':
            # number
            i0 = i
            i += 1
            while i < n:
                c = wkt[i]
                if c.isdigit() or c == '.':
                    i += 1
                else:
                    break
            num_str = wkt[i0:i]
            if '.' in num_str:
                parsed.append(float(num_str))
            else:
                parsed.append(int(num_str))
        elif c.isalpha():
            # keyword.
            # Either node or an enum.
            iae = alpha_end(wkt, i)
            if iae < n and wkt[iae] == '[':
                i1 = find_node_end(wkt, level, i)
                sub = parse_wkt_recursively(
                    wkt[i:i1], level[i:i1], key, wkt_dict
                )
                parsed.append(sub)
                i = i1+1
            else:
                # enum / constant:
                parsed.append(wkt[i:iae])
                i = iae
        else:
            raise RuntimeError(
                "Could not parse substring '" + wkt + "' because of offending "
                "character '" + c + "' at an unexpected position (" + str(i)
                + ")."
            )

    # Assemble the node:
    node = WktNode(tag, parsed, wkt)

    # Attach the node to its key:
    if key in wkt_dict:
        wkt_dict[key].append(node)
    else:
        wkt_dict[key] = [node]

    return node


def parse_wkt(wkt: str) -> tuple[WktNode, dict[str, list[WktNode]]]:
    """
    WktParser
    """
    # The parantheses level:
    level = _wkt_level(wkt)

    # The parsed dict:
    wkt_dict: dict[str, list[WktNode]] = dict()


    # The top layer is always (if valid CRS Wkt) a one-element list.
    i1 = find_node_end(wkt, level, 0)
    root = parse_wkt_recursively(wkt[:i1], level[:i1], "", wkt_dict)
    return root, wkt_dict



def _find_tag(
        tag: str,
        nodes: list[str | float | int | WktNode]
    ) -> WktNode | None:
    """
    This function finds a tag of a certain name within a list of WktNodes
    and values (str | float | int).
    """
    for n in nodes:
        if isinstance(n, WktNode) and n.tag == tag:
            return n


def _parse_a_f(wkt_dict: dict[str, list[WktNode]], crs_tag: str):
    """
    Parses the ellipsoid node.
    """
    # One function to parse the ellipsoid WKT node once we've found the
    # correct location (key):
    def handle_ellipsoid_node(key: str):
        ellps = wkt_dict[key][0]
        a, finv = ellps.content[1:3]
        if not (isinstance(a, (int,float))
                and isinstance(finv, (int,float))
        ):
            print(a, finv)
            raise TypeError("Wrong type in a or inverse f")

        # Handle possibly given length unit:
        if len(ellps.content) > 2:
            ua = ellps.content[2]
            if (isinstance(ua, WktNode) \
                    and ua.tag in ("LENGTHUNIT", "UNIT")
            ):
                # We may parse the length unit!
                # Something like 'UNIT["metre",1.0].....]'
                convfactor = ua.content[1]
                if not isinstance(convfactor, (float,int)):
                    raise TypeError(
                        "Wrong type in length unit factor"
                    )
                a *= convfactor

        # Handle the 'artificial value of zero' for the sphere
        # (section '8.2.1 Ellipsoid' of OGC 18-010r11):
        if finv == 0.0:
            f = 0.0
        else:
            f = 1.0 / finv

        # Save the unparsed string:
        return a, f, ellps.wkt

    # Both 'SPHEROID' and 'ELLIPSOID' are allowed, see section
    # '8.1.2 Ellipsoid' of OGC 18-010r11.
    # Also, we might find the ellipsoid either in the DATUM or the ENSEMBLE
    # section of the GEOGCRS (latter, for instance, in the ubiquitous EPSG:4326)
    for ellpskey in ("ELLIPSOID","SPHEROID"):
        # Might be a datum or a datum ensemble:
        for datum in ("DATUM","ENSEMBLE"):
            key = ".".join((crs_tag,datum,ellpskey))
            if key in wkt_dict:
                return handle_ellipsoid_node(key)


    raise RuntimeError("Did not find ellipsoid.")


def _parse_cs_axes(wkt_dict: dict[str, list[WktNode]], crs_tag: str, wkt: str):
    """
    Parses the CS and AXIS nodes.
    """
    # Step 1: ensure that we can find the CS tag:
    key = crs_tag + ".CS"
    if key not in wkt_dict:
        raise RuntimeError(
            "The required 'CS' tag is not present in WKT '" + wkt + "'."
        )

    # As by '7.5.2 Coordinate system type, dimension and coordinate
    # data type' and '7.5.4 Axis direction' of OGC 18-010r11, the
    # geographic CRS must be of type ellipsoidal and must have
    # 2 or 3 dimensions. Two dimensions must be lat (north) and
    # lon (east). In the 3d case, a third dimension must be
    # ellipsoidal height (up).
    # Hence, the question of elevation boils down to whether we have
    # two or three dimensions.
    # TODO: '7.6 Datum ensemble' mentions the ENSEMBLE only in combination with
    #       the geodetic and vertical CRSes. Is the above decision rule still
    #       valid for those geodetic datum ensemble CRSes? E.g. EPSG 4326
    #       as a very prominent and important example.
    cs = wkt_dict[key][0]
    if cs.content[0] != "ellipsoidal":
        raise RuntimeError(
            "CS type must be ellipsoidal"
        )
    Nax = cs.content[1]
    if Nax in (2,3):
        has_elevation = Nax == 3
    else:
        raise RuntimeError(
            "Wrong number of axes for ellipsoidal CRS. Must be 2 or 3."
        )

    # Now find the axes:
    key = crs_tag + ".AXIS"
    if key not in wkt_dict:
        raise RuntimeError(
            "The required 'AXIS' tag is not present in WKT '" + wkt
            + "'."
        )
    axes = wkt_dict[key]
    if len(axes) != Nax:
        raise RuntimeError(
            "The number of AXIS nodes (" + str(len(axes)) + ") does "
            "not correspond to the number of axes in CS definition ("
            + str(Nax) + ") in WKT '" + wkt + "'."
        )

    # Ensure we have the correct axes and simultaneously figure out the
    # order. We do this by iterating the AXIS tags that have already been
    # collected in `axes`.
    AXES_REQ = ("east", "north", "up")[:Nax]
    axes_real = []
    elev_index = None
    lon_index = lat_index = 0
    for i,ax in enumerate(axes):
        # First make sure that the axis is one of the required,
        # and that none appears twice:
        ax_dir = str(ax.content[1])
        if ax_dir not in AXES_REQ:
            raise RuntimeError(
                "Found an axis direction '" + ax_dir + "' that is not "
                "valid for a geographic CRS in WKT '" + wkt + "'."
            )
        if ax_dir in axes_real:
            raise RuntimeError(
                "Axis direction '" + ax_dir + "' is multiply defined "
                "in WKT '" + wkt + "'."
            )
        axes_real.append(ax_dir)

        # Now figure out the relevant order:
        order = _find_tag("ORDER", ax.content)
        o: int
        if order is None:
            # Defaults back to the order in which the AXIS tags are listed
            # if no ORDER tag is given (backwards compatibility mentioned)
            # in OGC 18-010r11).
            o = i
        else:
            o_ = order.content[0]
            if not isinstance(o_, int):
                raise RuntimeError(
                    "Wrong value '" + str(o_) + "' for ORDER tag in "
                    "WKT '" + wkt + "'."
                )
            o = o_
        if ax_dir == "east":
            lon_index = o
        elif ax_dir == "north":
            lat_index = o
        else:
            elev_index = o

    return has_elevation, lon_index, lat_index, elev_index




class WktCRS:
    """
    Representation of a geographic CRS.
    """
    a: float
    f: float
    is_geographic: bool
    has_axis_inverted: bool
    has_elevation: bool
    projection: str | None
    ellipsoid_wkt: str | None
    lon_index: int
    lat_index: int
    elev_index: int | None

    def __init__(self, wkt: str):
        # Parse:
        root, wkt_dict = parse_wkt(wkt)
        self.ellipsoid_wkt = None
        self.is_geographic = False
        self.has_axis_inverted = False
        self.has_elevation = False
        self.elev_index = None

        # Ellipsoid & axis order parameters:
        GEOCRS_KEYWORDS = set(("GEOGCRS", "GEOGCS"))
        if root.tag in GEOCRS_KEYWORDS:
            # Geographic CRS.
            self.projection = None
            self.is_geographic = True

            # 1) Ellipsoid.
            self.a, self.f, self.ellipsoid_wkt = _parse_a_f(
                wkt_dict, root.tag
            )

            # 2) Axes.
            #
            self.has_elevation, self.lon_index, self.lat_index,\
                self.elev_index \
                = _parse_cs_axes(
                    wkt_dict, root.tag, wkt
                )

            self.has_axis_inverted = self.lon_index > self.lat_index


        else:
            raise RuntimeError("Only geographic CRS supported.")