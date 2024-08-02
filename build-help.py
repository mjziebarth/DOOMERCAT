# QGIS plugin HTML help generation script. Translates `help.md`
# to a python source file containing a HTML string.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2021 Deutsches GeoForschungsZentrum Potsdam
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


# 1) Parse the markdown help first:

# Read the whole string and identify the locations of the double
# newlines:
with open('qgis-plugin/help.md','r') as f:
    help_md_s = f.read().split("\n\n")

# Join simple newlines:
def join_simple_newlines(lines):
    joined = ""
    for l in lines:
        if l[0] == "#":
            joined += l + "\\n"
        else:
            joined += l + " "
    return joined

for l in help_md_s:
    help_md = "\\n\\n".join([join_simple_newlines(l.splitlines()) for l in help_md_s])


# 2) HTML help (derived from markdown):
help_html = help_md.replace("\\n\\n","\\n")

# Replace bold letters:
help_html_s = help_html.split("**")
help_html = help_html_s[0]
begin = True
for s in help_html_s[1:]:
    if begin:
        help_html += "<b>" + s
    else:
        help_html += "</b>" + s
    begin = not begin

# Replace italics:
help_html_s = help_html.split("*")
help_html = help_html_s[0]
begin = True
for s in help_html_s[1:]:
    if begin:
        help_html += "<i>" + s
    else:
        help_html += "</i>" + s
    begin = not begin

# Replace code:
help_html_s = help_html.split("`")
help_html = help_html_s[0]
begin = True
for s in help_html_s[1:]:
    if begin:
        help_html += "<span style='background-color:#eeeeee;'>" + s
    else:
        help_html += "</span>" + s
    begin = not begin


# Replace headlines:
help_html_s = help_html.split("\\n")
help_html = ""
for line in help_html_s:
    if len(line) == 0:
        continue
    if line[0] == '#':
        # Am in headline:
        line_s = line.split('#')
        i = len(line_s)-1
        help_html += "<h" + str(i) + ">" + line_s[-1] + "</h" + str(i) + ">\\n"
    else:
        # Am in paragraph:
        help_html += "<p>" + line + "</p>\\n"

help_file = "help_html = \"" + help_html + "\"\n"


# Write out the two string definitions into a python file that
# can be imported from the plugin:
with open('build/doomercat_plugin/help.py','w') as f:
    f.write(help_file)
