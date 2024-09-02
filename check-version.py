#!/bin/python
# Make sure that all version strings in this repository are consistent. This
# file is adapted from the REHEATFUNQ source code.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
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

# Version in README.md:
version_readme = None
with open('README.md','r') as f:
    state = 0
    for line in f:
        if state == 0:
            # Scroll until changelog:
            if line[:-1] == "## Changelog":
                state = 1
        elif state == 1:
            # Find the first version:
            if line[:16] == "### [Unreleased]":
                continue
            if line[:5] == "### [":
                version_readme = line[5:].split(']')[0]
                break

# Version in setup.py:
version_toml = None
with open('pyproject.toml','r') as f:
    state = 0
    for line in f:
        if state == 0:
            # Find the [project] tag:
            if line[:-1] == "[project]":
                state = 1
        elif state == 1:
            stripped = line.strip()
            if stripped[:10] == "version = ":
                quote = stripped[10]
                version_toml = stripped.split(quote)[1]
                break

# Version in the Sphinx documentation:
version_sphinx = None
with open('doc/conf.py','r') as f:
    for line in f:
        stripped = line.strip()
        if stripped[:10] == "release = ":
            quote = stripped[10]
            version_sphinx = stripped.split(quote)[1]

# Version in the Meson build file:
version_meson = None
with open('meson.build','r') as f:
    openparen = 0
    closeparen = 0
    found_project = False
    for line in f:
        stripped = line.strip()
        if not found_project:
            if stripped[:8] == 'project(':
                found_project = True
        if found_project:
            if 'version :' in stripped:
                version_meson = stripped.split('version :')[1]\
                                   .split(',')[0].split(')')[0]
                version_meson = version_meson.split('\'')[1]

            openparen += stripped.count('(')
            closeparen += stripped.count(')')
            if openparen == closeparen:
                break

# Check if any could not be determined:
if version_readme is None:
    raise RuntimeError("Version in 'README.md' could not be determined.")
if version_toml is None:
    raise RuntimeError("Version in 'pyproject.toml' could not be determined.")
if version_sphinx is None:
    raise RuntimeError("Version in 'docs/conf.py' could not be determined.")
if version_meson is None:
    raise RuntimeError("Version in 'meson.build' could not be determined.")

# Compare:
if version_readme != version_toml:
    raise RuntimeError("Versions given in 'README.md' and 'pyproject.toml' "
                       "differ.")

if version_readme != version_sphinx:
    raise RuntimeError("Versions given in 'README.md' and 'docs/conf.py' "
                       "differ.")

if version_readme != version_meson:
    raise RuntimeError("Versions given in 'README.md' and 'meson.build' "
                       "differ.")

print(version_readme)