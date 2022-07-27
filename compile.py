# Compilation of the C++ dynamic library for setting up the Python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam
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

import os
import subprocess

# Source and header files:
sources = ['hotine.cpp', 'functions.cpp', 'arithmetic.cpp', 'dataset.cpp',
           'ctypesinterface.cpp', 'linalg.cpp', 'cost_hotine.cpp']
sources = ['cpp/src/' + s for s in sources]

include = 'cpp/include/'

#
# Compile routines for linux and windows:
#


def compile_linux(compiler: str = 'g++', native: bool = True,
                  openmp: bool = True) -> str:
    """
    Compiles the C++ code into the dynamic library
    'doomercat/_cppextensions.so'.
    """
    name = '_cppextensions'
    if compiler == 'g++':
        cmd = ['g++', '-I'+include, '-O3', '-g0','-fPIC']
        if native:
            cmd.append('-march=native')
            name = '_cppextensions_native'
        if openmp:
            cmd.append('-fopenmp')
        libname = name + '.so'
        cmd += sources + ['-shared', '-o', 'doomercat/' + libname,
                          '-Wl,-z,defs']
    else:
        raise NotImplementedError("Only g++ compilation implemented.")

    process = subprocess.run(cmd)
    if process.returncode != 0:
        raise RuntimeError("compile_linux failed.")

    return libname


def cross_compile_win(compiler: str = 'mingw-g++') -> str:
    """
    Compiles the C++ code into the dynamic library
    'doomercat/_cppextensions.dll'.

    Uses mingw g++ to cross compile from a linux machine.
    """
    if compiler == 'mingw-g++':
        cmd = ['x86_64-w64-mingw32-g++', '-I'+include, '-O3', '-g0', '-fPIC',
               '-fopenmp', '-DMS_WIN64', '-static-libgcc',
               '-static-libstdc++',
               '-Wl,-Bstatic,--whole-archive,--no-undefined', '-lwinpthread',
               '-lgomp','-Wl,-Bdynamic,--no-whole-archive']
        cmd += sources + ['-shared', '-o', 'doomercat/_cppextensions.dll']
    else:
        raise NotImplementedError("Only g++ compilation implemented.")

    process = subprocess.run(cmd)
    if process.returncode != 0:
        raise RuntimeError("cross_compile_win failed.")

    return '_cppextensions.dll'


def compile_win(compiler: str = '') -> str:
    """
    Compiles the C++ code into the dynamic library
    'doomercat/_cppextensions.dll'.

    Compiles natively in windows.
    """
    raise NotImplementedError("No native Windows compilation implemented.")

    return ''


def setup_compile() -> str:
    """
    This function can be called from the setup script to compile according to
    the current machine.
    """
    # Switch between operating systems:
    if os.name == 'nt':
        libname = compile_win()
    elif os.name == 'posix':
        # Assumes posix == linux, which is not true.
        # FIXME for later in case this becomes important.
        libname = compile_linux()

    return libname
