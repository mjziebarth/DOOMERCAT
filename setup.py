# Setup script for the DOOMERCAT python module.
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

# Imports:
from setuptools import setup, Command
from setuptools.command.build import build, SubCommand


import os
import subprocess
from pathlib import Path
from shutil import copyfile


DIR_LINUX = 'builddir'
DIR_CROSS = 'builddir-mingw'

# Compile the C++ code:
def setup_compile(build_lib: str):
    """
    This function can be called from the setup script to compile with
    the Meson backend.
    """
    print("build_lib:",build_lib)
    # First setup the meson directories:
    parent = Path('.').resolve()
    destination = Path(build_lib)
    def meson_setup(*options: str, force=False):
        os.chdir(parent)
        if not (parent / DIR_LINUX).is_dir() or force:
            cmd = ('meson', 'setup', *options, DIR_LINUX)
            subprocess.run(cmd, check=True)
        if not (parent / DIR_CROSS).is_dir() or force:
            cmd = ('meson','setup','--cross-file','x86_64-w64-mingw32.txt',
                   *options, DIR_CROSS)
            subprocess.run(cmd, check=True)

    meson_setup()

    # Now compile:
    def compile():
        os.chdir(parent / DIR_LINUX)
        cmd = ('meson','compile')
        subprocess.run(cmd, check=True)

        os.chdir(parent / DIR_CROSS)
        subprocess.run(cmd, check=True)

    recompile = True
    try:
        compile()
        recompile = False
    except:
        # Most probably the build directory was generated by an old
        # Meson version or is otherwise corrupt.
        # Wipe the build directory and rebuild:
        pass

    if recompile:
        meson_setup('--reconfigure', force=True)
        compile()

    os.chdir(parent)

    # Now copy the shared libraries:
    copyfile(
        parent / DIR_LINUX / "lib_cppextensions.so",
        destination / "doomercat" / "_cppextensions.so"
    )
    copyfile(
        parent / DIR_CROSS / "lib_cppextensions.dll",
        destination / "doomercat" / "_cppextensions.dll"
    )

    subprocess.run(("pwd"))
    subprocess.run(("ls",str((destination / "doomercat").resolve())))



class BuildDoomercatCommand(SubCommand, Command):
    """
    Custom command to build the Doomercat shared libraries.
    """
    parent: Path

    def initialize_options(self) -> None:
        self.parent = Path('.').resolve()

    def finalize_options(self) -> None:
        pass

    def get_source_files(self) -> list[str]:
        """
        Parse the meson.build file to obtain the list of sources.
        """
        with open('meson.build','r') as f:
            meson_build = f.readlines()
        sources = []
        in_sources = False
        for line in meson_build:
            if line.startswith("sources = ["):
                in_sources = True

            if in_sources:
                sources.append(line.split('[')[-1].split(']')[0])

            if in_sources and line.strip().endswith("]"):
                in_sources = False

        sources = [s.strip() for src in sources for s in src.split(',')]
        print("sources:",sources)
        return sources

    def run(self) -> None:
        setup_compile(self.get_finalized_command("build_py").build_lib)

    def get_outputs(self, include_bytecode=True) -> list[str]:
        outputs = [
            str((self.parent / DIR_LINUX / "lib_cppextensions.so")
                 .relative_to(self.parent)),
            str((self.parent / DIR_CROSS / "lib_cppextensions.dll")
                 .relative_to(self.parent))
        ]
        return outputs



class DummyBuild(build):
    """
    Places the build_doomercat at the end of the build command queue,
    enabling it to build and copy the shared libraries into the existing
    build directory.
    """
    sub_commands = build.sub_commands + [('build_doomercat', None)]

    def __init__(self, *args, **kwargs):
        print("sub_commands:", DummyBuild.sub_commands)
        super().__init__(*args, **kwargs)


# Setup:

setup(
    cmdclass = {
        'build' : DummyBuild,
        'build_doomercat' : BuildDoomercatCommand
    }
)
