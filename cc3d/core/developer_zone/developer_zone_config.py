# https://stackoverflow.com/questions/35071192/how-to-find-out-where-the-python-include-directory-is

import sys
import os
from pathlib import Path
import subprocess
import argparse
import cc3d
from sysconfig import get_paths
import sysconfig
import site
from pprint import pprint
import stat

from cc3d.core.utils import find_current_conda_env, find_conda
import tempfile


def get_conda_specs():
    conda_specs = {}
    conda_exec = find_conda()
    conda_env_name = find_current_conda_env(conda_exec=conda_exec)
    conda_shell_script = conda_exec.parent.parent.joinpath('etc', 'profile.d', 'conda.sh')

    conda_specs['conda_exec'] = conda_exec
    conda_specs['conda_env_name'] = conda_env_name
    conda_specs['conda_shell_script'] = conda_shell_script

    return conda_specs


def configure_developer_zone(cc3d_git_dir: Path, build_dir: Path):
    """
    Configures CC3D developer zone for compilation. Assumes that detected conda environment
    has compilers, cmake (>3.13) , swig installed. This function will work only
    when cc3d is installed from within conda environment

    :param cc3d_git_dir:
    :param build_dir:
    :return:
    """

    conda_specs = get_conda_specs()

    developer_zone_source = cc3d_git_dir.joinpath('CompuCell3D', 'DeveloperZone')
    build_dir.mkdir(exist_ok=True, parents=True)
    build_dir_content = os.listdir(build_dir)
    if len(build_dir_content):
        raise FileExistsError(f'Build Directory: {build_dir}  must be empty before we can use '
                              f'it for DeveloperZone configuration')

    ld_library = Path(sysconfig.get_config_var('LDLIBRARY'))
    lib_dir = Path(sysconfig.get_config_var('LIBDIR'))
    site_packages_dir = Path(site.getsitepackages()[0])

    python_exec = sys.executable
    install_dir = site_packages_dir

    py_include_dir= Path(sysconfig.get_config_var('INCLUDEPY'))
    bin_dir = Path(sysconfig.get_config_var('BINDIR'))

    cmake_exec = bin_dir.joinpath('cmake')


    swig_exec = bin_dir.joinpath('swig')

    if sys.platform.startswith('win'):
        raise RuntimeError('Unsupported yet')
    elif sys.platform.startswith('darwin'):
        cmake_generator_name = 'Unix Makefiles'
        cmake_c_compiler = bin_dir.joinpath('clang')
        cmake_cxx_compiler = bin_dir.joinpath('clang++')
    elif sys.platform.startswith('linux'):
        cmake_generator_name = 'Unix Makefiles'
        cmake_c_compiler = bin_dir.joinpath('gcc')
        cmake_cxx_compiler = bin_dir.joinpath('g++')
    else:
        raise RuntimeError(f'Unsupported platform {sys.platform}')

    cmd_cmake_generate = f'{cmake_exec} -G "{cmake_generator_name}" -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo ' \
            f'-DCMAKE_INSTALL_PREFIX:PATH={install_dir} ' \
            f'-DCMAKE_CXX_COMPILER:STRING={cmake_cxx_compiler} ' \
            f'-DCMAKE_C_COMPILER:STRING={cmake_c_compiler} ' \
            f'-DCOMPUCELL3D_GIT_DIR:PATH={cc3d_git_dir} ' \
            f'-DCOMPUCELL3D_INSTALL_PATH:PATH={install_dir} ' \
            f'-S {developer_zone_source} ' \
            f'-B {build_dir} ' \

    if sys.platform.startswith('darwin'):

        with tempfile.TemporaryDirectory() as tmpdirname:
            dev_zone_config_shell_script = Path(tmpdirname).joinpath('run_dev_config_script.sh')
            with dev_zone_config_shell_script.open('w') as out:
                out.write(f'#!/bin/sh\nsource {conda_specs["conda_shell_script"]} ; '
                  f'conda activate {conda_specs["conda_env_name"]} ; {cmd_cmake_generate}')
            dev_zone_config_shell_script.chmod(dev_zone_config_shell_script.stat().st_mode | stat.S_IEXEC)

            result = subprocess.run(f'{dev_zone_config_shell_script}', stdout=subprocess.PIPE)

        return result.stdout.decode('utf-8')
    else:
        raise RuntimeError(f'Unsupported platform {sys.platform}')


