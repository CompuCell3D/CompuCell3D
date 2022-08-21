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

    build_dir.mkdir(exist_ok=True, parents=True)
    build_dir_content = os.listdir(build_dir)
    if len(build_dir_content):
        raise FileExistsError(f'Build Directory: {build_dir}  must be empty before we can use '
                              f'it for DeveloperZone configuration')

    if sys.platform.startswith('darwin'):
        output = configure_developer_zone_mac(cc3d_git_dir=cc3d_git_dir, build_dir=build_dir, conda_specs=conda_specs)
    elif sys.platform.startswith('win'):
        output = configure_developer_zone_win(cc3d_git_dir=cc3d_git_dir, build_dir=build_dir, conda_specs=conda_specs)
    elif sys.platform.startswith('linux'):
        output = configure_developer_zone_linux(cc3d_git_dir=cc3d_git_dir, build_dir=build_dir, conda_specs=conda_specs)
    else:
        raise RuntimeError(f'Unsupported Platform: {sys.platform}')

    return output


def configure_developer_zone_win(cc3d_git_dir: Path, build_dir: Path, conda_specs: dict):
    """

    @param cc3d_git_dir:
    @param build_dir:
    @param conda_specs:
    @return:
    """

    developer_zone_source = cc3d_git_dir.joinpath('CompuCell3D', 'DeveloperZone')

    paths_dict = sysconfig.get_paths()
    activate_script = conda_specs['conda_exec'].parent.joinpath('activate.bat')
    conda_env_name = conda_specs['conda_env_name']
    py_version_nodot = sysconfig.get_config_var('py_version_nodot')

    stdlib = Path(paths_dict['stdlib'])
    bin_dir = stdlib.parent.joinpath('Library', 'bin')
    ld_library = stdlib.parent.joinpath('libs', f'python{py_version_nodot}.lib')

    site_packages_dir = Path(paths_dict['platlib'])

    install_dir = site_packages_dir

    python_include_dir = paths_dict['include']

    cmake_exec = bin_dir.joinpath('cmake.exe')

    cmake_generator_name = 'NMake Makefiles'

    cmd_cmake_generate = f'{cmake_exec} -G "{cmake_generator_name}" -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo ' \
                         f'-DCMAKE_INSTALL_PREFIX:PATH={install_dir} ' \
                         f'-DCOMPUCELL3D_GIT_DIR:PATH={cc3d_git_dir} ' \
                         f'-DCOMPUCELL3D_INSTALL_PATH:PATH={install_dir} ' \
                         f'-DPYTHON_INCLUDE_DIR:PATH={python_include_dir} ' \
                         f'-DPYTHON_LIBRARY:PATH={ld_library} ' \
                         f'-S {developer_zone_source} ' \
                         f'-B {build_dir} ' \

    result = subprocess.run(
        f'{activate_script} & conda activate {conda_env_name} & {cmd_cmake_generate}', stdout=subprocess.PIPE)
    out_str = result.stdout.decode('utf-8')
    print(out_str)

    return out_str


def configure_developer_zone_linux(cc3d_git_dir: Path, build_dir: Path, conda_specs: dict):
    """

    @param cc3d_git_dir:
    @param build_dir:
    @param conda_specs:
    @return:
    """
    developer_zone_source = cc3d_git_dir.joinpath('CompuCell3D', 'DeveloperZone')

    paths_dict = sysconfig.get_paths()

    conda_env_name = conda_specs['conda_env_name']
    installed_base = Path(sysconfig.get_config_var('installed_base'))

    ld_library = sysconfig.get_config_var('LDLIBRARY')
    libdir = sysconfig.get_config_var('LIBDIR')
    ld_library_abs_path = Path(libdir).joinpath(ld_library)

    bin_dir = installed_base.joinpath('bin')
    activate_script = bin_dir.joinpath('activate')

    site_packages_dir = Path(paths_dict['platlib'])

    install_dir = site_packages_dir

    python_include_dir = paths_dict['include']

    cmake_exec = bin_dir.joinpath('cmake')

    cmake_generator_name = 'Unix Makefiles'

    cmd_cmake_generate = f'{cmake_exec} -G "{cmake_generator_name}" -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo ' \
                         f'-DCMAKE_INSTALL_PREFIX:PATH={install_dir} ' \
                         f'-DCOMPUCELL3D_GIT_DIR:PATH={cc3d_git_dir} ' \
                         f'-DCOMPUCELL3D_INSTALL_PATH:PATH={install_dir} ' \
                         f'-DPYTHON_INCLUDE_DIR:PATH={python_include_dir} ' \
                         f'-DPYTHON_LIBRARY:PATH={ld_library_abs_path} ' \
                         f'-S {developer_zone_source} ' \
                         f'-B {build_dir} ' \

    result = subprocess.check_output(
        f'source {activate_script} ; conda activate {conda_env_name} ; {cmd_cmake_generate}', shell=True)

    out_str = result.decode('utf-8')
    print(out_str)

    return out_str


def configure_developer_zone_mac(cc3d_git_dir: Path, build_dir: Path, conda_specs: dict):
    developer_zone_source = cc3d_git_dir.joinpath('CompuCell3D', 'DeveloperZone')

    ld_library = Path(sysconfig.get_config_var('LDLIBRARY'))
    lib_dir = Path(sysconfig.get_config_var('LIBDIR'))
    site_packages_dir = Path(site.getsitepackages()[0])

    python_exec = sys.executable
    install_dir = site_packages_dir

    py_include_dir = Path(sysconfig.get_config_var('INCLUDEPY'))
    bin_dir = Path(sysconfig.get_config_var('BINDIR'))

    cmake_exec = bin_dir.joinpath('cmake')

    swig_exec = bin_dir.joinpath('swig')

    cmake_generator_name = 'Unix Makefiles'
    cmake_c_compiler = bin_dir.joinpath('clang')
    cmake_cxx_compiler = bin_dir.joinpath('clang++')

    cmd_cmake_generate = f'{cmake_exec} -G "{cmake_generator_name}" -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo ' \
                         f'-DCMAKE_INSTALL_PREFIX:PATH={install_dir} ' \
                         f'-DCMAKE_CXX_COMPILER:STRING={cmake_cxx_compiler} ' \
                         f'-DCMAKE_C_COMPILER:STRING={cmake_c_compiler} ' \
                         f'-DCOMPUCELL3D_GIT_DIR:PATH={cc3d_git_dir} ' \
                         f'-DCOMPUCELL3D_INSTALL_PATH:PATH={install_dir} ' \
                         f'-S {developer_zone_source} ' \
                         f'-B {build_dir} ' \

    with tempfile.TemporaryDirectory() as tmpdirname:
        dev_zone_config_shell_script = Path(tmpdirname).joinpath('run_dev_config_script.sh')
        with dev_zone_config_shell_script.open('w') as out:
            out.write(f'#!/bin/sh\nsource {conda_specs["conda_shell_script"]} ; '
                      f'conda activate {conda_specs["conda_env_name"]} ; {cmd_cmake_generate}')
        dev_zone_config_shell_script.chmod(dev_zone_config_shell_script.stat().st_mode | stat.S_IEXEC)

        result = subprocess.run(f'{dev_zone_config_shell_script}', stdout=subprocess.PIPE)

    return result.stdout.decode('utf-8')
