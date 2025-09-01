try:
    # needed to avoid import errors on some windows systems
    import vtk
except ImportError:
    print('vtk not found. Ignoring for now')

import os
import sys
from os.path import dirname, join, abspath
from pathlib import Path

__version__ = "4.7.0"
__revision__ = "5"
__githash__ = "f401d6d"


from cc3d import config

def get_version_revision_str():
    return f"{__version__}.{__revision__}"


def get_version_str():
    return f"{__version__}"


def get_version_info():
    """
    returns CC3D version string
    :return:
    """
    return f"CompuCell3D Version: {__version__} Revision: {__revision__} \n Commit Label: {__githash__}"


def get_formatted_version_info():
    """
    returns formatted CC3D version string
    :return:
    """
    formatted_version_info = f'#################################################\n' \
                             f'# {get_version_info()}\n' \
                             f'#################################################'
    return formatted_version_info


cc3d_py_dir = dirname(__file__)
cc3d_install_prefix = abspath(join(cc3d_py_dir, config.cc3d_install_prefix_rel))
cc3d_cpp_path = abspath(join(cc3d_py_dir, config.cc3d_cpp_path_rel))
cc3d_steppable_path = abspath(join(cc3d_cpp_path, 'CompuCell3DSteppables'))
cc3d_plugin_path = abspath(join(cc3d_cpp_path, 'CompuCell3DPlugins'))
cc3d_cpp_bin_path = abspath(join(cc3d_cpp_path, 'bin'))
cc3d_cpp_lib_path = abspath(join(cc3d_cpp_path, 'lib'))
cc3d_scripts_path = abspath(join(cc3d_py_dir, config.cc3d_scripts_path_rel))
cc3d_lib_shared = abspath(join(cc3d_install_prefix, 'bin'))
cc3d_lib_static = abspath(join(cc3d_install_prefix, 'lib'))

cc3d_cpp_bin_path_pathlib = Path(cc3d_cpp_bin_path)

os.environ['COMPUCELL3D_STEPPABLE_PATH'] = cc3d_steppable_path + os.sep
os.environ['COMPUCELL3D_PLUGIN_PATH'] = cc3d_plugin_path + os.sep

if 'CC3D_OPENCL_SOLVERS_DIR' not in os.environ:
    os.environ['CC3D_OPENCL_SOLVERS_DIR'] = cc3d_steppable_path + os.sep + "OpenCL" + os.sep
else:
    print("Detected CC3D_OPENCL_SOLVERS_DIR:", os.environ['CC3D_OPENCL_SOLVERS_DIR'])



if sys.platform.startswith('win'):
    path_env = os.environ['PATH']

    path_env_list = path_env.split(';')

    # needed for maboss
    python_exe = Path(sys.executable)
    python_exe_dir = python_exe.parent
    mingw_bin_path = python_exe_dir.joinpath('Library', 'mingw-w64', 'bin')

    path_env_list = list(map(lambda pth: abspath(pth), path_env_list))

    if sys.version_info >= (3, 8):

        if cc3d_lib_shared not in path_env_list:
            if Path(cc3d_lib_shared).exists() and Path(cc3d_lib_shared).is_dir():
                os.add_dll_directory(cc3d_lib_shared)

        os.add_dll_directory(os.environ['COMPUCELL3D_PLUGIN_PATH'])
        os.add_dll_directory(os.environ['COMPUCELL3D_STEPPABLE_PATH'])
        os.add_dll_directory(str(mingw_bin_path))


    else:

        if cc3d_lib_shared not in path_env_list:
            path_env_list.insert(0, cc3d_lib_shared)

        path_env_list.insert(0, os.environ['COMPUCELL3D_PLUGIN_PATH'])
        path_env_list.insert(0, os.environ['COMPUCELL3D_STEPPABLE_PATH'])
        path_env_list.insert(0, str(mingw_bin_path))

        os.environ['PATH'] = ';'.join(path_env_list)

elif sys.platform.startswith('darwin'):
    try:
        dyld_library_env = os.environ['DYLD_LIBRARY_PATH']
    except KeyError:
        dyld_library_env = ''

    dyld_env_list = dyld_library_env.split(':')

    if cc3d_cpp_lib_path not in dyld_env_list:
        dyld_env_list.insert(0, cc3d_cpp_lib_path)

    os.environ['DYLD_LIBRARY_PATH'] = ':'.join(dyld_env_list)

elif sys.platform.startswith('linux'):

    try:
        ld_library_env = os.environ['LD_LIBRARY_PATH']
    except KeyError:
        ld_library_env = ''

    ld_env_list = ld_library_env.split(':')
    if cc3d_cpp_lib_path not in ld_env_list:
        ld_env_list.insert(0, cc3d_cpp_lib_path)
    if cc3d_lib_static not in ld_env_list:
        ld_env_list.insert(0, cc3d_lib_static)

    os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_env_list)
