import os
import sys
from os.path import dirname, join, abspath
from pathlib import Path

versionMajor = 4
versionMinor = 2
versionBuild = 5
revisionNumber = "20210612"


def get_sha_label() -> str:
    """
    Fetches git sha tag - relies on the file sha_label.py . This file is NOT part of git repo but instead it is
    written during installation scripts run. Main use case is to know exact tag based on which binaries have been built
    :return: sha tag
    """
    try:
        import cc3d.commit_tag
        try:
            sha_tag = cc3d.commit_tag.sha_label
            return sha_tag
        except AttributeError:
            return revisionNumber

    except ImportError:
        return revisionNumber


def getVersionAsString():
    return str(versionMajor) + "." + str(versionMinor) + "." + str(versionBuild)


def getVersionMajor():
    return versionMajor


def getVersionMinor():
    return versionMinor


def getVersionBuild():
    return versionBuild


def getSVNRevision():
    return revisionNumber


def getSVNRevisionAsString():
    return str(getSVNRevision())


__version__ = getVersionAsString()
__revision__ = revisionNumber


def get_version_info():
    """
    returns CC3D version string
    :return:
    """
    return f"CompuCell3D Version: {__version__} Revision: {__revision__} \n Commit Label: {get_sha_label()}"


def get_formatted_version_info():
    """
    returns formatted CC3D version string
    :return:
    """
    formatted_version_info = f'#################################################\n' \
                             f'# {get_version_info()}\n' \
                             f'#################################################'
    return formatted_version_info


path_postfix = ''
if sys.platform.startswith('win'):
    path_postfix = '\\'
else:
    path_postfix = '/'

cc3d_py_dir = dirname(__file__)
# compucell3d_steppable_path = join(cc3d_py_dir, 'cpp', 'CompuCell3DSteppables')

os.environ['COMPUCELL3D_STEPPABLE_PATH'] = join(cc3d_py_dir, 'cpp', 'CompuCell3DSteppables') + path_postfix
os.environ['COMPUCELL3D_PLUGIN_PATH'] = join(cc3d_py_dir, 'cpp', 'CompuCell3DPlugins') + path_postfix
print(os.environ['COMPUCELL3D_STEPPABLE_PATH'])
print(os.environ['COMPUCELL3D_PLUGIN_PATH'])

if sys.platform.startswith('win'):
    path_env = os.environ['PATH']
    # needed for pyqt modules installed via conda
    python_exe = Path(sys.executable)
    python_exe_dir = python_exe.parent
    pyqt_library_bin_path = python_exe_dir.joinpath('Library', 'bin')
    path_env_list = path_env.split(';')

    # needed for maboss
    mingw_bin_path = python_exe_dir.joinpath('Library', 'mingw-w64', 'bin')

    path_env_list = list(map(lambda pth: abspath(pth), path_env_list))

    cc3d_bin_path = abspath(join(cc3d_py_dir, 'cpp', 'bin'))
    if cc3d_bin_path not in path_env_list:
        path_env_list.insert(0, cc3d_bin_path)

    # todo - this needs to have platform specific behavior
    path_env_list.insert(0, os.environ['COMPUCELL3D_PLUGIN_PATH'])
    path_env_list.insert(0, os.environ['COMPUCELL3D_STEPPABLE_PATH'])
    path_env_list.insert(0, str(pyqt_library_bin_path))
    path_env_list.insert(0, str(mingw_bin_path))

    os.environ['PATH'] = ';'.join(path_env_list)

elif sys.platform.startswith('darwin'):
    try:
        dyld_library_env = os.environ['DYLD_LIBRARY_PATH']
    except KeyError:
        dyld_library_env = ''

    dyld_env_list = dyld_library_env.split(':')

    cc3d_cpp_lib_path = abspath(join(cc3d_py_dir, 'cpp', 'lib'))
    if cc3d_cpp_lib_path not in dyld_env_list:
        dyld_env_list.insert(0, cc3d_cpp_lib_path)

    os.environ['DYLD_LIBRARY_PATH'] = ':'.join(dyld_env_list)

elif sys.platform.startswith('linux'):

    try:
        ld_library_env = os.environ['LD_LIBRARY_PATH']
    except KeyError:
        ld_library_env = ''

    ld_env_list = ld_library_env.split(':')
    cc3d_cpp_lib_path = abspath(join(cc3d_py_dir, 'cpp', 'lib'))
    if cc3d_cpp_lib_path not in ld_env_list:
        ld_env_list.insert(0, cc3d_cpp_lib_path)

    os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_env_list)
