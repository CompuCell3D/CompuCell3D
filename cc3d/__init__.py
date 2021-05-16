import os
import sys
from os.path import dirname, join, abspath
from . import config
from .config import versionMajor, versionMinor, versionBuild, revisionNumber


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

os.environ['COMPUCELL3D_STEPPABLE_PATH'] = cc3d_steppable_path + os.sep
os.environ['COMPUCELL3D_PLUGIN_PATH'] = cc3d_plugin_path + os.sep
print(os.environ['COMPUCELL3D_STEPPABLE_PATH'])
print(os.environ['COMPUCELL3D_PLUGIN_PATH'])

if sys.platform.startswith('win'):
    path_env = os.environ['PATH']

    path_env_list = path_env.split(';')

    path_env_list = list(map(lambda pth: abspath(pth), path_env_list))

    if cc3d_lib_shared not in path_env_list:
        path_env_list.insert(0, cc3d_lib_shared)
    if cc3d_cpp_bin_path not in path_env_list:
        path_env_list.insert(0, cc3d_cpp_bin_path)

    # todo - this needs to have platform specific behavior
    path_env_list.insert(0, os.environ['COMPUCELL3D_PLUGIN_PATH'])
    path_env_list.insert(0, os.environ['COMPUCELL3D_STEPPABLE_PATH'])

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
