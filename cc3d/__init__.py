import os
import sys
from os.path import dirname, join, abspath

versionMajor = 4
versionMinor = 2
versionBuild = 0
revisionNumber = "20200418"


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

    path_env_list = path_env.split(';')

    path_env_list = list(map(lambda pth: abspath(pth), path_env_list))

    cc3d_bin_path = abspath(join(cc3d_py_dir, 'cpp', 'bin'))
    if cc3d_bin_path not in path_env_list:
        path_env_list.insert(0, cc3d_bin_path)

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
    #
    # dyld_env_list = list(map(lambda pth: abspath(pth), dyld_env_list))
    #
    # cc3d_cpp_path = abspath(join(cc3d_py_dir, 'cpp'))
    # if cc3d_cpp_path not in dyld_env_list:
    #     dyld_env_list.insert(0, cc3d_cpp_path)
    #
    cc3d_cpp_lib_path = abspath(join(cc3d_py_dir, 'cpp', 'lib'))
    if cc3d_cpp_lib_path not in dyld_env_list:
        dyld_env_list.insert(0, cc3d_cpp_lib_path)

    # dyld_env_list.insert(0,os.environ['COMPUCELL3D_PLUGIN_PATH'])

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
