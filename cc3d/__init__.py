import os
import sys
from os.path import dirname, join, abspath

path_postfix = ''
if sys.platform.startswith('win'):
    path_postfix = '\\'

cc3d_py_dir = dirname(__file__)
# compucell3d_steppable_path = join(cc3d_py_dir, 'cpp', 'CompuCell3DSteppables')

os.environ['COMPUCELL3D_STEPPABLE_PATH'] = join(cc3d_py_dir, 'cpp', 'CompuCell3DSteppables') + path_postfix
os.environ['COMPUCELL3D_PLUGIN_PATH'] = join(cc3d_py_dir, 'cpp', 'CompuCell3DPlugins') + path_postfix

path_env = os.environ['PATH']

path_env_list = path_env.split(';')

path_env_list = list(map(lambda pth: abspath(pth), path_env_list))

cc3d_bin_path = abspath(join(cc3d_py_dir, 'cpp', 'bin'))
if cc3d_bin_path not in path_env_list:
    path_env_list.insert(0, cc3d_bin_path)

path_env_list.insert(0,os.environ['COMPUCELL3D_PLUGIN_PATH'])


os.environ['PATH'] = ';'.join(path_env_list)

print('ENVIRONMENT VARS=', os.environ)

