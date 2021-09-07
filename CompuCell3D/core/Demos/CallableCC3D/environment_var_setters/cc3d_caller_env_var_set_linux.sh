#!/bin/sh
current_directory=$(pwd)

# necessary to enforce standard convention for numeric values specification on non-English OS
export LC_NUMERIC="C.UTF-8"

# export PREFIX_CC3D=/home/m/411_auto
export PREFIX_CC3D=<path to where cc3d is installed>

# export PYTHON_INSTALL_PATH=/home/m/miniconda3/envs/cc3d_2021/bin
export PYTHON_INSTALL_PATH=<path to where python executable is. Make sure it is same python as used by cc3d>

export PATH=$PYTHON_INSTALL_PATH:$PATH
#export LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PREFIX_CC3D}/lib/:$LD_LIBRARY_PATH

export COMPUCELL3D_PLUGIN_PATH=${PREFIX_CC3D}/lib/site-packages/cc3d/cpp/CompuCell3DPlugins
export COMPUCELL3D_STEPPABLE_PATH=${PREFIX_CC3D}/lib/site-packages/cc3d/cpp/CompuCell3DSteppables

export LD_LIBRARY_PATH=${PREFIX_CC3D}/lib/site-packages/cc3d/cpp/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${COMPUCELL3D_PLUGIN_PATH}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${COMPUCELL3D_STEPPABLE_PATH}:$LD_LIBRARY_PATH

export COMPUCELL3D_MAJOR_VERSION=4
export COMPUCELL3D_MINOR_VERSION=1
export COMPUCELL3D_BUILD_VERSION=0

export PYTHONPATH=${PREFIX_CC3D}/lib/site-packages

