#!/bin/sh

#export PREFIX_CC3D=/Users/m/Demo/CC3D_4.1.1
export PREFIX_CC3D=<directory where cc3d is installed>

# if not using bundled python set PYTHON_INSTALL_PATH to point to the folder where Python used with CC3D is installed
export PYTHON_INSTALL_PATH=${PREFIX_CC3D}/python37/bin
export PATH=$PYTHON_INSTALL_PATH:$PATH

export PYTHONPATH=${PREFIX_CC3D}/lib/site-packages


