#!/bin/sh

# echo " "
# echo " dollar-zero AKA the first argument to this .command script is: "
# echo $0
# echo " "
cd "${0%/*}"

export PREFIX_CC3D=$(pwd)

current_directory=$(pwd)

cd $PREFIX_CC3D

echo "====> twedit++ working directory: $PREFIX_CC3D"

export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib/python:$DYLD_LIBRARY_PATH

export SWIG_LIB_INSTALL_DIR=${PREFIX_CC3D}/lib/python

export PYTHON_MODULE_PATH=${PREFIX_CC3D}/pythonSetupScripts


export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/Deps:${PREFIX_CC3D}/Deps/QtDeps:${PREFIX_CC3D}/player/vtk:${PREFIX_CC3D}/player/VTKLibs:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"

export PYTHONPATH=${PREFIX_CC3D}/player:$PYTHONPATH
echo "====> PYTHONPATH directory: $PYTHONPATH"




echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " setting the PYTHONLIB_26_SYSTEM shell variable as used by Twedit++ : "
export PYTHONLIB_26_SYSTEM=/System/Library/Frameworks/Python.framework/Versions/2.6
echo " ====> PYTHONLIB_26_SYSTEM: $PYTHONLIB_26_SYSTEM"


echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " setting the PATH shell variable as used by CompuCell3D: "
# avoid any previously user-defined DYLD_LIBRARY_PATH values:
export PATH=${PYTHONLIB_26_SYSTEM}/bin
echo " ====> PATH: $PATH"

# echo "---- ---- ---- ---- ---- ---- ---- ---- "
# echo " env is here:"
# echo "---- ---- ---- ---- ---- ---- ---- ---- "
# /usr/bin/env | /usr/bin/sort


export TWEDIT_MAJOR_VERSION=0
export TWEDIT_MINOR_VERSION=9
export TWEDIT_BUILD_VERSION=0


echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo 'Hello World. Python --version says:'
echo "---- ---- ---- ---- ---- ---- ---- ---- "
python2.6 --version
echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " "
echo "====> twedit++ $TWEDIT_MAJOR_VERSION.$TWEDIT_MINOR_VERSION.$TWEDIT_BUILD_VERSION starting from python"
echo " "
echo " Now starting Twedit++:"
echo " "
python2.6 ${PREFIX_CC3D}/Twedit++/twedit_plus_plus_cc3d.py $*


cd ${current_directory}
