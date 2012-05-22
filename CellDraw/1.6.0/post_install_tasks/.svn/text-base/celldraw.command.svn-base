#!/bin/bash

# echo " "
# echo " dollar-zero AKA the first argument to this .command script is: "
# echo $0
# echo " "
cd "${0%/*}"

export CELLDRAW_MAJOR_VERSION=1
export CELLDRAW_MINOR_VERSION=5
export CELLDRAW_BUILD_VERSION=1
export COMPUCELL3D_MAJOR_VERSION=3
export COMPUCELL3D_MINOR_VERSION=6
export COMPUCELL3D_BUILD_VERSION=1
echo " "
echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo "  CellDraw version $CELLDRAW_MAJOR_VERSION.$CELLDRAW_MINOR_VERSION.$CELLDRAW_BUILD_VERSION"
echo "  CompuCell3D version $COMPUCELL3D_MAJOR_VERSION.$COMPUCELL3D_MINOR_VERSION.$COMPUCELL3D_BUILD_VERSION"
echo "---- ---- ---- ---- ---- ---- ---- ---- "


# the "PREFIX_CC3D" shell variable is used by CellDraw code, its name can NOT be modified:
export PREFIX_CELLDRAW=$(pwd)
# the "PREFIX_CC3D" shell variable is used by CompuCell3D code, its name can NOT be modified:
export PREFIX_CC3D=${PREFIX_CELLDRAW}
# current_directory=$(pwd)
cd $PREFIX_CELLDRAW
echo "====> CellDraw working directory: $PREFIX_CELLDRAW"
echo " ====> CompuCell3D working directory: $PREFIX_CC3D"
pwd


echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " setting shell variables used by CompuCell3D: "
# the "PYTHON_MODULE_PATH" shell variable is used by CompuCell3D code, its name can NOT be modified:
export PYTHON_MODULE_PATH=${PREFIX_CC3D}/pythonSetupScripts
echo " ====> PYTHON_MODULE_PATH: $PYTHON_MODULE_PATH"
# the "COMPUCELL3D_PLUGIN_PATH" shell variable is used by CompuCell3D code, its name can NOT be modified:
export COMPUCELL3D_PLUGIN_PATH=${PREFIX_CC3D}/lib/CompuCell3DPlugins
echo " ====> COMPUCELL3D_PLUGIN_PATH: $COMPUCELL3D_PLUGIN_PATH"
# the "COMPUCELL3D_STEPPABLE_PATH" shell variable is used by CompuCell3D code, its name can NOT be modified:
export COMPUCELL3D_STEPPABLE_PATH=${PREFIX_CC3D}/lib/CompuCell3DSteppables
echo " ====> COMPUCELL3D_STEPPABLE_PATH: $COMPUCELL3D_STEPPABLE_PATH"
# the "SWIG_LIB_INSTALL_DIR" shell variable is used by CompuCell3D code, its name can NOT be modified:
export SWIG_LIB_INSTALL_DIR=${PREFIX_CC3D}/lib/python
echo " ====> SWIG_LIB_INSTALL_DIR: $SWIG_LIB_INSTALL_DIR"
# the "SOSLIB_PATH" shell variable is used within CompuCell3D code, its name can NOT be modified,
#   but it doesn't need to point to any specific path, so we just set it to ${PREFIX_CC3D}:
# export SOSLIB_PATH=${PREFIX_CC3D}/examplesSoslib
export SOSLIB_PATH=${PREFIX_CC3D}
echo " ====> SOSLIB_PATH: $SOSLIB_PATH"


echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " setting the DYLD_LIBRARY_PATH shell variable as used by CompuCell3D: "
# export DYLD_LIBRARY_PATH=:$DYLD_LIBRARY_PATH
# avoid any previously user-defined DYLD_LIBRARY_PATH values:
# export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib/:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib
# echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib/python:$DYLD_LIBRARY_PATH
# echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${COMPUCELL3D_PLUGIN_PATH}:$DYLD_LIBRARY_PATH
# echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${COMPUCELL3D_STEPPABLE_PATH}:$DYLD_LIBRARY_PATH
# echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/Deps:$DYLD_LIBRARY_PATH
# echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/Deps/QtDeps:$DYLD_LIBRARY_PATH
# echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/player/vtk:$DYLD_LIBRARY_PATH
# echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/player/VTKLibs:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"


echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " setting the PYTHONPATH shell variable as used by CellDraw: "
# export PYTHONPATH=${PYTHONLIB_26_SYSTEM}/lib/python2.5/:${PYTHONLIB_26_SYSTEM}/python2.5/lib-dynload/:${PYTHONLIB_26_SYSTEM}/bin/
export PYTHONPATH=${PREFIX_CELLDRAW}/player:$PYTHONPATH
echo "====> PYTHONPATH directory: $PYTHONPATH"


echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " setting the PYTHONLIB_26_SYSTEM shell variable as used by CompuCell3D: "
export PYTHONLIB_26_SYSTEM=/System/Library/Frameworks/Python.framework/Versions/2.6
echo " ====> PYTHONLIB_26_SYSTEM: $PYTHONLIB_26_SYSTEM"


echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " setting the PATH shell variable as used by CompuCell3D: "
# avoid any previously user-defined DYLD_LIBRARY_PATH values:
# export PATH=${PREFIX_CC3D}/LIBRARYDEPS/sipDeps:${PYTHONLIB_26_SYSTEM}/bin:${PYTHONLIB_26_SYSTEM}:${PREFIX_CC3D}/LIBRARYDEPS:${PREFIX_CC3D}/LIBRARYDEPS/LIBRARY-PYTHON-2.5/Extras/lib/python/wx/lib:$PATH
# export PATH=${PYTHONLIB_26_SYSTEM}/bin:${PYTHONLIB_26_SYSTEM}
export PATH=${PYTHONLIB_26_SYSTEM}/bin
echo " ====> PATH: $PATH"
# echo "---- ---- ---- ---- ---- ---- ---- ---- "
# echo " env is here:"
# echo "---- ---- ---- ---- ---- ---- ---- ---- "
# /usr/bin/env | /usr/bin/sort






echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo 'Hello World. Python --version says:'
echo "---- ---- ---- ---- ---- ---- ---- ---- "
python2.6 --version
echo "---- ---- ---- ---- ---- ---- ---- ---- "
echo " "
echo "====> CellDraw $CELLDRAW_MAJOR_VERSION.$CELLDRAW_MINOR_VERSION.$CELLDRAW_BUILD_VERSION now starting from Python."
echo " "
python2.6  ${PREFIX_CELLDRAW}/CellDraw/cellDrawMain.pyw

# cd ${current_directory}
cd ${PREFIX_CC3D}
