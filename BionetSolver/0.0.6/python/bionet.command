#!/bin/bash

# echo " "
# echo " dollar-zero AKA the first argument to this .command script is: "
# echo $0
# echo " "
cd "${0%/*}"

export PREFIX_CC3D=$(pwd)

current_directory=$(pwd)



cd $PREFIX_CC3D

echo " "
echo " ====> CompuCell3D working directory: $PREFIX_CC3D"
pwd
echo " "

export PYTHONLIB25=/System/Library/Frameworks/Python.framework/Versions/2.6/
#export DYLD_LIBRARY_PATH=:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib/:$DYLD_LIBRARY_PATH
echo " ====> PYTHONLIB25: $PYTHONLIB25"

export COMPUCELL3D_PLUGIN_PATH=${PREFIX_CC3D}/lib/CompuCell3DPlugins
echo " ====> COMPUCELL3D_PLUGIN_PATH: $COMPUCELL3D_PLUGIN_PATH"
export COMPUCELL3D_STEPPABLE_PATH=${PREFIX_CC3D}/lib/CompuCell3DSteppables
echo " ====> COMPUCELL3D_STEPPABLE_PATH: $COMPUCELL3D_STEPPABLE_PATH"
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib/python:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${COMPUCELL3D_PLUGIN_PATH}:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${COMPUCELL3D_STEPPABLE_PATH}:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export SWIG_LIB_INSTALL_DIR=${PREFIX_CC3D}/lib/python
echo " ====> SWIG_LIB_INSTALL_DIR: $SWIG_LIB_INSTALL_DIR"
export PYTHON_MODULE_PATH=${PREFIX_CC3D}/pythonSetupScripts
echo " ====> PYTHON_MODULE_PATH: $PYTHON_MODULE_PATH"

export PATH=${PREFIX_CC3D}/LIBRARYDEPS/sipDeps:${PYTHONLIB25}/bin:${PYTHONLIB25}:${PREFIX_CC3D}/LIBRARYDEPS:${PREFIX_CC3D}/LIBRARYDEPS/LIBRARY-PYTHON-2.5/Extras/lib/python/wx/lib/:$PATH
echo " ====> PATH: $PATH"


#export PYTHONPATH=${PYTHONLIB25}/lib/python2.5/:${PYTHONLIB25}/python2.5/lib-dynload/:${PYTHONLIB25}/bin/

export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/Deps/:${PREFIX_CC3D}/Deps/QtDeps:${PREFIX_CC3D}/player/vtk/:${PREFIX_CC3D}/player/VTKLibs:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
echo " "
echo 'Hello World'
echo " "
python --version
echo " "
echo " ====> PATH: $PATH"
echo " "


export COMPUCELL3D_MAJOR_VERSION=3
export COMPUCELL3D_MINOR_VERSION=5
export COMPUCELL3D_BUILD_VERSION=0

export SOSLIB_PATH=${PREFIX_CC3D}/examplesSoslib
echo " ====> SOSLIB_PATH: $SOSLIB_PATH"
echo " "

echo "CompuCell3D - version $COMPUCELL3D_MAJOR_VERSION.$COMPUCELL3D_MINOR_VERSION.$COMPUCELL3D_BUILD_VERSION"

echo " "
echo " Now starting CompuCell3D in GUI mode:"
echo " "
python ${PREFIX_CC3D}/bionet.py $* --currentDir=${current_directory}

cd ${current_directory}

