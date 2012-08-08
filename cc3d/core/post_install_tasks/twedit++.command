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

# export PYTHONLIB25=/System/Library/Frameworks/Python.framework/Versions/2.6/
# echo " ====> PYTHONLIB25: $PYTHONLIB25"

# export COMPUCELL3D_PLUGIN_PATH=${PREFIX_CC3D}/lib/CompuCell3DPlugins
# export COMPUCELL3D_STEPPABLE_PATH=${PREFIX_CC3D}/lib/CompuCell3DSteppables

export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib/:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/lib/python:$DYLD_LIBRARY_PATH
# export DYLD_LIBRARY_PATH=${COMPUCELL3D_PLUGIN_PATH}:$DYLD_LIBRARY_PATH
# export DYLD_LIBRARY_PATH=${COMPUCELL3D_STEPPABLE_PATH}:$DYLD_LIBRARY_PATH

export SWIG_LIB_INSTALL_DIR=${PREFIX_CC3D}/lib/python

export PYTHON_MODULE_PATH=${PREFIX_CC3D}/pythonSetupScripts

# export PATH=${PREFIX_CC3D}/LIBRARYDEPS/sipDeps:${PYTHONLIB25}/bin:${PYTHONLIB25}:${PREFIX_CC3D}/LIBRARYDEPS:${PREFIX_CC3D}/LIBRARYDEPS/LIBRARY-PYTHON-2.5/Extras/lib/python/wx/lib/:$PATH
# echo " ====> PATH: $PATH"

export DYLD_LIBRARY_PATH=${PREFIX_CC3D}/Deps/:${PREFIX_CC3D}/Deps/QtDeps:${PREFIX_CC3D}/player/vtk/:${PREFIX_CC3D}/player/VTKLibs:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"

export PYTHONPATH=${PREFIX_CC3D}/player:$PYTHONPATH
echo "====> PYTHONPATH directory: $PYTHONPATH"


export TWEDIT_MAJOR_VERSION=0
export TWEDIT_MINOR_VERSION=9
export TWEDIT_BUILD_VERSION=0

python --version
echo "====> twedit++ $TWEDIT_MAJOR_VERSION.$TWEDIT_MINOR_VERSION.$TWEDIT_BUILD_VERSION starting from python"

python ${PREFIX_CC3D}/Twedit++/twedit_plus_plus_cc3d.py $*

cd ${current_directory}
