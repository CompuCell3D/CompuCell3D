#!/bin/bash

# echo " "
# echo " dollar-zero AKA the first argument to this .command script is: "
# echo $0
# echo " "
cd "${0%/*}"

export PREFIX_CELLDRAW=$(pwd)

current_directory=$(pwd)



export PREFIX_CC3D=${PREFIX_CELLDRAW}

echo " "
echo "====> CellDraw working directory: $PREFIX_CELLDRAW"
echo "====> CompuCell3D working directory: $PREFIX_CC3D"
echo "       pwd : "
pwd
echo " "

export PYTHONLIB26=/System/Library/Frameworks/Python.framework/Versions/2.6/
#export DYLD_LIBRARY_PATH=:$DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${PREFIX_CELLDRAW}/lib/:$DYLD_LIBRARY_PATH
echo " ====> PYTHONLIB26: $PYTHONLIB26"

export COMPUCELL3D_PLUGIN_PATH=${PREFIX_CELLDRAW}/lib/CompuCell3DPlugins
echo " ====> COMPUCELL3D_PLUGIN_PATH: $COMPUCELL3D_PLUGIN_PATH"
export COMPUCELL3D_STEPPABLE_PATH=${PREFIX_CELLDRAW}/lib/CompuCell3DSteppables
echo " ====> COMPUCELL3D_STEPPABLE_PATH: $COMPUCELL3D_STEPPABLE_PATH"
export DYLD_LIBRARY_PATH=${PREFIX_CELLDRAW}/lib/python:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${COMPUCELL3D_PLUGIN_PATH}:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH=${COMPUCELL3D_STEPPABLE_PATH}:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
export SWIG_LIB_INSTALL_DIR=${PREFIX_CELLDRAW}/lib/python
echo " ====> SWIG_LIB_INSTALL_DIR: $SWIG_LIB_INSTALL_DIR"
export PYTHON_MODULE_PATH=${PREFIX_CELLDRAW}/pythonSetupScripts
echo " ====> PYTHON_MODULE_PATH: $PYTHON_MODULE_PATH"

export PATH=${PREFIX_CELLDRAW}/LIBRARYDEPS/sipDeps:${PYTHONLIB26}/bin:${PYTHONLIB26}:${PREFIX_CELLDRAW}/LIBRARYDEPS:${PREFIX_CELLDRAW}/LIBRARYDEPS/LIBRARY-PYTHON-2.6/Extras/lib/python/wx/lib/:$PATH
echo " ====> PATH: $PATH"


#export PYTHONPATH=${PYTHONLIB26}/lib/python2.6/:${PYTHONLIB26}/python2.6/lib-dynload/:${PYTHONLIB26}/bin/

export PYTHONPATH=${PREFIX_CELLDRAW}/player:$PYTHONPATH
echo "====> PYTHONPATH directory: $PYTHONPATH"


export DYLD_LIBRARY_PATH=${PREFIX_CELLDRAW}/Deps/:${PREFIX_CELLDRAW}/Deps/QtDeps:${PREFIX_CELLDRAW}/player/vtk/:${PREFIX_CELLDRAW}/player/VTKLibs:$DYLD_LIBRARY_PATH
echo " ====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
echo " "
echo 'Hello World'
echo " "
python --version
echo " "
echo " ====> PATH: $PATH"
echo " "

export SOSLIB_PATH=${PREFIX_CELLDRAW}/examplesSoslib
echo " ====> SOSLIB_PATH: $SOSLIB_PATH"
echo " "

export CELLDRAW_MAJOR_VERSION=1
export CELLDRAW_MINOR_VERSION=5
export CELLDRAW_BUILD_VERSION=0

echo " "
echo "====> CellDraw $CELLDRAW_MAJOR_VERSION.$CELLDRAW_MINOR_VERSION.$CELLDRAW_BUILD_VERSION now starting from Python."
echo " "
python ${PREFIX_CELLDRAW}/CellDraw/cellDrawMain.pyw

cd ${current_directory}

