#!/bin/bash

CMAKE_GENERATOR="Unix Makefiles"
declare -a CMAKE_CONFIG_ARGS

CMAKE_CONFIG_ARGS+=(-DCMAKE_BUILD_TYPE:STRING=Release)
CMAKE_CONFIG_ARGS+=(-DCMAKE_PREFIX_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_FIND_ROOT_PATH:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_INSTALL_PREFIX:PATH=${PREFIX})
CMAKE_CONFIG_ARGS+=(-DCMAKE_INSTALL_RPATH:PATH=${PREFIX}/lib)
CMAKE_CONFIG_ARGS+=(-DBUILD_SHARED_LIBS:BOOL=ON)
CMAKE_CONFIG_ARGS+=(-DNO_OPENCL:BOOLEAN=ON)
CMAKE_CONFIG_ARGS+=(-DBUILD_STANDALONE:BOOLEAN=OFF)
CMAKE_CONFIG_ARGS+=(-DPython3_EXECUTABLE:PATH=${PYTHON})
if [[ $(uname) == Darwin ]]; then
    CMAKE_CONFIG_ARGS+=(-DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT})
fi
if [[ $(uname) == Linux ]]; then
    # works with ubuntu builds only on RHEL you need to change those or hope that cmake figures out how to properly detect GL it is not that hard...
    CMAKE_CONFIG_ARGS+=(-DOPENGL_gl_LIBRARY=/usr/lib/x86_64-linux-gnu/libGL.so)
    CMAKE_CONFIG_ARGS+=(-DOPENGL_glx_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLX.so)
fi
 

mkdir build
cd build

cmake -G "${CMAKE_GENERATOR}" \
      "${CMAKE_CONFIG_ARGS[@]}" \
      "${SRC_DIR}/CompuCell3D"
make -j${CPU_COUNT} VERBOSE=1
make install


## Upgrade pip to fix TLS/PEP517 issues
#$PYTHON -m pip install --upgrade pip setuptools wheel
#
## Install pip-only dependencies
#$PYTHON -m pip install --no-deps --prefix=$PREFIX libroadrunner antimony
#
### --- Install pip-only packages ---
##$PYTHON -m pip install --no-deps libroadrunner libantimony