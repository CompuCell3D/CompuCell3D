#!/bin/bash

# echo " "
# echo " dollar-zero AKA the first argument to this .command script is: "
# echo $0
# echo " "
cd "${0%/*}"

export PREFIX_CELLDRAW=$(pwd)

current_directory=$(pwd)


cd $PREFIX_CELLDRAW

echo "====> CellDraw working directory: $PREFIX_CELLDRAW"

export PYTHONLIB26=/System/Library/Frameworks/Python.framework/Versions/2.6
export DYLD_LIBRARY_PATH=${PREFIX_CELLDRAW}/../Frameworks
echo "====> PYTHONLIB26: $PYTHONLIB26"
echo "====> DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"

export PYTHONPATH=${PREFIX_CELLDRAW}/../Resources/site-packages:$PYTHONPATH
echo "====> PYTHONPATH directory: $PYTHONPATH"

export PATH=${PYTHONLIB26}/bin:${PYTHONLIB26s}
echo "====> PATH: $PATH"

export PREFIX_CC3D=/Applications/CC3D_3.6.0_MacOSX106
echo "====> PREFIX_CC3D: $PREFIX_CC3D"


export CELLDRAW_MAJOR_VERSION=1
export CELLDRAW_MINOR_VERSION=6
export CELLDRAW_BUILD_VERSION=0

/usr/bin/pythonw2.6 --version
echo "====> CellDraw $CELLDRAW_MAJOR_VERSION.$CELLDRAW_MINOR_VERSION.$CELLDRAW_BUILD_VERSION now starting from Python."

# on Mac OS X 10.6.x we explicitly call Python 2.6:
/usr/bin/pythonw2.6 ${PREFIX_CELLDRAW}/cellDrawMain.pyw

cd ${current_directory}
