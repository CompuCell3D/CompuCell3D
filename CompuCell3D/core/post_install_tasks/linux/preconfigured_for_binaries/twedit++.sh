#!/bin/sh

# necessary to enforce standard convention for numeric values specification on non-English OS
export LC_NUMERIC="C.UTF-8"


export PREFIX_CC3D="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHON_EXEC=${PREFIX_CC3D}/Python37/bin/python

# FONTCONFIG env vars ensure that all the qt fonts are loaded properly
export FONTCONFIG_FILE=${PREFIX_CC3D}/Python37/etc/fonts/fonts.conf
export FONTCONFIG_PATH=${PREFIX_CC3D}/Python37/etc/fonts/

export LD_LIBRARY_PATH=${PREFIX_CC3D}/lib/site-packages/cc3d/cpp/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${COMPUCELL3D_PLUGIN_PATH}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${COMPUCELL3D_STEPPABLE_PATH}:$LD_LIBRARY_PATH

export PYTHONPATH=${PREFIX_CC3D}/lib/site-packages

${PYTHON_EXEC} ${PREFIX_CC3D}/lib/site-packages/cc3d/twedit5/twedit_plus_plus_cc3d.py $*
