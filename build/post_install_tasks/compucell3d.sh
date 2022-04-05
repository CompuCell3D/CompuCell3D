#!/bin/sh
current_directory=$(pwd)


# necessary to enforce standard convention for numeric values specification on non-English OS
export LC_NUMERIC="C.UTF-8"


# export PREFIX_CC3D=/N/u/mehtasau/Carbonate/cc3d_build_wdocs
export PREFIX_CC3D="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHON_EXEC=/N/u/mehtasau/Carbonate/miniconda3/envs/cc3denv/bin/python3.7


cd $PREFIX_CC3D
#export LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PREFIX_CC3D}/lib/:$LD_LIBRARY_PATH

export COMPUCELL3D_PLUGIN_PATH=${PREFIX_CC3D}/lib/site-packages/cc3d/cpp/CompuCell3DPlugins
export COMPUCELL3D_STEPPABLE_PATH=${PREFIX_CC3D}/lib/site-packages/cc3d/cpp/CompuCell3DSteppables

export LD_LIBRARY_PATH=${PREFIX_CC3D}/lib/site-packages/cc3d/cpp/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${COMPUCELL3D_PLUGIN_PATH}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${COMPUCELL3D_STEPPABLE_PATH}:$LD_LIBRARY_PATH

export PYTHONPATH=${PREFIX_CC3D}/lib/site-packages

export exit_code=0
${PYTHON_EXEC} ${PREFIX_CC3D}/lib/site-packages/cc3d/player5/compucell3d.pyw $* --currentDir=${current_directory}
exit_code=$?

cd ${current_directory}
exit ${exit_code}
