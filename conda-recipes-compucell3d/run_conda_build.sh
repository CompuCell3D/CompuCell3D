export version="4.4.0"
export build_number=0
export numpy_version="1.21"
export cc3d_network_solvers_version="0.3.0"
export cmake_version="3.21" 
export vtk_version="9.2" 
export swig_version="4" 
export boost_version="1.78" 
export tbb_devel_version="2021" 

conda build -c local -c conda-forge -c compucell3d . --python=3.7
#conda render .