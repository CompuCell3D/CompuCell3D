set version="4.4.0"
set build_number=0
set numpy_version="1.21"
set cc3d_network_solvers_version="0.3.0"
set cmake_version="3.21" 
set vtk_version="9.2" 
set swig_version="4" 
set boost_version="1.78" 
set tbb_devel_version="2021" 

conda build -c conda-forge -c compucell3d . --python=3.7
#conda render .