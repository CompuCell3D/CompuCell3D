// -*-c++-*-


%module ("threads"=1) PDESolvers

//enables better handling of STL exceptions
%include "exception.i"
// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::map handling
%include "std_vector.i"

%include "stl.i"


%import "../../../CAPython/CoreObjects.i"
//%import "../CoreObjects.i"


%include "typemaps.i"

// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.
//DOCSTRINGS



%include <windows.i>

%{
// CompuCell3D Include Files
//#include <CompuCell3D/Field3D/Point3D.h>
//#include <CompuCell3D/Field3D/Dim3D.h>

#include <CompuCell3D/Field3D/Array3D.h>
#include <CA/modules/PDESolvers/DiffSecrData.h>
#include <CA/modules/PDESolvers/DiffusionSolverFE.h>



// Namespaces
using namespace std;
using namespace CompuCell3D;



%}




//////%include stl.i //to ensure stl functionality 
//////
//////// // // %include "CompuCellExtraIncludes.i"
//////
//////// C++ std::string handling
//////%include "std_string.i"
//////
//////// C++ std::map handling
//////%include "std_map.i"
//////
//////// C++ std::map handling
//////%include "std_set.i"
//////
//////// C++ std::map handling
//////%include "std_vector.i"
//////
//////%include "stl.i"
//////
////////enables better handling of STL exceptions
//////%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

// %include "swig_includes/numpy.i"
// // // %include "pyinterface/swig_includes/numpy.i"

// // // %init %{
    // // // import_array();
// // // %}


//C arrays
//%include "carrays.i"

// ******************************
// Third Party Classes
// ******************************
#define PDESOLVERS_EXPORT

//%include <CompuCell3D/Field3D/Point3D.h>
//%include <CompuCell3D/Field3D/Dim3D.h>


%include <CompuCell3D/Field3D/Array3D.h>

//%template(Array3DContiguousFloat) CompuCell3D::Array3DContiguous<float>;

//////
//////%include <CA/modules/PDESolvers/DiffusableVectorCommon.h>
//////
%template(stdvectorstring) std::vector<std::string>;

//%ignore CompuCell3D::SecretionData::secretionConst;
%include <CA/modules/PDESolvers/DiffSecrData.h>
%include <CA/modules/PDESolvers/DiffusionSolverFE.h>


