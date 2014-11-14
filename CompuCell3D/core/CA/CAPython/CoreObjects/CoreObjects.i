

%module ("threads"=1) CoreObjects

%include "typemaps.i"

// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.
//DOCSTRINGS



%include <windows.i>
//enables better handling of STL exceptions
// %include "exception.i"
// // C++ std::vector handling
// %include "std_vector.i"

%{
// CompuCell3D Include Files
#include <CompuCell3D/Field3D/Point3D.h> //notice order matters here because Dim3D inherits from Point3d
#include <CompuCell3D/Field3D/Dim3D.h>

#include <CompuCell3D/Field3D/Field3D.h>

//necessary to get registration of change watcher working in Python
#include <CompuCell3D/Field3D/Field3DChangeWatcher.h>
#include <CA/CACellStackFieldChangeWatcher.h>


// Namespaces
using namespace std;
using namespace CompuCell3D;



%}




// // // %include stl.i //to ensure stl functionality 

// // // // // // %include "CompuCellExtraIncludes.i"

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_set.i"

// C++ std::vector handling
%include "std_vector.i"

%include "stl.i"

//enables better handling of STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

// %include "swig_includes/numpy.i"
// %include "pyinterface/swig_includes/numpy.i"

// %init %{
    // import_array();
// %}


//C arrays
//%include "carrays.i"

// ******************************
// Third Party Classes
// ******************************
// // // #define PDESOLVERS_EXPORT


//%template(Array3DContiguousFloat) CompuCell3D::Array3DContiguous<float>;

%include <CompuCell3D/Field3D/Point3D.h>//notice order matters here because Dim3D inherits from Point3d
%include <CompuCell3D/Field3D/Dim3D.h>

%include <CompuCell3D/Field3D/Field3D.h>

// floatfield
%ignore CompuCell3D::Field3D<float>::typeStr;
%template(floatfield) CompuCell3D::Field3D<float>;

%extend CompuCell3D::Point3D{
 std::string __str__(){
   std::ostringstream s;
   s<<(*self);
   return s.str();
 }
};

%extend CompuCell3D::Dim3D{
 std::string __str__(){
   std::ostringstream s;
   s<<(*self);
   return s.str();
 }
};

%template(vectorfloat) std::vector<float>;
%template(vectorunsignedchar) std::vector<unsigned char>;
%template(vectorstring) std::vector<std::string>;
%template(vectorint) std::vector<int>;

//%template(vectorCACellPtr) std::vector<CACell*>;
//%template(mapLongCACellPtr)std::map<long,CACell *> ;

%include <CompuCell3D/Field3D/Field3DChangeWatcher.h>
%template (Field3DChangeWatcherTemplate) CompuCell3D::Field3DChangeWatcher<CACell*>;

%include <CA/CACellStackFieldChangeWatcher.h>
