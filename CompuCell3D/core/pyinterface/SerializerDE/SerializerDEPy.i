
// Module Name
%module("threads"=1) SerializerDEPy




// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{


#include <SerializerDE.h>
//#include <CompuCell3D/Field3D/Point3D.h>
//#include <CompuCell3D/Field3D/Dim3D.h>
#include <vtkIntArray.h>
    
#define SERIALIZERDE_EXPORT

   

// System Libraries
#include <iostream>
#include <stdlib.h>
// #include <Coordinates3D.h>



   
// Namespaces
using namespace std;
using namespace CompuCell3D;

%}

#define SERIALIZERDE_EXPORT




// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

// Pointer handling
%include "cpointer.i"


/* %include <GraphicsDataFields.h> */
/* %include <mainCC3D.h> */

/* %include <mainCC3DWrapper.h> */

//instantiate vector<int>
%template(vectorint) std::vector<int>;
%template(vectorstring) std::vector<std::string>;



%include <SerializerDE.h>
//%include <CompuCell3D/Field3D/Point3D.h>
//%include <CompuCell3D/Field3D/Dim3D.h>

