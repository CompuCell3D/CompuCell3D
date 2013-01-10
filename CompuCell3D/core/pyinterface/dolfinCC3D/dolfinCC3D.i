
// Module Name
%module("threads"=1) dolfinCC3D

//%module Example
// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{
#include <dolfinCC3D.h>
#include <CleaverDolfinUtil.h>
//#include <CompuCell3D/Field3D/Dim3D.h>


#define DOLFINCC3D_EXPORT
// Namespaces
using namespace std;
using namespace CompuCell3D;


%}

#define DOLFINCC3D_EXPORT


// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

//%include <CompuCell3D/Field3D/Dim3D.h>

%template(vectorint) std::vector<unsigned char>;

%include <dolfinCC3D.h>
%include <CleaverDolfinUtil.h>



