
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

#include <boost/shared_ptr.hpp>

#include <dolfinCC3D.h>
#include <CleaverDolfinUtil.h>
//There is no need to include dolfin header files here 
//#include <CompuCell3D/Field3D/Dim3D.h>
// #include <dolfin/mesh/Mesh.h>
// #include <dolfin/mesh/MeshFunction.h>
// #include <dolfin/mesh/SubDomain.h>
#include <CustomSubDomains.h>


#define DOLFINCC3D_EXPORT
// Namespaces
using namespace std;
using namespace CompuCell3D;
using namespace dolfin; // helps SWIG figure out names from dolfin namespace otherwise it would be necesaary to use fully qualified names


%}

#define DOLFINCC3D_EXPORT


//-----------------------------------------------------------------------------
// Include macros for shared_ptr support
//-----------------------------------------------------------------------------
%include <boost_shared_ptr.i>

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"



%include "dolfin/swig/import/mesh.i" // this file is essential to wrap any C++ class which inherits from SubDomain (or any non-dolfin class which inherits from dolfin class)
// %include "dolfin/swig/import/function.i" // when passing Function objects from Python to C++ and avoiding type casts it is necessary to include this file
//%include "dolfin/swig/forwarddeclarations.i" // this file is essential to wrap any C++ class which inherits from SubDomain (or any non-dolfin class which inherits from dolfin class)
//%include "dolfin/swig/globalincludes.i" // this file is essential to wrap any C++ class which inherits from SubDomain (or any non-dolfin class which inherits from dolfin class)
//%include "dolfin/swig/mesh/pre.i" // this file is essential to wrap any C++ class which inherits from SubDomain (or any non-dolfin class which inherits from dolfin class)


//%include <CompuCell3D/Field3D/Dim3D.h>

%template(vectorint) std::vector<unsigned char>;
%template(vectorlong) std::vector<long>;

// %shared_ptr(dolfin::Function) // when passing Function objects from Python to C++ and avoiding type casts it is necessary to include this file

%include <dolfinCC3D.h>
%include <CleaverDolfinUtil.h>




// namespace dolfin {
//   class Mesh;
//   template<typename T> class MeshFunction;
// }
// 
// %template() dolfin::MeshFunction<TYPE>;
// 
// // Shared_ptr declarations
// %shared_ptr(dolfin::MeshFunction<TYPE>)





// %shared_ptr(dolfin::MeshFunction<bool>)
// 
// %shared_ptr(dolfin::SubDomain)
// %include <dolfin/mesh/SubDomain.h>



%shared_ptr(dolfin::SubDomain) //many dolfin classes when wrapped in Python first have their pointers wrapped using boost shared ptr . Therefore it is necesary to let SWIG know how to handle base class (SubDomain) and that whatever inherits base class should be accessible viashared ptr to base class
// if we do not use shared_ptr(dolfin::SubDomain) we might get the sinilar error TypeError: in method 'SubDomain_mark', argument 1 of type 'dolfin::SubDomain const *'

%shared_ptr(dolfin::OmegaCustom1) // tell SWIG that dolfin::OmegaCustom1 ptr is also wrapped in shared_ptr
%shared_ptr(dolfin::OmegaCustom0)


%include <CustomSubDomains.h>







