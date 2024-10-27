
// Module Name
%module CC3DAuxFields

// ************************************************************
// Module Includes
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.



%{
#include <NumpyArrayWrapper.h>
#include <NumpyArrayWrapperImpl.h>
#include <NumpyArrayWrapper3DImpl.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>


//#define XMLUTILS_EXPORT
// Namespaces
using namespace std;
using namespace CompuCell3D;

%}





//#define XMLUTILS_EXPORT

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

//C++ std::list handling
%include "std_list.i"

//typedef std::vector<size_t>::size_type array_size_t;


//%include <CompuCell3D/Field3D/Field3D.h>
%include "Field3D/Field3D.h"
%include "Field3D/Field3DImpl.h"
%include "Field3D/WatchableField3D.h"

%include <NumpyArrayWrapper.h>
%include <NumpyArrayWrapperImpl.h>
%include <NumpyArrayWrapper3DImpl.h>

//using namespace CompuCell3D; // use either this or use fully qualified class name (including namespace - as below)
%template(floatfieldaux) CompuCell3D::Field3D<float>;
%ignore CompuCell3D::Field3D<float>::typeStr;
%template(doublefieldaux) CompuCell3D::Field3D<double>;
%ignore CompuCell3D::Field3D<double>::typeStr;


%template (cc3dauxfield_vectorsize_t) std::vector<size_t>;
%template (cc3dauxfield_vectordouble) std::vector<double>;
//
//%template (cc3dauxfield_vector_array_size_t) std::vector<array_size_t>;
//%template (cc3dauxfield_vector_unsigned_int) std::vector<unsigned int>;
//
//
//
//
//
//
//
//
//
//
%template (NumpyArrayWrapperImplDouble) CompuCell3D::NumpyArrayWrapperImpl<double>;
%template (NumpyArrayWrapperImplFloat) CompuCell3D::NumpyArrayWrapperImpl<float>;


%template (NumpyArrayWrapper3DImplDouble) CompuCell3D::NumpyArrayWrapper3DImpl<double>;
%template (NumpyArrayWrapper3DImplFloat) CompuCell3D::NumpyArrayWrapper3DImpl<float>;
