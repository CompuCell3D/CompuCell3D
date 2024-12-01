
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
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>

#include <CompuCell3D/Field3D/ndarray_adapter.h>


#include <core/Utils/Coordinates3D.h>

#include <CompuCell3D/Field3D/VectorField3D.h>
#include <CompuCell3D/Field3D/VectorNumpyArrayWrapper3DImpl.h>


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
%include "Field3D/Point3D.h"
%include "Field3D/Dim3D.h"
%include "Field3D/Field3D.h"
%include "Field3D/Field3DImpl.h"
%include "Field3D/WatchableField3D.h"
%include "Field3D/ndarray_adapter.h"


%include <core/Utils/Coordinates3D.h>
%include "Field3D/VectorField3D.h"


%include <NumpyArrayWrapper.h>
%include <NumpyArrayWrapperImpl.h>
%include <NumpyArrayWrapper3DImpl.h>

%include <VectorNumpyArrayWrapper3DImpl.h>

//using namespace CompuCell3D; // use either this or use fully qualified class name (including namespace - as below)
%template(floatfieldaux) CompuCell3D::Field3D<float>;
%ignore CompuCell3D::Field3D<float>::typeStr;
%template(doublefieldaux) CompuCell3D::Field3D<double>;
%ignore CompuCell3D::Field3D<double>::typeStr;

%template(vector_ndarray_adapter_float) NdarrayAdapter<float, 4>;
%template(vector_ndarray_adapter_double) NdarrayAdapter<double, 4>;

%template(float_vector_field_3_impl_daux) CompuCell3D::VectorField3D<float>;

%template (cc3dauxfield_vectorsize_t) std::vector<size_t>;
%template (cc3dauxfield_vectordouble) std::vector<double>;
%template (cc3dauxfield_vectorfloat) std::vector<float>;

%template (cc3dauxfield_coordinates3d_float) Coordinates3D<float>;
%template (cc3dauxfield_coordinates3d_double) Coordinates3D<double>;
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


%template (VectorNumpyArrayWrapper3DImplFloat) CompuCell3D::VectorNumpyArrayWrapper3DImpl<float>;
%template (VectorNumpyArrayWrapper3DImplDouble) CompuCell3D::VectorNumpyArrayWrapper3DImpl<double>;