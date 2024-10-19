
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

typedef std::vector<double>::size_type array_size_t;

%template (cc3dauxfield_vectordouble) std::vector<double>;
%template (cc3dauxfield_vector_array_size_t) std::vector<array_size_t>;
%template (cc3dauxfield_vector_unsigned_int) std::vector<unsigned int>;

%include <NumpyArrayWrapper.h>
