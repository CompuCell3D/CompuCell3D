
// Module Name
// %module("threads"=1) Example
%module Example
// #define SwigPyIterator Example_SwigPyIterator
// %{
// #define SwigPyIterator Example_SwigPyIterator
// %}
// %rename (Example_SwigPyIterator) SwigPyIterator;

//%module Example
// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{
#include <PyNewPlugin.h>


#define EXAMPLECLASS_EXPORT
// Namespaces
using namespace std;


%}

#define EXAMPLECLASS_EXPORT
// %ignore SwigPyIterator;
// %rename (SwigPyIteratorExample) SwigPyIterator;
// #define SwigPyIterator Example_PySwigIterator


// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"


%include <PyNewPlugin.h>



