// Module Name
%module("threads"=1) FiPyInterface

%include stl.i

// Pointer handling
%include "cpointer.i"
%include "typemaps.i"

// C++ std::vector handling
%include "std_vector.i"


namespace std
{
   %template(IntVector) std::vector<int>;
   %template(IntIntVector) std::vector<std::vector<int> >;
}


%{
#include <FiPyInterface.h>



using namespace std;
using namespace CompuCell3D;
#define FIPYINTERFACEEXTRACTOR_EXPORT

%}

#define FIPYINTERFACEEXTRACTOR_EXPORT
%include <FiPyInterface.h>



//%typemap(out) vector<int> {
//  $result = SWIG_NewPointerObj($1, SWIGTYPE_p_MyObject, 0);
//}

%inline %{
   void fillArray3D_inline(PyObject * _FiPyArray,void * _Field) {
      cout << "Test Inline" << endl;
   }
    void ExampleFunc2(PyObject * _FiPyArray,CompuCell3D::Field3D<float>* _Field) {
	    cout << "Hello From ExampleFunc2\n";
    }
%}


