
// Module Name
%module("threads"=1) Fields

//%module Example
// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{

#include <sstream>  
#include <BasicUtils/BasicException.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Field3D/Dim3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/Field3DImpl.h>



// #define DOLFINCC3D_EXPORT
// Namespaces
using namespace std;
using namespace CompuCell3D;


%}

// #define DOLFINCC3D_EXPORT











// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

//enables better handling of STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}


%include <BasicUtils/BasicException.h>


%include <CompuCell3D/Field3D/Point3D.h>

%extend CompuCell3D::Point3D{
  std::string __str__(){
    std::ostringstream s;
    s<<(*self);
    return s.str();
  }
};


%include <CompuCell3D/Field3D/Dim3D.h>

%extend CompuCell3D::Dim3D{
  std::string __str__(){
    std::ostringstream s;
    s<<(*self);
    return s.str();
  }
};


%include <CompuCell3D/Field3D/Field3D.h>
%include <CompuCell3D/Field3D/Field3DImpl.h>

%ignore CompuCell3D::Field3D<int>::typeStr;
%ignore CompuCell3D::Field3DImpl<int>::typeStr;

%template(Field3DInt) CompuCell3D::Field3D<int>;
%template(Field3DImplInt) CompuCell3D::Field3DImpl<int>;


%define FIELD3DEXTENDER(type,returnType)
%extend  type{
  std::string __str__(){
    std::ostringstream s;
    s<<#type<<" dim"<<self->getDim();
    return s.str();
  }
    
  returnType __getitem__(PyObject *_indexTuple) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error(" Wrong Syntax: Expected someting like: field[1,2,3]");
    }

    return self->get(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))));    
  }

  void __setitem__(PyObject *_indexTuple,returnType _val) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error("Wrong Syntax: Expected someting like: field[1,2,3]=object");
    }

    return self->set(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))),_val);    
  }
  
  
}
%enddef    

 FIELD3DEXTENDER(CompuCell3D::Field3D<int>,int)
 FIELD3DEXTENDER(CompuCell3D::Field3DImpl<int>,int)

%extend CompuCell3D::Field3DImpl<int>{
//   std::string __str__(){
//     std::ostringstream s;
//     s<<"Field3DImpl<int> dim"<<self->getDim();
//     return s.str();
//   }
  
//   %pythoncode %{
//     
//     %}
  
  
//   int __getitem__(PyObject *_indexTuple) {
//     if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
// 	throw std::runtime_error("Wrong Syntax: please use make sure you access field like here: field[1,2,3]");
//     }
// 
//     return self->get(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))));    
//   }
// 
//   void __setitem__(PyObject *_indexTuple,int _val) {
//     if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
// 	throw std::runtime_error("Wrong Syntax: please use make sure you access field like here: field[1,2,3]=1.0");
//     }
// 
//     return self->set(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))),_val);    
//   }
  
  
//   int __getitem__(PyObject *_indexTuple) {
// //     std::ostringstream s;
//     cerr<<"INSIDE GETITEM"<<endl;
//     cerr<<"PyTuple_Check( _indexTuple)="<<PyTuple_Check( _indexTuple)<<endl;
//     cerr<<"PyTuple_GET_SIZE(_indexTuple)="<<PyTuple_GET_SIZE(_indexTuple)<<endl;
//     if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
// 	cerr<<"THIS IS EXCEPTION HANDLING"<<endl;
// // 	PyErr_SetString(PyExc_SyntaxError,"Wrong Syntax: please use make sure you access field like here: field[1,2,3]");
// 	throw std::runtime_error("Wrong Syntax: please use make sure you access field like here: field[1,2,3]");
// // 	throw BasicException("Wrong Syntax: please use make sure you access field like here: field[1,2,3]");
// 	  
// 	
// // 	SWIG_exception(SWIG_SyntaxError,"Wrong Syntax: Expected different syntax e.g. field[1,2,3]");
// // 	return "PROBLEM WITH SYNTAX";
// // 	return NULL;
//     }
// //     int check=PyTuple_Check( _indexTuple);
//     return self->get(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))));
// //     int x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
// //     int y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
// //     int z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
// //     int val=self->get(Point3D(x,y,z));
// //     s<<"THIS IS INDEX "<<_indexTuple<<" check="<<check <<" x="<<x<<" y="<<y<<" z="<<z<<" val="<<val;
// //     // s<<"THIS IS INDEX "<<i<<" and "<<j;
//     
// //     return s.str();
//     
//   }
};



// %include <CompuCell3D/Field3D/Dim3D.h>


// %include <CustomSubDomains.h>

// %include <CustomExpressions.h>


