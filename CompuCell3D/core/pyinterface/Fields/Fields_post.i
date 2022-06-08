
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

%pythoncode %{
    def __setitem__(self,_indexTyple,_val):
        print 'this is setitem in python'
        
        self.setitem(_indexTyple,_val,self.volumeTracker)
%}
    
    
  std::string __str__(){
    std::ostringstream s;
    s<<#type<<" dim"<<self->getDim();
    return s.str();
  }
  

//   void numberAttrib_set(int _i){numberAttrib=_i;}
//   int numberAttrib_get(void){return numberAttrib;}
  
  returnType __getitem__(PyObject *_indexTuple) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error(" Wrong Syntax: Expected someting like: field[1,2,3]");
    }

    return self->get(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))));    
  }
  
  
  void setitem(PyObject *_indexTuple,returnType _val,void *volumeTracker=0) {
    if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
        throw std::runtime_error("Wrong Syntax: Expected someting like: field[1,2,3]=object");
    }
    
    PyObject *xCoord=PyTuple_GetItem(_indexTuple,0);
    PyObject *yCoord=PyTuple_GetItem(_indexTuple,1);
    PyObject *zCoord=PyTuple_GetItem(_indexTuple,2);
    
    Py_ssize_t  start_x, stop_x, step_x, sliceLength;
    Py_ssize_t  start_y, stop_y, step_y;
    Py_ssize_t  start_z, stop_z, step_z;
    
    Dim3D dim=self->getDim();
    
    if (PySlice_Check(xCoord)){
//         cerr<<"inside x slice"<<endl;
        PySlice_GetIndicesEx((PySliceObject*)xCoord,dim.x-1,&start_x,&stop_x,&step_x,&sliceLength);
//         cerr<<"AFTER SLICE INDEX PROCESSING"<<endl;
        
//         cerr<<"start_x="<<PyInt_AsSsize_t((PyObject*)&start_x)<<endl;
//         cerr<<"stop_x="<<PyInt_AsSsize_t((PyObject*)&stop_x)<<endl;
//         cerr<<"step_x="<<PyInt_AsSsize_t((PyObject*)&step_x)<<endl;
        
    }else{
        start_x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
        stop_x=start_x;
        step_x=1;
    }

    if (PySlice_Check(yCoord)){
        
        PySlice_GetIndicesEx((PySliceObject*)yCoord,dim.y-1,&start_y,&stop_y,&step_y,&sliceLength);
        
        
    }else{
        start_y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
        stop_y=start_y;
        step_y=1;
    }
    
    if (PySlice_Check(zCoord)){
        
        PySlice_GetIndicesEx((PySliceObject*)zCoord,dim.z-1,&start_z,&stop_z,&step_z,&sliceLength);
        
        
    }else{
        start_z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
        stop_z=start_z;
        step_z=1;
    }

    
//     cerr<<"start x="<< start_x<<endl;
//     cerr<<"stop x="<< stop_x<<endl;
//     cerr<<"step x="<< step_x<<endl;
//     cerr<<"sliceLength="<<sliceLength<<endl;
    
    
    int x,y,z;
    PyObject *sliceX=0,*sliceY=0,* sliceZ=0;
    
    for (Py_ssize_t x=start_x ; x<=stop_x ; x+=step_x)
        for (Py_ssize_t y=start_y ; y<=stop_y ; y+=step_y)
            for (Py_ssize_t z=start_z ; z<=stop_z ; z+=step_z){
                $self->set(Point3D(x,y,z),_val); 
//                 $self->runSteppers();
//                 CompuCell3D_Field3D_Sl_int_Sg__runSteppers(self);
            }
    
//     return self->set(Point3D(PyInt_AsLong(PyTuple_GetItem(_indexTuple,0)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,1)),PyInt_AsLong(PyTuple_GetItem(_indexTuple,2))),_val);    
  }
  
  
}
%enddef    

 FIELD3DEXTENDER(CompuCell3D::Field3D<int>,int)
 FIELD3DEXTENDER(CompuCell3D::Field3DImpl<int>,int)
