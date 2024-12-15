
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

#include <numpy/arrayobject.h>


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

%include <core/pyinterface/FieldExtender/FieldExtender.i>

// %define FIELD3DEXTENDERBASE(className,returnType)
// %extend  className{
//
//         std::string __str__(){
//             std::ostringstream s;
//             s <<#className << " dim" << self->getDim();
//             return s.str();
//         }
//
//         returnType min(){
//             returnType minVal = self->get(Point3D(0, 0, 0));
//
//             Dim3D dim = self->getDim();
//
//             for (int x = 0; x < dim.x; ++x)
//                 for (int y = 0; y < dim.y; ++y)
//                     for (int z = 0; z < dim.z; ++z) {
//                         returnType val = self->get(Point3D(x, y, z));
//                         if (val < minVal) minVal = val;
//                     }
//
//             return minVal;
//
//         }
//
//         returnType max(){
//             returnType maxVal = self->get(Point3D(0, 0, 0));
//
//             Dim3D dim = self->getDim();
//
//             for (int x = 0; x < dim.x; ++x)
//                 for (int y = 0; y < dim.y; ++y)
//                     for (int z = 0; z < dim.z; ++z) {
//                         returnType val = self->get(Point3D(x, y, z));
//                         if (val > maxVal) maxVal = val;
//                     }
//
//             return maxVal;
//
//         }
//
//         returnType __getitem__(PyObject *_indexTuple) {
//             if (!PyTuple_Check(_indexTuple) || PyTuple_GET_SIZE(_indexTuple) != 3) {
//                 throw
//                 std::runtime_error(std::string(#className)+std::string(
//                         ": Wrong Syntax: Expected something like: field[1,2,3]"));
//             }
//             PyObject *xCoord = PyTuple_GetItem(_indexTuple, 0);
//             PyObject *yCoord = PyTuple_GetItem(_indexTuple, 1);
//             PyObject *zCoord = PyTuple_GetItem(_indexTuple, 2);
//             Py_ssize_t x, y, z;
//
//             //x-coord
//             if (PyInt_Check(xCoord)) {
//                 x = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 0));
//             } else if (PyLong_Check(xCoord)) {
//                 x = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 0));
//             } else if (PyFloat_Check(xCoord)) {
//                 x = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 0)));
//             } else {
//                 throw
//                 std::runtime_error(
//                         "Wrong Type (X): only integer or float values are allowed here - floats are rounded");
//             }
//             //y-coord
//             if (PyInt_Check(yCoord)) {
//                 y = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 1));
//             } else if (PyLong_Check(yCoord)) {
//                 y = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 1));
//             } else if (PyFloat_Check(yCoord)) {
//                 y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 1)));
//             } else {
//                 throw
//                 std::runtime_error(
//                         "Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
//             }
//             //z-coord
//             if (PyInt_Check(zCoord)) {
//                 z = PyInt_AsLong(PyTuple_GetItem(_indexTuple, 2));
//             } else if (PyLong_Check(zCoord)) {
//                 z = PyLong_AsLong(PyTuple_GetItem(_indexTuple, 2));
//             } else if (PyFloat_Check(zCoord)) {
//                 z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple, 2)));
//             } else {
//                 throw
//                 std::runtime_error(
//                         "Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
//             }
//
//             return self->get(Point3D(x, y, z));
//         }
// }
//
// %enddef
//
//
//
// %define FIELD3DEXTENDER(className,returnType)
// FIELD3DEXTENDERBASE(className,returnType)
//
// %extend className{
//
//     %pythoncode %{
//
//         def normalizeSlice(self, s):
//             norm = lambda x : x if x is None else int(round(x))
//             return slice ( norm(s.start),norm(s.stop), norm(s.step) )
//
//         def __setitem__(self,_indexTyple,_val):
//             newSliceTuple = tuple(map(lambda x : self.normalizeSlice(x) if isinstance(x,slice) else x , _indexTyple))
//             self.setitem(newSliceTuple,_val)
//
//     %}
//
//   void setitem(PyObject *_indexTuple,returnType _val) {
//   // void __setitem__(PyObject *_indexTuple,returnType _val) {
//     if (!PyTuple_Check( _indexTuple) || PyTuple_GET_SIZE(_indexTuple)!=3){
//         throw std::runtime_error("Wrong Syntax: Expected something like: field[1,2,3]=object");
//     }
//
//     PyObject *xCoord=PyTuple_GetItem(_indexTuple,0);
//     PyObject *yCoord=PyTuple_GetItem(_indexTuple,1);
//     PyObject *zCoord=PyTuple_GetItem(_indexTuple,2);
//
//     Py_ssize_t  start_x, stop_x, step_x, sliceLength;
//     Py_ssize_t  start_y, stop_y, step_y;
//     Py_ssize_t  start_z, stop_z, step_z;
//
//     Dim3D dim=self->getDim();
//
//     if (PySlice_Check(xCoord)){
// 		int ok = PySlice_GetIndices(xCoord, dim.x, &start_x, &stop_x, &step_x);
//
//      // cout<<"extracting slices for x axis"<<endl;
//      //cerr<<"start x="<< start_x<<endl;
//      //cerr<<"stop x="<< stop_x<<endl;
//      //cerr<<"step x="<< step_x<<endl;
//      //cerr<<"sliceLength="<<sliceLength<<endl;
//      //cerr<<"ok="<<ok<<endl;
//
//     }else{
//         if (PyInt_Check(xCoord)){
//             start_x=PyInt_AsLong(PyTuple_GetItem(_indexTuple,0));
//             stop_x=start_x;
//             step_x=1;
//         }else if (PyLong_Check(xCoord)){
//             start_x=PyLong_AsLong(PyTuple_GetItem(_indexTuple,0));
//             stop_x=start_x;
//             step_x=1;
//         }else if (PyFloat_Check(xCoord)){
//             start_x = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,0)));
//             stop_x=start_x;
//             step_x=1;
//         }
//         else{
//             throw std::runtime_error("Wrong Type (X): only integer or float values are allowed here - floats are rounded");
//         }
//
//         start_x %= dim.x;
//         stop_x %= dim.x;
//         stop_x += 1;
//
//         if (start_x < 0)
//             start_x = dim.x + start_x;
//
//         if (stop_x < 0)
//             stop_x = dim.x + stop_x;
//
//     }
//
//     if (PySlice_Check(yCoord)){
//
// 		int ok = PySlice_GetIndices(yCoord, dim.y, &start_y, &stop_y, &step_y);
//
//
//     }else{
//         if (PyInt_Check(yCoord)){
//             start_y=PyInt_AsLong(PyTuple_GetItem(_indexTuple,1));
//             stop_y=start_y;
//             step_y=1;
//         }else if (PyLong_Check(yCoord)){
//             start_y=PyLong_AsLong(PyTuple_GetItem(_indexTuple,1));
//             stop_y=start_y;
//             step_y=1;
//         }else if (PyFloat_Check(yCoord)){
//             start_y = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,1)));
//             stop_y=start_y;
//             step_y=1;
//         }
//         else{
//             throw std::runtime_error("Wrong Type (Y): only integer or float values are allowed here - floats are rounded");
//         }
//
//         start_y %= dim.y;
//         stop_y %= dim.y;
//         stop_y += 1;
//
//         if (start_y < 0)
//             start_y = dim.y + start_y;
//
//         if (stop_y < 0)
//             stop_y = dim.y + stop_y;
//
//     }
//
//     if (PySlice_Check(zCoord)){
//
// 	   int ok = PySlice_GetIndices(zCoord, dim.z, &start_z, &stop_z, &step_z);
//
//     }else{
//         if (PyInt_Check(zCoord)){
//             start_z=PyInt_AsLong(PyTuple_GetItem(_indexTuple,2));
//             stop_z=start_z;
//             step_z=1;
//         }else if (PyLong_Check(zCoord)){
//             start_z=PyLong_AsLong(PyTuple_GetItem(_indexTuple,2));
//             stop_z=start_z;
//             step_z=1;
//         }else if (PyFloat_Check(zCoord)){
//             start_z = (Py_ssize_t) round(PyFloat_AsDouble(PyTuple_GetItem(_indexTuple,2)));
//             stop_z=start_z;
//             step_z=1;
//         }
//         else{
//             throw std::runtime_error("Wrong Type (Z): only integer or float values are allowed here - floats are rounded");
//         }
//         start_z %= dim.z;
//         stop_z %= dim.z;
//         stop_z += 1;
//
//         if (start_z < 0)
//             start_z = dim.z + start_z;
//
//         if (stop_z < 0)
//             stop_z = dim.z + stop_z;
//
//
//     }
//
//
//     PyObject *sliceX=0,*sliceY=0,* sliceZ=0;
//
//     //cout << "start_x, stop_x = " << start_x << "," << stop_x << endl;
//     //cout << "start_y, stop_y = " << start_y << "," << stop_y << endl;
//     //cout << "start_z, stop_z = " << start_z << "," << stop_z << endl;
//     for (Py_ssize_t x=start_x ; x<stop_x ; x+=step_x)
//         for (Py_ssize_t y=start_y ; y<stop_y ; y+=step_y)
//             for (Py_ssize_t z=start_z ; z<stop_z ; z+=step_z){
//                 $self->set(Point3D(x,y,z),_val);
//             }
//
//   }
//
//
// }
// %enddef


//using namespace CompuCell3D; // use either this or use fully qualified class name (including namespace - as below)
%template(floatfieldaux) CompuCell3D::Field3D<float>;
%ignore CompuCell3D::Field3D<float>::typeStr;
FIELD3DEXTENDER(CompuCell3D::Field3D<float>,float)


%template(doublefieldaux) CompuCell3D::Field3D<double>;
%ignore CompuCell3D::Field3D<double>::typeStr;
FIELD3DEXTENDER(CompuCell3D::Field3D<double>,double)

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


%extend NdarrayAdapter<float,3>{

  void initFromNumpy(PyObject *_numpyArrayObj){
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=3){
            throw std::runtime_error(std::string("FloatField3D")+std::string(": Error: Array dimension should be 3"));
        }

        std::vector<long> strides(3,1);
        std::vector<long> shape(3,0);

        shape[0]=PyArray_DIM(pyarray,0);
        shape[1]=PyArray_DIM(pyarray,1);
        shape[2]=PyArray_DIM(pyarray,2);
        $self->setShape(shape);

        strides[0]=shape[2]*shape[1];
        strides[1]=shape[2];
        strides[2]=1;

        $self->setStrides(strides);

        $self->setData(static_cast<float*>(PyArray_DATA(pyarray)));
  }

  float getItem(const std::vector<long> & _coord){
      return (*($self))[_coord[0]][_coord[1]][_coord[2]];
  }

};


%extend NdarrayAdapter<float,4>{

  void initFromNumpy(PyObject *_numpyArrayObj){
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=4){
            throw std::runtime_error(std::string("VectorField3D")+std::string(": Error: Array dimension should be 4"));
        }

        std::vector<long> strides(4,1);
        std::vector<long> shape(4,0);

        shape[0]=PyArray_DIM(pyarray,0);
        shape[1]=PyArray_DIM(pyarray,1);
        shape[2]=PyArray_DIM(pyarray,2);
        shape[3]=PyArray_DIM(pyarray,3);

        $self->setShape(shape);

        strides[0]=shape[3]*shape[2]*shape[1];
        strides[1]=shape[3]*shape[2];
        strides[2]=shape[3];
        strides[3]=1;

        $self->setStrides(strides);


        $self->setData(static_cast<float*>(PyArray_DATA(pyarray)));
  }

  float getItem(const std::vector<long> & _coord){
      return (*($self))[_coord[0]][_coord[1]][_coord[2]][_coord[3]];
  }


};


%extend NdarrayAdapter<double,3>{

  void initFromNumpy(PyObject *_numpyArrayObj){
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=3){
            throw std::runtime_error(std::string("FloatField3D")+std::string(": Error: Array dimension should be 3"));
        }

        std::vector<long> strides(3,1);
        std::vector<long> shape(3,0);

        shape[0]=PyArray_DIM(pyarray,0);
        shape[1]=PyArray_DIM(pyarray,1);
        shape[2]=PyArray_DIM(pyarray,2);
        $self->setShape(shape);

        strides[0]=shape[2]*shape[1];
        strides[1]=shape[2];
        strides[2]=1;

        $self->setStrides(strides);

        $self->setData(static_cast<double*>(PyArray_DATA(pyarray)));
  }

  double getItem(const std::vector<long> & _coord){
      return (*($self))[_coord[0]][_coord[1]][_coord[2]];
  }

};


%extend NdarrayAdapter<double,4>{

  void initFromNumpy(PyObject *_numpyArrayObj){
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=4){
            throw std::runtime_error(std::string("VectorField3D")+std::string(": Error: Array dimension should be 4"));
        }

        std::vector<long> strides(4,1);
        std::vector<long> shape(4,0);

        shape[0]=PyArray_DIM(pyarray,0);
        shape[1]=PyArray_DIM(pyarray,1);
        shape[2]=PyArray_DIM(pyarray,2);
        shape[3]=PyArray_DIM(pyarray,3);

        $self->setShape(shape);

        strides[0]=shape[3]*shape[2]*shape[1];
        strides[1]=shape[3]*shape[2];
        strides[2]=shape[3];
        strides[3]=1;

        $self->setStrides(strides);


        $self->setData(static_cast<double*>(PyArray_DATA(pyarray)));
  }

  double getItem(const std::vector<long> & _coord){
      return (*($self))[_coord[0]][_coord[1]][_coord[2]][_coord[3]];
  }


};
