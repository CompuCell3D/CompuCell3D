
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

%include "stdint.i"
//needed for mapping between size_t/array_size_t and Python long integers
%typemap(out) size_t {
    $result = PyLong_FromSize_t($1);
}

%typemap(out) CompuCell3D::array_size_t {
    $result = PyLong_FromSize_t($1);
}


%typemap(in) std::vector<CompuCell3D::array_size_t> const & (std::vector<CompuCell3D::array_size_t> temp_vec) {
    if (!PySequence_Check($input)) {
        SWIG_exception_fail(SWIG_TypeError, "Expected a sequence (list or tuple)");
    }
    Py_ssize_t size = PySequence_Size($input);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PySequence_GetItem($input, i);
        if (!PyLong_Check(item)) {
            SWIG_exception_fail(SWIG_TypeError, "All items must be integers");
        }
        temp_vec.push_back((CompuCell3D::array_size_t) PyLong_AsUnsignedLong(item));
        Py_DECREF(item);
    }
    $1 = &temp_vec;
}




%typemap(out) std::vector<size_t> {
    $result = PyList_New($1.size());
    for (size_t i = 0; i < $1.size(); ++i) {
        PyList_SetItem($result, i, PyLong_FromSize_t($1[i]));
    }
}

%template(cc3dauxfield_vectorsize_t) std::vector<size_t>;

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


%extend CompuCell3D::Point3D{
        std::string __str__(){
            std::ostringstream s;
            s<<(*self);
            return s.str();
        }



        %pythoncode %{
            def __getstate__(self):
                return (self.x,self.y,self.z)

            def __setstate__(self,tup):
                print( 'tuple=',tup)
                self.this = _CompuCell.new_Point3D(tup[0],tup[1],tup[2])
                self.thisown=1

            def to_tuple(self):
                return self.x, self.y, self.z

%}
};



%extend CompuCell3D::Dim3D{
        std::string __str__(){
            std::ostringstream s;
            s<<(*self);
            return s.str();
        }

        %pythoncode %{
            def to_tuple(self):
                return self.x, self.y, self.z

            def __reduce__(self):
                return Dim3D, (self.x, self.y, self.z)
  %}
};


%template(floatfieldaux) CompuCell3D::Field3D<float>;
%ignore CompuCell3D::Field3D<float>::typeStr;
FIELD3DEXTENDER(CompuCell3D::Field3D<float>,float)


%template(doublefieldaux) CompuCell3D::Field3D<double>;
%ignore CompuCell3D::Field3D<double>::typeStr;
FIELD3DEXTENDER(CompuCell3D::Field3D<double>,double)

%template(vector_ndarray_adapter_float) NdarrayAdapter<float, 4>;
%template(vector_ndarray_adapter_double) NdarrayAdapter<double, 4>;

%template(float_vector_field_3_impl_daux) CompuCell3D::VectorField3D<float>;

//%template (cc3dauxfield_vectorsize_t) std::vector<size_t>;
%template (cc3dauxfield_vectordouble) std::vector<double>;
%template (cc3dauxfield_vectorfloat) std::vector<float>;

%template (cc3dauxfield_coordinates3d_float) Coordinates3D<float>;
%template (cc3dauxfield_coordinates3d_double) Coordinates3D<double>;


//
//%template (cc3dauxfield_vector_array_size_t) std::vector<array_size_t>;
//%template (cc3dauxfield_vector_unsigned_int) std::vector<unsigned int>;
//
//


%template (NumpyArrayWrapperImplChar) CompuCell3D::NumpyArrayWrapperImpl<char>;
%template (NumpyArrayWrapperImplUChar) CompuCell3D::NumpyArrayWrapperImpl<unsigned char>;

%template (NumpyArrayWrapperImplShort) CompuCell3D::NumpyArrayWrapperImpl<short>;
%template (NumpyArrayWrapperImplUShort) CompuCell3D::NumpyArrayWrapperImpl<unsigned short>;

%template (NumpyArrayWrapperImplInt) CompuCell3D::NumpyArrayWrapperImpl<int>;
%template (NumpyArrayWrapperImplUInt) CompuCell3D::NumpyArrayWrapperImpl<unsigned int>;

%template (NumpyArrayWrapperImplLong) CompuCell3D::NumpyArrayWrapperImpl<long>;
%template (NumpyArrayWrapperImplULong) CompuCell3D::NumpyArrayWrapperImpl<unsigned long>;


%template (NumpyArrayWrapperImplDouble) CompuCell3D::NumpyArrayWrapperImpl<double>;
%template (NumpyArrayWrapperImplFloat) CompuCell3D::NumpyArrayWrapperImpl<float>;

// Ignore a specific method
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<float>::getType;
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<double>::getType;
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<char>::getType;
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<unsigned char>::getType;
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<short>::getType;
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<unsigned short>::getType;

%ignore CompuCell3D::NumpyArrayWrapper3DImpl<int>::getType;
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<unsigned int>::getType;

%ignore CompuCell3D::NumpyArrayWrapper3DImpl<long>::getType;
%ignore CompuCell3D::NumpyArrayWrapper3DImpl<unsigned long>::getType;


//scalar fields
%template (NumpyArrayWrapper3DImplDouble) CompuCell3D::NumpyArrayWrapper3DImpl<double>;
%template (NumpyArrayWrapper3DImplFloat) CompuCell3D::NumpyArrayWrapper3DImpl<float>;

%template (NumpyArrayWrapper3DImplChar) CompuCell3D::NumpyArrayWrapper3DImpl<char>;
%template (NumpyArrayWrapper3DImplUChar) CompuCell3D::NumpyArrayWrapper3DImpl<unsigned char>;

%template (NumpyArrayWrapper3DImplShort) CompuCell3D::NumpyArrayWrapper3DImpl<short>;
%template (NumpyArrayWrapper3DImplUShort) CompuCell3D::NumpyArrayWrapper3DImpl<unsigned short>;

%template (NumpyArrayWrapper3DImplInt) CompuCell3D::NumpyArrayWrapper3DImpl<int>;
%template (NumpyArrayWrapper3DImplUInt) CompuCell3D::NumpyArrayWrapper3DImpl<unsigned int>;

%template (NumpyArrayWrapper3DImplLong) CompuCell3D::NumpyArrayWrapper3DImpl<long>;
%template (NumpyArrayWrapper3DImplULong) CompuCell3D::NumpyArrayWrapper3DImpl<unsigned long>;

//vector fields
%template (VectorNumpyArrayWrapper3DImplFloat) CompuCell3D::VectorNumpyArrayWrapper3DImpl<float>;
%template (VectorNumpyArrayWrapper3DImplDouble) CompuCell3D::VectorNumpyArrayWrapper3DImpl<double>;


//%extend CompuCell3D::VectorNumpyArrayWrapper3DImpl<float> {
//        VectorNumpyArrayWrapper3DImpl(const std::vector<size_t>& dims) {
//            std::vector<CompuCell3D::array_size_t> conv_dims(dims.begin(), dims.end());
//            return new CompuCell3D::VectorNumpyArrayWrapper3DImpl<float>(conv_dims);
//        }
//}

//%extend CompuCell3D::VectorNumpyArrayWrapper3DImpl<float> {
//        static CompuCell3D::VectorNumpyArrayWrapper3DImpl<float>* fromSizeTVector(const std::vector<size_t>& dims) {
//            std::vector<CompuCell3D::array_size_t> conv_dims(dims.begin(), dims.end());
//            return new CompuCell3D::VectorNumpyArrayWrapper3DImpl<float>(conv_dims);
//        }
//}

%extend CompuCell3D::VectorNumpyArrayWrapper3DImpl<float> {
        static CompuCell3D::VectorNumpyArrayWrapper3DImpl<float>* fromSizeTVector(std::vector<size_t> *dims) {
            std::vector<CompuCell3D::array_size_t> conv_dims(dims->begin(), dims->end());
            return new CompuCell3D::VectorNumpyArrayWrapper3DImpl<float>(conv_dims);
        }
}

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
