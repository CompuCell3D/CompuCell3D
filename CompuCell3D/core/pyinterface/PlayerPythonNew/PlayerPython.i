
// Module Name
%module("threads"=1) PlayerPython


%include "windows.i"

//%include "typemaps.i"

// *************************************************************
// Module Includes 
// *************************************************************


namespace CompuCell3D{
 class CellG;   
 typedef CellG * cellGPtr_t;
}


// in SWIG tydefs have to be explicitly redeclared in the interface (.i) file. Also note that SWIG struggles with proper handling of
// preprocessor _WIN32 macros so it is best to add -DSWIGWIN option to the actual swig command and look for this Macro together with _WIN32

%inline %{
#if defined(SWIGWIN) || defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)

	typedef long long vtk_obj_addr_int_t;

#else
	typedef long vtk_obj_addr_int_t;

#endif



%}


%{

// #include <Potts3D/Cell.h>
#include <Utils/Coordinates3D.h>
#include <FieldStorage.h>

#include <ndarray_adapter.h>

#include <FieldExtractorTypes.h>

#include <FieldExtractorBase.h>
#include <FieldExtractor.h>
#include <FieldExtractorCML.h>
#include <FieldWriter.h>
#include <FieldWriterCML.h>
#include <FieldStreamer.h>
#include <vtkIntArray.h>
    
#define FIELDEXTRACTOR_EXPORT

// System Libraries
#include <iostream>
#include <stdlib.h>

#include <numpy/arrayobject.h>

   
// Namespaces
using namespace std;
using namespace CompuCell3D;
class CellG;


%}

#define FIELDEXTRACTOR_EXPORT

//necessary to get proper wrapping of the numpy arrays
%include "swig_includes/numpy.i"

%init %{
    import_array();
%}


// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

// Pointer handling
%include "cpointer.i"

//enables better handling of STL exceptions
%include "exception.i"

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}




%include <Utils/Coordinates3D.h>

// this is a bug in swig - it is looking for a trait for CompuCell3D::CellG when wrapping  %template(mapCellGPtrToFloat) std::map<CompuCell3D::CellG*,double>;
// the easies workaround is to wrap vector <CompuCell3D::CellG> which iwill produce required trait instantiation
// more on this topic http://cvblob.googlecode.com/svn-history/r361/branches/0.10.4_pythonswig/interfaces/swig/general/cvblob.i
%template(vectorCell) std::vector<CompuCell3D::CellG>;
%template(mapCellGPtrToFloat) std::map<CompuCell3D::CellG*,float>;

%template(Coodrinates3DFloat) Coordinates3D<float>;
%template(mapCellGPtrToCoordinates3DFloat) std::map<CompuCell3D::CellG*,Coordinates3D<float> >;


%template(vectorint) std::vector<int>;
%template(vectorlong) std::vector<long>;
%template(vectorfloat) std::vector<float>;
%template(vectorstring) std::vector<std::string>;




%include <ndarray_adapter.h>

%template(NdarrayAdapterDouble3) NdarrayAdapter<float,3>; //for storing scalar fieldas
%template(NdarrayAdapterDouble4) NdarrayAdapter<float,4>; //for storing vector fieldas

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

%include <FieldExtractorTypes.h>
%include <FieldStorage.h>
%include <FieldExtractorBase.h>
%include <FieldExtractor.h>
%include <FieldExtractorCML.h>
%include <FieldWriter.h>
%include <FieldWriterCML.h>
%include <FieldStreamer.h>


%extend CompuCell3D::ScalarFieldCellLevel{    


  void __setitem__(CompuCell3D::CellG * _cell,float _val) {
      (*($self))[_cell]=_val;

  }
  
  float __getitem__(CompuCell3D::CellG * _cell) {
      //this has side effect that if the _cell is not in the map it will be inserted with  matching value 0.0
      return (*($self))[_cell];
      
  }

  
};

// needed in numpy 1.22 and higher to get PyArray_SimpleNew. any function
// using PyArray_SimpleNew cannot release GIL
// see https://stackoverflow.com/questions/74861186/access-vaiolation-in-pyarray-simplenew
%nothread CompuCell3D::VectorFieldCellLevel::__getitem__;
%nothread CompuCell3D::VectorFieldCellLevel::__setitem__;

%extend CompuCell3D::VectorFieldCellLevel{    

  void __setitem__(CompuCell3D::CellG * _cell,PyObject *_numpyArrayObj) {
      
        if (PyList_Check(_numpyArrayObj)){//in case user passes regular python list instead of numpy array

            if (PyList_Size(_numpyArrayObj)!=3){
                throw std::runtime_error(std::string("VectorFieldCellLevel")+std::string(": Error: Array dimension should be 3"));
            }
            float x,y,z;
            x=PyFloat_AsDouble(PyList_GetItem(_numpyArrayObj,0));
            y=PyFloat_AsDouble(PyList_GetItem(_numpyArrayObj,1));
            z=PyFloat_AsDouble(PyList_GetItem(_numpyArrayObj,2));
            (*($self))[_cell]=Coordinates3D<float>(x,y,z);
            return;
        }
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj); 

        int dim=PyArray_DIM(pyarray,0);

        if (PyArray_DIM(pyarray,0)!=3){         
            throw std::runtime_error(std::string("VectorFieldCellLevel")+std::string(": Error: Array dimension should be 3"));
        }
        
        float *data =static_cast<float*>(PyArray_DATA(pyarray));
      
      (*($self))[_cell]=Coordinates3D<float>(data[0],data[1],data[2]);

  }
  
// %pythoncode %{
//     def __setitem__(self,_cell,_array):# we intercept assignments of the array and wrap any array object in numpy array and pass it to C++ fcn.
//         import numpy
//         self.setitem(_cell,numpy.array(_array,dtype=numpy.float32))
// %}  

  
  PyObject* __getitem__(CompuCell3D::CellG * _cell) {
      Coordinates3D<float> &vec=(*($self))[_cell];
//      cerr<<"x,y,z="<<vec.x<<","<<vec.y<<","<<vec.z<<endl;
     int size=3;
    npy_intp dims[] = {size};


    PyObject* numpyArray= PyArray_SimpleNew(1,dims,NPY_FLOAT);

    float *data =static_cast<float*>(PyArray_DATA(numpyArray));
    data[0]=vec.x;
    data[1]=vec.y;
    data[2]=vec.z;

    return numpyArray;
  }
};


%inline %{
	void setSwigPtr(void * _ptr){
		using namespace std;
		cerr<<"THIS IS setSwigPtr"<<endl;
		
	}
	
	void add(double a, double b, double *result) {
		*result = a + b;
        
	}
    
		
%}

%inline %{

   void fillScalarValue(PyObject * _numpyArrayObj, int _x, int _y, int _z, float _value){
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=3){         
            throw std::runtime_error(std::string("FloatField3D")+std::string(": Error: Array dimension shuold be 3"));
        }
       
        int dim_x=PyArray_DIM(pyarray,0);
        int dim_y=PyArray_DIM(pyarray,1);
        int dim_z=PyArray_DIM(pyarray,2);
        
        float * data=static_cast<float*>(PyArray_DATA(pyarray));
        data[_x*dim_z*dim_y + _y*dim_z + _z]=_value; //assuming default numpy strides
        
   }
    
   void clearScalarField(CompuCell3D::Dim3D _dim, PyObject * _numpyArrayObj){ 
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=3){         
            throw std::runtime_error(std::string("FloatField3D")+std::string(": Error: Array dimension should be 3"));
        }
       
        int dim_x=PyArray_DIM(pyarray,0);
        int dim_y=PyArray_DIM(pyarray,1);
        int dim_z=PyArray_DIM(pyarray,2);
        
        float * data=static_cast<float*>(PyArray_DATA(pyarray));
        for (int i = 0; i < dim_x*dim_y*dim_z;++i){
            data[i]=0.0;
        }
        
       

   }     
    

   void clearScalarValueCellLevel(CompuCell3D::FieldStorage::scalarFieldCellLevel_t * _field){ 
		_field->clear();
   }     
   	
   void fillScalarValueCellLevel(CompuCell3D::FieldStorage::scalarFieldCellLevel_t * _field, CompuCell3D::CellG* _cell, float _value){
      _field->insert(std::make_pair(_cell,_value));
   }
   
   
    void insertVectorIntoVectorField(PyObject * _numpyArrayObj,int _xPos, int _yPos, int _zPos, float _x, float _y, float _z){
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=4){         
            throw std::runtime_error(std::string("VectorField3D")+std::string(": Error: Array dimension should be 4"));
        }
       
        int dim_x=PyArray_DIM(pyarray,0);
        int dim_y=PyArray_DIM(pyarray,1);
        int dim_z=PyArray_DIM(pyarray,2);
        
        
        
        float * data=static_cast<float*>(PyArray_DATA(pyarray));
        //assuming default numpy strides
        long baseInd=3*_xPos*dim_z*dim_y + 3*_yPos*dim_z + 3*_zPos; // this gets us to the 3 dim vector
        data[baseInd+0]=_x; 
        data[baseInd+1]=_y; 
        data[baseInd+2]=_z; 
        

    }   
   
    
   void insertVectorIntoVectorCellLevelField(CompuCell3D::FieldStorage::vectorFieldCellLevel_t * _field,CompuCell3D::CellG* _cell, float _x, float _y, float _z){
       
      _field->insert(std::make_pair(_cell,Coordinates3D<float>(_x,_y,_z)));
   }

   void clearVectorCellLevelField(CompuCell3D::FieldStorage::vectorFieldCellLevel_t * _field){
      _field->clear();
   }
   


    void clearVectorField(CompuCell3D::Dim3D _dim, PyObject * _numpyArrayObj){
        
        PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
        
        int ndim=PyArray_NDIM(pyarray);
        if (ndim!=4){         
            throw std::runtime_error(std::string("VectorField3D")+std::string(": Error: Array dimension should be 4"));
        }
       
        int dim_x=PyArray_DIM(pyarray,0);
        int dim_y=PyArray_DIM(pyarray,1);
        int dim_z=PyArray_DIM(pyarray,2);
        int dim_vec=PyArray_DIM(pyarray,3);
        
        float * data=static_cast<float*>(PyArray_DATA(pyarray));
        for (int i = 0; i < dim_x*dim_y*dim_z*dim_vec;++i){
            data[i]=0.0;
        }



    }
   
   Coordinates3D<float> * findVectorInVectorCellLEvelField(CompuCell3D::FieldStorage::vectorFieldCellLevel_t * _field,CompuCell3D::CellG* _cell){
      CompuCell3D::FieldStorage::vectorFieldCellLevelItr_t vitr;
      vitr=_field->find(_cell);
      if(vitr != _field->end()){
         return & vitr->second;
      }else{

         return 0;
      }

   }

%}

