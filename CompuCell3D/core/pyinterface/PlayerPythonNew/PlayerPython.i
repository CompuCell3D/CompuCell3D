
// Module Name
%module("threads"=1) PlayerPython




// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.

namespace CompuCell3D{
 class CellG;   
 typedef CellG * cellGPtr_t;
}

%{

#include <Potts3D/Cell.h>
#include <Utils/Coordinates3D.h>
#include <GraphicsData.h>
#include <FieldStorage.h>

#include <ndarray_adapter.h>

#include <FieldExtractorBase.h>
#include <FieldExtractor.h>
#include <FieldExtractorCML.h>
#include <FieldWriter.h>
//#include <CompuCell3D/Field3D/Point3D.h>
//#include <CompuCell3D/Field3D/Dim3D.h>
#include <vtkIntArray.h>

// #include <MyArray.h>
    
#define FIELDEXTRACTOR_EXPORT

   

// System Libraries
#include <iostream>
#include <stdlib.h>
#include <Coordinates3D.h>


#include <numpy/arrayobject.h>

   
// Namespaces
using namespace std;
using namespace CompuCell3D;

%}

#define FIELDEXTRACTOR_EXPORT

%include "numpy.i"


%init %{
    import_array();
%}
// // // //notice that the names IN_ARRAY3, IN_ARRAY2,IN_ARRAY1 are actually important! With different name the numpy typemaps will not be handled correctly
// // // %apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(double * _data, int _dim_x,int _dim_y,int _dim_z)}
// // // 
// // // %apply (double* IN_ARRAY1, int DIM1) {(double* _data, int _dim_x)}
// // // %apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* _data, int _dim_x,int _dim_y)}

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


// %{
//     namespace swig {
//         template <>  struct traits<CompuCell3D::CellG> {
//             typedef pointer_category category;
//             static const char* type_name() { return"CompuCell3D::CellG"; }
//         };
//     }
// %}



// this is a bug in swig - it is looking for a trait for CompuCell3D::CellG when wrapping  %template(mapCellGPtrToFloat) std::map<CompuCell3D::CellG*,double>;
// the easies workaround is to wrap vector <CompuCell3D::CellG> which iwill produce required trait instantiation

// more on this topic http://cvblob.googlecode.com/svn-history/r361/branches/0.10.4_pythonswig/interfaces/swig/general/cvblob.i

%include <Utils/Coordinates3D.h>

%template(vectorCell) std::vector<CompuCell3D::CellG>;
%template(mapCellGPtrToFloat) std::map<CompuCell3D::CellG*,float>;

%template(Coodrinates3DFloat) Coordinates3D<float>;
%template(mapCellGPtrToCoordinates3DFloat) std::map<CompuCell3D::CellG*,Coordinates3D<float> >;



// %template(mapCellGPtrToFloat) std::map<CompuCell3D::cellGPtr_t,float>;


// %template(mapCellGPtrToFloat) std::map<CompuCell3D::CellG*,float>;
// %template(vectorCellGPtr) std::vector<CompuCell3D::CellG*>;


// // // %apply (double* IN_ARRAY1, int DIM1) {(double* input, int size)}


// // // %inline %{
// // // /*
// // // * Memory managment function used in sg::base::DataVector::__array()
// // // * it simply decrements the number of references to the PyObject datavector 
// // // * every time a referencing ndarray is deleted.
// // // * After reference counter reaches 0, the memory of DataVector will be deallocated 
// // // * automatically. 
// // // */
// // // void free_array(void* ptr, void* dv){
// // //                         double* vec = (double *) ptr;
// // //                         PyObject* datavector = (PyObject*)dv;
// // //                         Py_DECREF(datavector);
// // //                 }
// // // %}
// // // 
// // // class MyArray {
// // // public:
// // // /*
// // // * Constructor allocated memory of given size
// // // */
// // // MyArray(int size);
// // // 
// // // /*
// // // * Constructor creates MyArray object using given data
// // // */
// // // MyArray(double* input, int size);
// // // 
// // // //getters
// // // double* getData();
// // // int getSize();
// // // double getItem(int idx);
// // // 
// // // %extend {
// // //     
// // //     void init(PyObject* npArray){
// // //         PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(npArray);
// // //         $self->data=static_cast<double*>(PyArray_DATA(pyarray));
// // //         
// // //     }
// // //     
// // //     // Create a ndarray view from the MyArray data
// // //     // an alternative approach using ARGOUTVIEW will fail since it does not allow to do a proper memory management
// // //     PyObject* __array(PyObject* myarray){
// // //         //Get the data and number of entries
// // //       double *vec = $self->getData();
// // //       int n = $self->getSize();
// // // 
// // //       npy_intp dims[1] = {n};
// // //       
// // //       // Create a ndarray with data from vec
// // //       PyObject* arr = PyArray_SimpleNewFromData(1,dims, NPY_DOUBLE, vec);
// // //       
// // //       // Let the array base point on the original data, free_array is a additional destructor for our ndarray, 
// // //       // since we need to DECREF MyArray object
// // //       PyObject* base = PyCObject_FromVoidPtrAndDesc((void*)vec, (void*)myarray, free_array);
// // //       PyArray_BASE(arr) = base;
// // //       
// // //       // Increase the number of references to PyObject MyArray, after the object the variable is reinitialized or deleted the object
// // //       // will still be on the heap, if the reference counter is positive.
// // //       Py_INCREF(myarray);
// // //       
// // //       return arr;
// // //     }
// // //      %pythoncode
// // //      {
// // //         def array(self):   
// // //           print 'CALLING ARRAY\n\n\n\n'
// // //           return self.__array(self)
// // //      }
// // //   }
// // // };
// // // 


/* %include <GraphicsDataFields.h> */
/* %include <mainCC3D.h> */

/* %include <mainCC3DWrapper.h> */

//instantiate vector<int>
// %include <Potts3D/Cell.h>

%template(vectorint) std::vector<int>;
%template(vectorlong) std::vector<long>;
%template(vectorfloat) std::vector<float>;
%template(vectorstring) std::vector<std::string>;





%include <ndarray_adapter.h>

%template(NdarrayAdapterDouble3) NdarrayAdapter<float,3>; //for storing scalar fieldas
%template(NdarrayAdapterDouble4) NdarrayAdapter<float,4>; //for storing vector fieldas
// %template(NdarrayAdapterDouble1) NdarrayAdapter<double,1>;
// %template(NdarrayAdapterDouble1) NdarrayAdapter<double,1>;
// // // %template(getItemTraitsDouble3) getItemTraits<double,3>;
// %template(getItemTraitsDouble1) getItemTraits<double,1>;


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
        
        
// //         $self->dim_x=PyArray_DIM(pyarray,0);
// //         $self->dim_y=PyArray_DIM(pyarray,1);
// //         $self->dim_z=PyArray_DIM(pyarray,2);
//         
//         cerr<<" dim_x="<<$self->dim_x<<endl;
//         cerr<<" dim_y="<<$self->dim_y<<endl;
//         cerr<<" dim_z="<<$self->dim_z<<endl;
//         
//         $self->strides[0]=$self->dim_z*$self->dim_y;
//         $self->strides[1]=$self->dim_z;
//         $self->strides[2]=1;
//         
//         cerr<<"THIS IS NUMBER OF DIMENSION FOR NUMPY ARRAY="<<ndim<<endl;
//         $self->data=static_cast<double*>(PyArray_DATA(pyarray));
        $self->setData(static_cast<float*>(PyArray_DATA(pyarray)));
  }
      
  float getItem(const std::vector<long> & _coord){
//       return 0.0;  
//       return $self->operator [] (_coord[0])[_coord[1]][_coord[2]];
      return (*($self))[_coord[0]][_coord[1]][_coord[2]];
  }

//       void clear(){
//         
//             
//         float * data=static_cast<float*>($self->data);
//             data[i]=0.0;
//         for (int i = 0; i < $self->shape[0]*$self->shape[1]*$self->shape[2];++i){
//         }
//         
//     }

      
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
        
        
// //         $self->dim_x=PyArray_DIM(pyarray,0);
// //         $self->dim_y=PyArray_DIM(pyarray,1);
// //         $self->dim_z=PyArray_DIM(pyarray,2);
//         
//         cerr<<" dim_x="<<$self->dim_x<<endl;
//         cerr<<" dim_y="<<$self->dim_y<<endl;
//         cerr<<" dim_z="<<$self->dim_z<<endl;
//         
//         $self->strides[0]=$self->dim_z*$self->dim_y;
//         $self->strides[1]=$self->dim_z;
//         $self->strides[2]=1;
//         
//         cerr<<"THIS IS NUMBER OF DIMENSION FOR NUMPY ARRAY="<<ndim<<endl;
//         $self->data=static_cast<double*>(PyArray_DATA(pyarray));
        $self->setData(static_cast<float*>(PyArray_DATA(pyarray)));
  }
      
  float getItem(const std::vector<long> & _coord){
//       return 0.0;  
//       return $self->operator [] (_coord[0])[_coord[1]][_coord[2]];
      return (*($self))[_coord[0]][_coord[1]][_coord[2]][_coord[3]];
  }
  
//     void clear(){
//         
//             
//         float * data=static_cast<float*>($self->data);
//         for (int i = 0; i < $self->shape[0]*$self->shape[1]*$self->shape[2]*$self->shape[3];++i){
//             data[i]=0.0;
//         }
//         
//     }
  
  
      
};


%include <FieldStorage.h>
%include <FieldExtractorBase.h>
%include <FieldExtractor.h>
%include <FieldExtractorCML.h>
%include <FieldWriter.h>

// // // %extend CompuCell3D::FloatField3D{
// // //   void initFromNumpy(PyObject *_numpyArrayObj){
// // //         PyArrayObject * pyarray=reinterpret_cast<PyArrayObject*>(_numpyArrayObj);
// // //         int ndim=PyArray_NDIM(pyarray);
// // //         if (ndim!=3){         
// // //             throw std::runtime_error(std::string("FloatField3D")+std::string(": Error: Array dimension shuold be 3"));
// // //         }
// // //         
// // //         $self->dim_x=PyArray_DIM(pyarray,0);
// // //         $self->dim_y=PyArray_DIM(pyarray,1);
// // //         $self->dim_z=PyArray_DIM(pyarray,2);
// // //         
// // //         cerr<<" dim_x="<<$self->dim_x<<endl;
// // //         cerr<<" dim_y="<<$self->dim_y<<endl;
// // //         cerr<<" dim_z="<<$self->dim_z<<endl;
// // //         
// // //         $self->strides[0]=$self->dim_z*$self->dim_y;
// // //         
// // //         $self->strides[1]=$self->dim_z;
// // //         $self->strides[2]=1;
// // //         
// // //         cerr<<"THIS IS NUMBER OF DIMENSION FOR NUMPY ARRAY="<<ndim<<endl;
// // //         $self->data=static_cast<double*>(PyArray_DATA(pyarray));
// // //       
// // //   }
// // //   
// // // };



%extend CompuCell3D::ScalarFieldCellLevel{    


  void __setitem__(CompuCell3D::CellG * _cell,float _val) {
      (*($self))[_cell]=_val;
//       $self->insert(std::make_pair(_cell,_val));
  }
  
  float __getitem__(CompuCell3D::CellG * _cell) {
      return (*($self))[_cell]; //this has side efect that if the _cell is not in the map it will be inserted with  matching value 0.0
      
//       CompuCell3D::ScalarFieldCellLevel::container_t::iterator mitr;
//       mitr=$self->find()
//       $self->insert(std::make_pair(_cell,_val));
  }

//   void clear(){
//         $self->clear();
//   }

  
};


%extend CompuCell3D::VectorFieldCellLevel{    


//   void __setitem__(CompuCell3D::CellG * _cell,const std::vector<float> & _vec) {
//       (*($self))[_cell]=Coordinates3D<float>(_vec[0],_vec[1],_vec[2]);
// //       $self->insert(std::make_pair(_cell,_val));
//   }


  void __setitem__(CompuCell3D::CellG * _cell,PyObject *_numpyArrayObj) {
      
        if (PyList_Check(_numpyArrayObj)){//in case user passes regular python list instead of numpy array
//             cerr<<"THIS IS PY LIST"<<endl;
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
//         cerr<<"this is pyarray="<<pyarray<<endl;
        int dim=PyArray_DIM(pyarray,0);
//         cerr<<" THIS IS DIM="<<dim<<endl;
        if (PyArray_DIM(pyarray,0)!=3){         
            throw std::runtime_error(std::string("VectorFieldCellLevel")+std::string(": Error: Array dimension should be 3"));
        }
        
        float *data =static_cast<float*>(PyArray_DATA(pyarray));
      
      (*($self))[_cell]=Coordinates3D<float>(data[0],data[1],data[2]);
//       $self->insert(std::make_pair(_cell,_val));
  }
  
// %pythoncode %{
//     def __setitem__(self,_cell,_array):# we intercept assignments of the array and wrap any array object in numpy array and pass it to C++ fcn.
//         import numpy
//         self.setitem(_cell,numpy.array(_array,dtype=numpy.float32))
// %}  

  
  PyObject* __getitem__(CompuCell3D::CellG * _cell) {
      Coordinates3D<float> &vec=(*($self))[_cell];
//     cerr<<"x,y,z="<<vec.x<<","<<vec.y<<","<<vec.z<<endl;  
    npy_intp dim=3;
    
    PyObject* numpyArray= PyArray_SimpleNew(1,&dim,NPY_FLOAT32);
    
    float *data =static_cast<float*>(PyArray_DATA(numpyArray));
    data[0]=vec.x;
    data[1]=vec.y;
    data[2]=vec.z;
    
    return numpyArray;
  }
};


//%include <CompuCell3D/Field3D/Point3D.h>
//%include <CompuCell3D/Field3D/Dim3D.h>

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
    
// // //    void fillScalarValue(CompuCell3D::FieldStorage::floatField3D_t * _field, int _x, int _y, int _z, float _value){
// // //       (*_field)[_x][_y][_z]=_value;
// // //    }

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
        
//          (*_field)[_xPos][_yPos][_zPos]=Coordinates3D<float>(_x,_y,_z);
    }   
   
//     void insertVectorIntoVectorField(CompuCell3D::FieldStorage::vectorField3D_t * _field,int _xPos, int _yPos, int _zPos, float _x, float _y, float _z){
//          (*_field)[_xPos][_yPos][_zPos]=Coordinates3D<float>(_x,_y,_z);
//     }   
    
   void insertVectorIntoVectorCellLevelField(CompuCell3D::FieldStorage::vectorFieldCellLevel_t * _field,CompuCell3D::CellG* _cell, float _x, float _y, float _z){
       
      _field->insert(std::make_pair(_cell,Coordinates3D<float>(_x,_y,_z)));
   }

   void clearVectorCellLevelField(CompuCell3D::FieldStorage::vectorFieldCellLevel_t * _field){
      _field->clear();
   }
   
// // //    void clearScalarField(CompuCell3D::Dim3D _dim, CompuCell3D::FieldStorage::floatField3D_t * _fieldPtr){
// // // 	 
// // //          for (int x=0;x<_dim.x;++x)
// // //             for (int y=0;y<_dim.y;++y)
// // //                 for (int z=0;z<_dim.z;++z){
// // //                     (*_fieldPtr)[x][y][z]=0.0;
// // //                 }
// // //     }

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

// %inline %{
// //extern SimthreadBase * getSimthreadBasePtr();
// //extern SimthreadBase *simthreadBasePtr;
// //extern double numberGlobal;
// //extern double getNumberGlobal();

// %}

// //%include <simthreadAccessor.h>
// %include <PyScriptRunnerObject.h> 

// //setting up interface from Coordinates3D.h
// %include <Coordinates3D.h>
// %template (coordinates3Dfloat) Coordinates3D<float>;

// %inline %{
	// SimthreadBase * getSimthread(int simthreadIntPtr){
		// return (SimthreadBase *)simthreadIntPtr;
	// }
	
// %}

// %inline %{

   // void fillScalarValue(GraphicsDataFields::floatField3D_t * _field, int _x, int _y, int _z, float _value){
      // (*_field)[_x][_y][_z]=_value;
   // }

   // void insertVectorIntoVectorCellLevelField(GraphicsDataFields::vectorFieldCellLevel_t * _field,CompuCell3D::CellG* _cell, float _x, float _y, float _z){
      // _field->insert(std::make_pair(_cell,Coordinates3D<float>(_x,_y,_z)));
   // }

   // void clearVectorCellLevelField(GraphicsDataFields::vectorFieldCellLevel_t * _field){
      // _field->clear();
   // }

   // Coordinates3D<float> * findVectorInVectorCellLEvelField(GraphicsDataFields::vectorFieldCellLevel_t * _field,CompuCell3D::CellG* _cell){
      // GraphicsDataFields::vectorFieldCellLevelItr_t vitr;
      // vitr=_field->find(_cell);
      // if(vitr != _field->end()){
         // return & vitr->second;
      // }else{

         // return 0;
      // }
      


   // }


// %}
