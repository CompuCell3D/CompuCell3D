
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
#include <stddef.h>
%}



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



%include "Fields_pre.i"

%{
    #include <numpy/arrayobject.h>
%}


%include "swig_includes/numpy.i"

%init %{
    import_array();
%}

%include <Utils/Coordinates3D.h>

%template (Coordinates3DDouble) Coordinates3D<double>; 




// %include <CompuCell3D/Boundary/BoundaryStrategy.h>

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



%feature("compactdefaultargs"); 
%include "typemaps_Fields.i"

%typemap(in) Coordinates3D<double>  (Coordinates3D<double> coord)  {
  /* Check if is a list */
  cerr<<"inside Coordinates3D<double> conversion typemap"<<endl;
    if (PyList_Check($input)) {
        int size = PyList_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            coord.x= (double)PyFloat_AsDouble(PyList_GetItem($input,0));
            coord.y=(double)PyFloat_AsDouble(PyList_GetItem($input,1));
            coord.z=(double)PyFloat_AsDouble(PyList_GetItem($input,2));
            $1=coord;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list/numpy array of 3 double values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        }

    }else if (PyTuple_Check($input)){
        //check if it is a tuple
        int size = PyTuple_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            coord.x= (double)PyFloat_AsDouble(PyTuple_GetItem($input,0));
            coord.y=(double)PyFloat_AsDouble(PyTuple_GetItem($input,1));
            coord.z=(double)PyFloat_AsDouble(PyTuple_GetItem($input,2));
            $1=coord;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list/numpy array of 3 double values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        }                
    }else if(PyArray_Check($input)){
    
        if (PyArray_NDIM($input)!=1 || PyArray_DIM($input,0)!=3){ // checking if the argument is a vector with 3 values
            SWIG_exception(SWIG_ValueError,"Expected a list/numpy array of 3 double values e.g. [12,31,48].");
        }
        
        if (PyArray_ISFLOAT ($input)){
            double * arrayContainerPtr=(double *) PyArray_DATA($input);
            coord.x=arrayContainerPtr[0];
            coord.y=arrayContainerPtr[1];
            coord.z=arrayContainerPtr[2];
        
            
        }  else  if (PyArray_ISINTEGER ($input)){

            int * arrayContainerPtr=(int *) PyArray_DATA($input);
            coord.x=arrayContainerPtr[0];
            coord.y=arrayContainerPtr[1];
            coord.z=arrayContainerPtr[2];

        
        }else{
            SWIG_exception(SWIG_ValueError,"The values in the array should be either floating point numbers or inttegers. Please use explicit type conversion for all the values");
        }
                
        
        
        $1=coord;        
    }
    else{
        
         int res = SWIG_ConvertPtr($input,(void **) &$1, $&1_descriptor,0);
         
         
        if (SWIG_IsOK(res)) {
            //Notice that we have to call PyDECREF on return value from PyObject_GetAttrString otherwise we leak memory!    
            PyObject * xPyRef=PyObject_GetAttrString($input,"x"); coord.x=(double)PyFloat_AsDouble(xPyRef); Py_DECREF(xPyRef);
            PyObject * yPyRef=PyObject_GetAttrString($input,"y"); coord.y=(double)PyFloat_AsDouble(yPyRef); Py_DECREF(yPyRef);
            PyObject * zPyRef=PyObject_GetAttrString($input,"z"); coord.z=(double)PyFloat_AsDouble(zPyRef); Py_DECREF(zPyRef);    
            $1=coord;
        } else {
        
            SWIG_exception(SWIG_ValueError,"Expected CompuCell.Coordinates3DDouble object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
                      
        }
         
    }
}


%typemap(in) Coordinates3D<double> &  (Coordinates3D<double> coord)  { // note that (CompuCell3D::Point3D pt) causes pt to be allocated on the stack - no need to worry abuot freeing memory
  /* Check if is a list */  
    if (PyList_Check($input)) {
        int size = PyList_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            coord.x= (double)PyFloat_AsDouble(PyList_GetItem($input,0));
            coord.y=(double)PyFloat_AsDouble(PyList_GetItem($input,1));
            coord.z=(double)PyFloat_AsDouble(PyList_GetItem($input,2));
            $1=&coord;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 double values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case

        }

    }else if (PyTuple_Check($input)){
        //check if it is a tuple
        int size = PyTuple_Size($input);        
        if (size==3){
            // CompuCell3D::Point3D pt;    
            coord.x= (double)PyFloat_AsDouble(PyTuple_GetItem($input,0));
            coord.y=(double)PyFloat_AsDouble(PyTuple_GetItem($input,1));
            coord.z=(double)PyFloat_AsDouble(PyTuple_GetItem($input,2));
            $1=&coord;
        }else{
            SWIG_exception(SWIG_ValueError,"Expected a list of 3 double values e.g. [12,31,48]."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
        }                
    }else if(PyArray_Check($input)){
    
        if (PyArray_NDIM($input)!=1 || PyArray_DIM($input,0)!=3){ // checking if the argument is a vector with 3 values
            SWIG_exception(SWIG_ValueError,"Expected a list/numpy array of 3 double values e.g. [12,31,48].");
        }
        
        if (! PyArray_ISFLOAT ($input)){
            SWIG_exception(SWIG_ValueError,"The values in the array appear not to be floating point numbers. Please use explicit casting to double for all the values");
        }        
        
        double * arrayContainerPtr=(double *) PyArray_DATA($input);
        coord.x=arrayContainerPtr[0];
        coord.y=arrayContainerPtr[1];
        coord.z=arrayContainerPtr[2];
        
        
        $1=&coord;        
    }else{
        
         int res = SWIG_ConvertPtr($input,(void **) &$1, $1_descriptor,0);
         
         
        if (SWIG_IsOK(res)) {
            
            //Notice that we have to call PyDECREF on return value from PyObject_GetAttrString otherwise we leak memory!    
            PyObject * xPyRef=PyObject_GetAttrString($input,"x"); coord.x=(double)PyFloat_AsDouble(xPyRef); Py_DECREF(xPyRef);
            PyObject * yPyRef=PyObject_GetAttrString($input,"y"); coord.y=(double)PyFloat_AsDouble(yPyRef); Py_DECREF(yPyRef);
            PyObject * zPyRef=PyObject_GetAttrString($input,"z"); coord.z=(double)PyFloat_AsDouble(zPyRef); Py_DECREF(zPyRef);    

            $1=&coord;
        } else {
        
            SWIG_exception(SWIG_ValueError,"Expected CompuCell.Coordinates3DDouble object."); //have to use SWIG_exception to throw exception from typemap - simple throw seems not to work in this case
                      
        }
         
    }
}





%inline %{

    
    
class BS{
    public:
    BS(void){
        cerr<<"BS constructor"<<endl;
    }
    
    float get(CompuCell3D::Point3D &  pt){
        cerr<<"got new get request"<<endl;
        return 0.0;
    }

    float getNonRef(CompuCell3D::Point3D   pt){
        cerr<<"got new get Non ref request"<<endl;
        return 0.0;
    }

    float getByDim(CompuCell3D::Dim3D & dim){
        cerr<<"got ByDim new get request"<<endl;
        return 0.0;
    }

    float getByDimNonRef(CompuCell3D::Dim3D dim){
        cerr<<"got ByDim new get Non ref request"<<endl;
        return 0.0;
    }

    float getByCoord(Coordinates3D<double> & coord){
        cerr<<"got ByCoord new get request"<<endl;
        return 0.0;
    }

    float getByCoordNonRef(Coordinates3D<double> coord){
        cerr<<"got ByCoord new get Non ref request"<<endl;
        return 0.0;
    }
    
    
    // void getNeighborDirect(Point3D  pt,unsigned int   idx ,bool checkBounds=true, bool calculatePtTrans=false){
    void getNeighborDirect(CompuCell3D::Point3D &  pt,unsigned int idx =10,bool checkBounds=true, bool calculatePtTrans=false) const {
    
        cerr<<"THIS IS GET NEIGHBOR DIRECT"<<endl;
        cerr<<"pt="<<pt<<endl;        
    }
    
    
};


%}




// turns on proper handling of default arguments - only one wrapper code will get generated for a function
// alternative way could be to use typecheck maps but I had trouble with it.
// compactdefaultargs has one disadvantage - it will not with all languages e.g Java and C# 
// for more information see e.g. http://tech.groups.yahoo.com/group/swig/message/13432 


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
 
%inline %{
        void fcn(CompuCell3D::Dim3D _dim){
            cerr<<" THIS IS DIMENSION "<<_dim<<endl;
            // throw std::runtime_error(" DEMO: Wrong Syntax: Expected someting like: field[1,2,3]");
        }

%}

%template (vector_int) std::vector<int>;

%inline %{
        void fcnVec(const std::vector<int> & _vec){
            cerr<<" THIS IS VECTOR SIZE "<<_vec.size()<<endl;
        }

%}

// %typemap(in) CompuCell3D::Dim3D; //deleting a typamap

%inline %{
        void fcnDim(CompuCell3D::Dim3D _dim){
            cerr<<" THIS IS DIMENSION FCN DIM = "<<_dim<<endl;
        }

%}


%inline %{
        void buildCell(const CompuCell3D::Point3D  & _pt, long _val=11){
        // void buildCell(CompuCell3D::Point3D & _pt, long _val, bool checkBounds=true, bool calculatePtTrans=false){
        // void buildCell(const CompuCell3D::Point3D _pt){
            cerr<<" THIS IS BUILD CELLS = "<<_pt<<endl;
            cerr<<" this is value="<<_val<<endl;
        }

%}


%inline %{
        void fcnCoordinates(Coordinates3D<double> _coord){
            cerr<<" THIS IS Coordinates3D<double> = "<<_coord<<endl;
        }

%}


%inline %{
        void buildCoordinates(const Coordinates3D<double> & _coord, long _val=11){
        // void buildCell(CompuCell3D::Point3D & _pt, long _val, bool checkBounds=true, bool calculatePtTrans=false){
        // void buildCell(const CompuCell3D::Point3D _pt){
            cerr<<" buildCoordinates = "<<_coord<<endl;
            cerr<<" this is value="<<_val<<endl;
        }

%}




// %include <CompuCell3D/Field3D/Dim3D.h>


// %include <CustomSubDomains.h>

// %include <CustomExpressions.h>


