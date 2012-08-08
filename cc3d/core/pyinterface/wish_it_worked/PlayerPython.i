

// Module Name
// %module("threads"=1) PlayerPython
%module PlayerPython
// #define SwigPyIterator PlayerPython_SwigPyIterator
%{
#define SwigPyIterator PlayerPython_SwigPyIterator
%}
// %rename (PlayerPython_SwigPyIterator) SwigPyIterator;

// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.


%{

#include <GraphicsData.h>
#include <FieldStorage.h>
#include <FieldExtractorBase.h>
#include <FieldExtractor.h>
#include <FieldExtractorCML.h>
#include <FieldWriter.h>
//#include <CompuCell3D/Field3D/Point3D.h>
//#include <CompuCell3D/Field3D/Dim3D.h>
#include <vtkIntArray.h>
    
#define FIELDEXTRACTOR_EXPORT

   

// System Libraries
#include <iostream>
#include <stdlib.h>
#include <Coordinates3D.h>



   
// Namespaces
using namespace std;
using namespace CompuCell3D;

%}

#define FIELDEXTRACTOR_EXPORT


// %ignore SwigPyIterator;
// %rename (SwigPyIteratorPlayer) SwigPyIterator;


// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

// Pointer handling
%include "cpointer.i"


/* %include <GraphicsDataFields.h> */
/* %include <mainCC3D.h> */

/* %include <mainCC3DWrapper.h> */

//instantiate vector<int>
%template(vectorint) std::vector<int>;
%template(vectorstring) std::vector<std::string>;



%include <FieldStorage.h>
%include <FieldExtractorBase.h>
%include <FieldExtractor.h>
%include <FieldExtractorCML.h>
%include <FieldWriter.h>

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

   void fillScalarValue(CompuCell3D::FieldStorage::floatField3D_t * _field, int _x, int _y, int _z, float _value){
      (*_field)[_x][_y][_z]=_value;
   }

   void clearScalarValueCellLevel(CompuCell3D::FieldStorage::scalarFieldCellLevel_t * _field){ 
		_field->clear();
   }     
   	
   void fillScalarValueCellLevel(CompuCell3D::FieldStorage::scalarFieldCellLevel_t * _field, CompuCell3D::CellG* _cell, float _value){
      _field->insert(std::make_pair(_cell,_value));
   }
   
    void insertVectorIntoVectorField(CompuCell3D::FieldStorage::vectorField3D_t * _field,int _xPos, int _yPos, int _zPos, float _x, float _y, float _z){
         (*_field)[_xPos][_yPos][_zPos]=Coordinates3D<float>(_x,_y,_z);
    }   
   void insertVectorIntoVectorCellLevelField(CompuCell3D::FieldStorage::vectorFieldCellLevel_t * _field,CompuCell3D::CellG* _cell, float _x, float _y, float _z){
      _field->insert(std::make_pair(_cell,Coordinates3D<float>(_x,_y,_z)));
   }

   void clearVectorCellLevelField(CompuCell3D::FieldStorage::vectorFieldCellLevel_t * _field){
      _field->clear();
   }
   
   void clearScalarField(CompuCell3D::Dim3D _dim, CompuCell3D::FieldStorage::floatField3D_t * _fieldPtr){
	 
         for (int x=0;x<_dim.x;++x)
            for (int y=0;y<_dim.y;++y)
                for (int z=0;z<_dim.z;++z){
                    (*_fieldPtr)[x][y][z]=0.0;
                }
    }

    void clearVectorField(CompuCell3D::Dim3D _dim, CompuCell3D::FieldStorage::vectorField3D_t * _fieldPtr){
        
        for (int x=0;x<_dim.x;++x)
            for (int y=0;y<_dim.y;++y)
                for (int z=0;z<_dim.z;++z){
                    (*_fieldPtr)[x][y][z]=Coordinates3D<float>(0.,0.,0.);
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
