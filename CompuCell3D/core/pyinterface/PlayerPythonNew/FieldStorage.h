#ifndef FIELDSTORAGE_H
#define FIELDSTORAGE_H

#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>
#include "CellGraphicsData.h"

#include <CompuCell3D/Potts3D/Cell.h>
#include "FieldExtractorDLLSpecifier.h"
#include <CompuCell3D/Field3D/Dim3D.h>

#include "ndarray_adapter.h"

namespace CompuCell3D{
	//have to declare here all the classes that will be passed to this class from Python
	class Dim3D;


    
class FIELDEXTRACTOR_EXPORT ScalarFieldCellLevel:public std::map<CompuCell3D::CellG*,float > {
public:    
    typedef std::map<CompuCell3D::CellG*,float > container_t;
};

class FIELDEXTRACTOR_EXPORT VectorFieldCellLevel:public std::map<CompuCell3D::CellG*,Coordinates3D<float> > {
public:        
    typedef std::map<CompuCell3D::CellG*,Coordinates3D<float> > container_t;
};

class FIELDEXTRACTOR_EXPORT FieldStorage{
   public://typedef's
      
      //pixel based scalar field  
      typedef NdarrayAdapter<float,3> floatField3D_t;      
      typedef std::map<std::string,floatField3D_t*>::iterator floatField3DNameMapItr_t;
      
      //pixel based vector field  
      typedef NdarrayAdapter<float,4> vectorField3D_t;                
      typedef std::map<std::string,vectorField3D_t *>::iterator vectorFieldNameMapItr_t;

      //cell level vector fields (represented as maps)
      typedef VectorFieldCellLevel vectorFieldCellLevel_t;                      
      typedef std::map<std::string,vectorFieldCellLevel_t *>::iterator vectorFieldCellLevelNameMapItr_t;
      typedef vectorFieldCellLevel_t::iterator vectorFieldCellLevelItr_t;

      //cell level scalar fields (represented as maps)      
      typedef ScalarFieldCellLevel scalarFieldCellLevel_t;      
      typedef std::map<std::string,scalarFieldCellLevel_t *>::iterator scalarFieldCellLevelNameMapItr_t;
      typedef scalarFieldCellLevel_t::iterator scalarFieldCellLevelItr_t;


      typedef std::vector<std::vector<std::vector<CellGraphicsData> > > field3DGraphicsData_t;

      // typedef std::map<std::string,std::string> plotNamePlotTypeMap_t;
      //The above map is used to mark what type a give plot is of For example cAMP will be of type "scalar"
      //The allowed key_words are
      //cell_field - for plots of te cell types
      //scalar - for concentration plots
      //vector_cell_level - for vector plots where vector is an attribute of a cell
      //vector_pixel_level - for vector plots where vector is an attribute of a pixel
   public:

    FieldStorage();
    ~FieldStorage();

    void setDim(Dim3D _dim);
    Dim3D getDim();

    void allocateCellField(Dim3D _dim);

                //numpy arrays thin wrapper
      
      
    // pixel based scalar field a.k.a. concentration field
    floatField3D_t* createFloatFieldPy(Dim3D _dim,std::string _fieldName);
    void clearScalarField(Dim3D _dim,floatField3D_t * _fieldPtr);
    std::vector<std::string> getScalarFieldNameVector();
    floatField3D_t * getScalarFieldByName(std::string _fieldName);

    //pixel based  vector fields 
    vectorField3D_t * createVectorFieldPy(Dim3D _dim,std::string _fieldName);
    void clearVectorField(Dim3D _dim,vectorField3D_t * _fieldPtr);
    std::vector<std::string> getVectorFieldNameVector();
    vectorField3D_t * getVectorFieldFieldByName(std::string _fieldName);

    //vector fields cell level
    vectorFieldCellLevel_t * createVectorFieldCellLevelPy(std::string _fieldName);
    void clearVectorCellLevelField(vectorFieldCellLevel_t * _field);
    std::vector<std::string> getVectorFieldCellLevelNameVector();
    vectorFieldCellLevel_t * getVectorFieldCellLevelFieldByName(std::string _fieldName);



    //scalar fields cell level
    scalarFieldCellLevel_t * createScalarFieldCellLevelPy(std::string _fieldName);
    void clearScalarCellLevelField(scalarFieldCellLevel_t * _field);
    std::vector<std::string> getScalarFieldCellLevelNameVector();
    scalarFieldCellLevel_t * getScalarFieldCellLevelFieldByName(std::string _fieldName);
    
    void clearAllocatedFields();
    
    std::vector<std::vector<std::vector<CellGraphicsData> > > field3DGraphicsData;/// 3d field
    
   private:
       
    Dim3D fieldDim;
      
    std::map<std::string,floatField3D_t*> floatField3DNameMap; ///this map will be filled externally
    std::map<std::string, vectorField3D_t*> vectorFieldNameMap;
    std::map<std::string, vectorFieldCellLevel_t * > vectorFieldCellLevelNameMap;///this map will be filled externally
    std::map<std::string, scalarFieldCellLevel_t * > scalarFieldCellLevelNameMap;///this map will be filled externally
    
};


};



#endif
