#include "FieldStorage.h"
#include <iostream>

using namespace std;
using namespace CompuCell3D;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::FieldStorage(){

}

void FieldStorage::setDim(Dim3D _dim){fieldDim=_dim;}

Dim3D FieldStorage::getDim(){return fieldDim;}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 void FieldStorage::clearAllocatedFields(){

    // field3DGraphicsData.clear();
    // field3DConcentration.clear();
   std::map<std::string,floatField3D_t*>::iterator mitr;
   
    for( mitr = floatField3DNameMap.begin(); mitr != floatField3DNameMap.end() ; ++mitr){
       if(mitr->second)
          delete mitr->second;
    }
   
    floatField3DNameMap.clear();

    
    for (map<std::string, vectorFieldCellLevel_t * >::iterator vitr=vectorFieldCellLevelNameMap.begin() ; vitr!=vectorFieldCellLevelNameMap.end() ; ++vitr){
       if(vitr->second)
          delete vitr->second;
    }

    vectorFieldCellLevelNameMap.clear();

    for (map<std::string, scalarFieldCellLevel_t * >::iterator vitr=scalarFieldCellLevelNameMap.begin() ; vitr!=scalarFieldCellLevelNameMap.end() ; ++vitr){
       if(vitr->second)
          delete vitr->second;
    }

    scalarFieldCellLevelNameMap.clear();

 }



FieldStorage::~FieldStorage(){

   clearAllocatedFields();

}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::allocateCellField(Dim3D _dim){
    fieldDim=_dim;   
    field3DGraphicsData.assign(fieldDim.x,vector<vector<CellGraphicsData> >(fieldDim.y,vector<CellGraphicsData>(fieldDim.z)));
}

//Pixel based scalar field
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::floatField3D_t* FieldStorage::createFloatFieldPy(Dim3D _dim,std::string _fieldName){
    NdarrayAdapter<float,3> * arrayPtr=new NdarrayAdapter<float,3>;
    vector<long> strides(3,1);
    vector<long> shape(3,0);
    
    strides[0]=_dim.z*_dim.y;
    strides[1]=_dim.z;
    strides[2]=1;
    
    shape[0]=_dim.x;
    shape[1]=_dim.y;
    shape[2]=_dim.z;
    
    
    arrayPtr->setStrides(strides);
    arrayPtr->setShape(shape);
    
    floatField3DNameMap.insert(std::make_pair(_fieldName,arrayPtr));
    
    
    return arrayPtr;        
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearScalarField(Dim3D _dim, FieldStorage::floatField3D_t * _fieldPtr){
        
    for (int x=0;x<_dim.x;++x)
        for (int y=0;y<_dim.y;++y)
            for (int z=0;z<_dim.z;++z){
                (*_fieldPtr)[x][y][z]=0.0;
            }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getScalarFieldNameVector(){
    vector<string> fieldNameVec;
    floatField3DNameMapItr_t mitr;
    for (mitr=floatField3DNameMap.begin()  ; mitr !=floatField3DNameMap.end() ; ++mitr){
        fieldNameVec.push_back(mitr->first);	
    }
    return fieldNameVec;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::floatField3D_t * FieldStorage::getScalarFieldByName(std::string _fieldName){

    floatField3DNameMapItr_t mitr=floatField3DNameMap.find(_fieldName);
    if(mitr!=floatField3DNameMap.end())
        return mitr->second;
    else
        return 0;
}

//Pixel based vector field
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorField3D_t * FieldStorage::createVectorFieldPy(Dim3D _dim , std::string _fieldName){
    

    vectorField3D_t * arrayPtr=new vectorField3D_t;
    vector<long> strides(4,1);
    vector<long> shape(4,0);
    
    strides[0]=3*_dim.z*_dim.y;
    strides[1]=3*_dim.z;
    strides[2]=3*1;
    strides[3]=1;
    
    shape[0]=_dim.x;
    shape[1]=_dim.y;
    shape[2]=_dim.z;
    shape[3]=3;
    
    arrayPtr->setStrides(strides);
    arrayPtr->setShape(shape);
    
    vectorFieldNameMap.insert(std::make_pair(_fieldName,arrayPtr));
    
    
    return arrayPtr;


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearVectorField(Dim3D _dim, FieldStorage::vectorField3D_t * _fieldPtr){
         
         for (int x=0;x<_dim.x;++x)
                for (int y=0;y<_dim.y;++y)
                        for (int z=0;z<_dim.z;++z)
                            for (int i=0;i<3;++i){
                                (*_fieldPtr)[x][y][z][i]=0.0;
                        }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getVectorFieldNameVector(){
        vector<string> fieldNameVec;
        vectorFieldNameMapItr_t mitr;
        for (mitr=vectorFieldNameMap.begin()  ; mitr !=vectorFieldNameMap.end() ; ++mitr){
                fieldNameVec.push_back(mitr->first);    
        }
        return fieldNameVec;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorField3D_t * FieldStorage::getVectorFieldFieldByName(std::string _fieldName){
        vectorFieldNameMapItr_t mitr=vectorFieldNameMap.find(_fieldName);
        if(mitr!=vectorFieldNameMap.end())
                return mitr->second;
        else
                return 0;
}

//Scalar field cell Level

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::scalarFieldCellLevel_t * FieldStorage::createScalarFieldCellLevelPy(std::string _fieldName){
    scalarFieldCellLevel_t * fieldPtr = new scalarFieldCellLevel_t;
    scalarFieldCellLevelNameMap.insert(make_pair(_fieldName,fieldPtr));
    return fieldPtr ;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearScalarCellLevelField(FieldStorage::scalarFieldCellLevel_t * _field){
        _field->clear();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getScalarFieldCellLevelNameVector(){
        vector<string> fieldNameVec;
        scalarFieldCellLevelNameMapItr_t mitr;
        for (mitr=scalarFieldCellLevelNameMap.begin()  ; mitr !=scalarFieldCellLevelNameMap.end() ; ++mitr){
                fieldNameVec.push_back(mitr->first);    
        }
        return fieldNameVec;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::scalarFieldCellLevel_t * FieldStorage::getScalarFieldCellLevelFieldByName(std::string _fieldName){
        scalarFieldCellLevelNameMapItr_t mitr=scalarFieldCellLevelNameMap.find(_fieldName);
        if(mitr!=scalarFieldCellLevelNameMap.end())
                return mitr->second;
        else
                return 0;
}

//Vector field cell Level

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorFieldCellLevel_t * FieldStorage::createVectorFieldCellLevelPy(std::string _fieldName){
    vectorFieldCellLevel_t * fieldPtr = new vectorFieldCellLevel_t;
    vectorFieldCellLevelNameMap.insert(make_pair(_fieldName,fieldPtr));
    return fieldPtr ;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FieldStorage::clearVectorCellLevelField(FieldStorage::vectorFieldCellLevel_t * _field){
    _field->clear();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> FieldStorage::getVectorFieldCellLevelNameVector(){
    vector<string> fieldNameVec;
    vectorFieldCellLevelNameMapItr_t mitr;
    for (mitr=vectorFieldCellLevelNameMap.begin()  ; mitr !=vectorFieldCellLevelNameMap.end() ; ++mitr){
            fieldNameVec.push_back(mitr->first);	
    }
    return fieldNameVec;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FieldStorage::vectorFieldCellLevel_t * FieldStorage::getVectorFieldCellLevelFieldByName(std::string _fieldName){
    vectorFieldCellLevelNameMapItr_t mitr=vectorFieldCellLevelNameMap.find(_fieldName);
    if(mitr!=vectorFieldCellLevelNameMap.end())
        return mitr->second;
    else
        return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





