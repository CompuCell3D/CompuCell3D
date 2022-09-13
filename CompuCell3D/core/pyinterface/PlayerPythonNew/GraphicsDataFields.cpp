#include "GraphicsDataFields.h"
#include <iostream>
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
GraphicsDataFields::GraphicsDataFields(){

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphicsDataFields::clearAllocatedFields(){

    field3DGraphicsData.clear();
    field3DConcentration.clear();
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

   for (map<std::string,vectorFloatField3D_t*>::iterator vitr=vectorFloatField3DNameMap.begin() ; vitr!=vectorFloatField3DNameMap.end(); ++vitr){
      if(vitr->second)
         delete vitr->second;
      
   }
   vectorFloatField3DNameMap.clear();

   for(map<std::string,vectorCellFloatField3D_t*>::iterator vitr=vectorCellFloatField3DNameMap.begin() ; vitr!=vectorCellFloatField3DNameMap.end();++vitr){
      if(vitr->second)
         delete vitr->second;
      
   }
   vectorCellFloatField3DNameMap.clear();
   
   //clearing plotNamePLotTypeMap
   plotNamePlotTypeMap.clear();
}



GraphicsDataFields::~GraphicsDataFields(){

   clearAllocatedFields();

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphicsDataFields::insertPlotNamePlotTypePair(const std::string & _plotName , const std::string & _plotType){

plotNamePlotTypeMap.insert(make_pair(_plotName,_plotType));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string GraphicsDataFields::checkPlotType(const std::string & _plotName){

   plotNamePlotTypeMap_t::iterator mitr;
   mitr=plotNamePlotTypeMap.find(_plotName);

   if(mitr != plotNamePlotTypeMap.end()){
      return mitr->second;
   }else{
      return string("unknown");
   }
   

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphicsDataFields::allocateField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN){

   sizeL=_sizeL;
   sizeM=_sizeM;
   sizeN=_sizeN;
   field3DGraphicsData.assign(_sizeL,vector<vector<GraphicsData> >(_sizeM,vector<GraphicsData>(_sizeN)));

   field3DConcentration.assign(_sizeL,vector<vector<float> >(_sizeM,vector<float>(_sizeN)));

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphicsDataFields::allocateFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN , std::string _name){

   floatField3D_t * floatFieldPtr = new floatField3D_t ;
   
   floatFieldPtr->assign(_sizeL,vector<vector<float> >(_sizeM,vector<float>(_sizeN)));
   floatField3DNameMap.insert(std::make_pair(_name,floatFieldPtr));

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphicsDataFields::allocateVectorFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN , std::string _name){

   vectorFloatField3D_t * fieldPtr = new vectorFloatField3D_t ;
   
   fieldPtr->assign(_sizeL,vector<vector<Coordinates3D<float> > >(_sizeM,vector<Coordinates3D<float> >(_sizeN)));
   vectorFloatField3DNameMap.insert(std::make_pair(_name,fieldPtr));

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphicsDataFields::allocateVectorCellFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN , std::string _name){

   vectorCellFloatField3D_t * fieldPtr = new vectorCellFloatField3D_t ;
   
   fieldPtr->assign(_sizeL,vector<vector<pair<Coordinates3D<float>,CompuCell3D::CellG*> > >
   (_sizeM,vector<pair<Coordinates3D<float>,CompuCell3D::CellG*> >(_sizeN)));
   vectorCellFloatField3DNameMap.insert(std::make_pair(_name,fieldPtr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphicsDataFields::allocateVectorFieldCellLevel(std::string _name){

   vectorFieldCellLevel_t * fieldPtr = new vectorFieldCellLevel_t;
   vectorFieldCellLevelNameMap.insert(make_pair(_name,fieldPtr));

}