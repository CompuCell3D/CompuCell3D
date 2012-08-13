#include "CellVelocity.h"
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>
#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>

#include <CompuCell3D/Simulator.h>
#include <iostream>
#include <string>
#include <CompuCell3D/plugins/CellVelocity/CellVelocityPlugin.h>
#include <CompuCell3D/plugins/CellVelocity/CellInstantVelocityPlugin.h>
#include <CompuCell3D/plugins/CellVelocity/CellVelocityData.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <PublicUtilities/NumericalUtils.h>
#include <sstream>


using namespace std;

namespace CompuCell3D {
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CellVelocity::CellVelocity()
 : Steppable()
{
   updateFrequency=1;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CellVelocity::~CellVelocity()
{
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::init(Simulator *simulator){



   potts = simulator->getPotts();
   ///getting cell inventory
   cellInventoryPtr=& potts->getCellInventory(); 


      
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::extraInit(Simulator *simulator){

   CellVelocityPlugin * cellVelocityPlugin=0;
   cellVelocityPlugin=(CellVelocityPlugin*)(Simulator::pluginManager.get("CellVelocity"));
   
   ASSERT_OR_THROW("CellVelocity plugin not initialized!", cellVelocityPlugin);
   cellVelocityDataAccessorPtr = cellVelocityPlugin->getCellVelocityDataAccessorPtr();


   fieldDim=potts->getCellFieldG()->getDim();
   
   potts->getBoundaryXName()=="Periodic" ? boundaryConditionIndicator.x=1 : boundaryConditionIndicator.x=0 ;
   potts->getBoundaryYName()=="Periodic" ? boundaryConditionIndicator.y=1 : boundaryConditionIndicator.y=0;
   potts->getBoundaryZName()=="Periodic" ? boundaryConditionIndicator.z=1 : boundaryConditionIndicator.z=0;
   cerr<<"boundaryConditionIndicator="<<boundaryConditionIndicator<<endl;
   
   
   
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::start(){
   zeroCellVelocities();
   updateCOMList();

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::updateCOMList(){

CellInventory::cellInventoryIterator cInvItr;
///loop over all the cells in the inventory
float xCom;
float yCom;
float zCom;

Coordinates3D<float> oldCM,newCM,v;
CellG * cell;

   

   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
      
      cell=*cInvItr;
      
      newCM.XRef()=cell->xCM/(float)cell->volume ;
      newCM.YRef()=cell->yCM/(float)cell->volume;
      newCM.ZRef()=cell->zCM/(float)cell->volume;
      //cerr<<"cell->xCM/(float)cell->volume="<<cell->xCM/(float)cell->volume<<" newCM.X()="<<newCM.X()<<endl;

      oldCM=cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> getLastCM();
//       cerr<<"cell "<<cell<<"enough data="<<cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr)->enoughData<<endl;
      if(cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr)->enoughData){
         v.XRef()=findMin(newCM.X()-oldCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         v.YRef()=findMin(newCM.Y()-oldCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         v.ZRef()=findMin(newCM.Z()-oldCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );
         
         v.XRef()/=(float)updateFrequency;
         v.YRef()/=(float)updateFrequency;
         v.ZRef()/=(float)updateFrequency;
         
         cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> push_front(newCM.X() , newCM.Y() , newCM.Z());
         cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> setAverageVelocity(v);
//          cerr<<"vel="<<v<<"   oldCM="<<oldCM<<" newCM="<<newCM<<" xCM="<<cell->xCM<<endl;
      }else{
         v.XRef()=0.;
         v.YRef()=0.;
         v.ZRef()=0.;
         cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> push_front(newCM.X() , newCM.Y() , newCM.Z());
         cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> setAverageVelocity(v);
//          cerr<<"zero velocity vel="<<v<<endl;
      }

/*      cerr<<"xCom="<<xCom<<" yCom="<<yCom<<" zCom="<<zCom<<endl;
      cerr<<"cellVelocityDataAccessorPtr="<<cellVelocityDataAccessorPtr<<endl;
      cerr<<"list ptr="<<cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr)<<endl;
      cerr<<"list size="<<cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr)->size()<<endl;*/
      
      
      //cerr<<*(cldeque<Coordinates3D<float> > *)cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> cellCOMPtr;
      
   }
   
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::zeroCellVelocities(){

CellInventory::cellInventoryIterator cInvItr;
///loop over all the cells in the inventory
float xCom;
float yCom;
float zCom;

CellG * cell;


   cerr<<"ACCESSOR"<<cellVelocityDataAccessorPtr<<endl;

   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr != cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
      
      cell=*cInvItr;
      

//       cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> push_front(0.0,0.0,0.0);
//       cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> push_front(0.0,0.0,0.0);
//       cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> setAverageVelocity(0.,0.,0.);
      
   }
   
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::step(const unsigned int currentStep){

   if(!(currentStep % updateFrequency)){
      updateCOMList();
   }
   
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::resizeCellVelocityData(){

//    ASSERT_OR_THROW("CellInventory is not initialized",cellInventoryPtr);
//    
//    CellInventory::cellInventoryIterator cInvItr;
//    CellG * cell;
//    
//    for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
//       
//       cell=*cInvItr;
//       
//       cellVelocityDataAccessorPtr -> get(cell->extraAttribPtr) -> resize(numberOfCOMPoints);
//       
//    }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocity::readXML(XMLPullParser &in){
   
/*  in.skip(TEXT);
  bool resize=false;
   
  while (in.check(START_ELEMENT)) {
    string attributeName=in.getName();
    if (attributeName == "NumberOfCOMPoints") {
    
        numberOfCOMPoints = BasicString::parseUInteger(in.matchSimple());
        CellVelocityData::setCldequeCapacity(numberOfCOMPoints);
        resize=true;
      
    }else if (attributeName == "EnoughDataThreshold") {
    
      enoughDataThreshold = BasicString::parseUInteger(in.matchSimple());
      CellVelocityData::setEnoughDataThreshold(enoughDataThreshold);       
    }
    else {
      throw BasicException(string("Unexpected element '") + in.getName() +
            "'!", in.getLocation());
    }

    in.skip(TEXT);
  }
   
  
  if(resize){
      resizeCellVelocityData();
  }
*/


in.skip(TEXT);
  

  while (in.check(START_ELEMENT)) {
    string attributeName=in.getName();
    if (attributeName == "UpdateFrequency") {
    
        updateFrequency = BasicString::parseUInteger(in.matchSimple());
   }    
    else {
      throw BasicException(string("Unexpected element '") + in.getName() +
            "'!", in.getLocation());
    }

    in.skip(TEXT);
  }
   
  

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

};
