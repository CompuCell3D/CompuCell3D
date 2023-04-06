#include "CellVelocityPlugin.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/plugins/CenterOfMass/CenterOfMassPlugin.h>
#include <Utils/cldeque.h>
#include <CompuCell3D/plugins/CellVelocity/CellVelocityData.h>
#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include "CellVelocityDataAccessor.h"
#include <iostream>

using namespace std;

namespace CompuCell3D {
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CellVelocityPlugin::CellVelocityPlugin()
 : Plugin(),cellVelocityDataAccessorPtr(0)
 //, CellGGChangeWatcher()
{
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CellVelocityPlugin::~CellVelocityPlugin()
{
   if(cellVelocityDataAccessorPtr){
      delete cellVelocityDataAccessorPtr;
      cellVelocityDataAccessorPtr=0;
   }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocityPlugin::init(Simulator *_simulator){
   

    Potts3D *potts = _simulator->getPotts();
    simulator=_simulator;
   
/*   cellVelocityDataAccessorPtr=new CellVelocityDataAccessor<CellVelocityData>(cldequeCapacity,enoughDataThreshold) ;
   potts->getCellFactoryGroupPtr()->registerClass(cellVelocityDataAccessorPtr);*/
   
/*   potts->registerCellGChangeWatcher(this);
   COMPlugin = (CenterOfMassPlugin *)Simulator::pluginManager.get("CenterOfMass");
   watchingAllowed=false;*/
   

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocityPlugin::extraInit(Simulator *_simulator){
   

    Potts3D *potts = _simulator->getPotts();

   cellVelocityDataAccessorPtr=new CellVelocityDataAccessor<CellVelocityData>(cldequeCapacity,enoughDataThreshold) ;
   potts->getCellFactoryGroupPtr()->registerClass(cellVelocityDataAccessorPtr);
    
//    potts->registerCellGChangeWatcher(this);
//    COMPlugin = (CenterOfMassPlugin *)Simulator::pluginManager.get("CenterOfMass");
   //watchingAllowed=false;
   

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void CellVelocityPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {
//    
//    // vel = COM(t_n) - COM(t_{n-1})
//    CC3D_Log(LOG_TRACE) << "********CELL VELOCITY FIELD CHANGE";
//    
//    if(!simulator)
//       return;
// 
//    if(simulator->getStep() < 1)
//       return;
//  
//    if(oldCell){
// 
//        if(oldCell->volume  > 0 ){
   //                 CC3D_Log(LOG_TRACE) << "cell velocity oldCell->volume="<<oldCell->volume;
//             CC3D_Log(LOG_TRACE) << " old x,y,z,CM="<<oldCell->xCM<<" "<<oldCell->yCM<<" "<<oldCell->zCM<<" ";
// 
//          cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr)->push_front  (
//                                            (oldCell->xCM)/(float)(oldCell->volume) -(oldCell->xCM+pt.x)/((float)oldCell->volume+1)   ,
//                                            (oldCell->yCM)/(float)(oldCell->volume) -(oldCell->yCM+pt.y)/((float)oldCell->volume+1)   ,
//                                            (oldCell->zCM)/(float)(oldCell->volume) -(oldCell->zCM+pt.z)/((float)oldCell->volume+1)
//                                         );
//        }
            // CC3D_Log(LOG_TRACE) << " old Cell velocity="<<cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr)->getInstantenousVelocity().X()
//        <<cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr)->getInstantenousVelocity().Y()
//        <<cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr)->getInstantenousVelocity().Z();
//        <<endl;*/
//     }
//    
//    if(newCell){
//             CC3D_Log(LOG_TRACE) << "cell velocity newCell->volume="<<newCell->volume;
//          CC3D_Log(LOG_TRACE) << " new x,y,z,CM="<<newCell->xCM<<" "<<newCell->yCM<<" "<<newCell->zCM<<" ";
//          if(newCell->volume  > 1 ){
//             cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr)->push_front  (
//                                              (newCell->xCM)/(float)(newCell->volume) -(newCell->xCM-pt.x)/((float)newCell->volume-1)   ,
//                                              (newCell->yCM)/(float)(newCell->volume) -(newCell->yCM-pt.y)/((float)newCell->volume-1)   ,
//                                              (newCell->zCM)/(float)(newCell->volume) -(newCell->zCM-pt.z)/((float)newCell->volume-1)
//                                           );
//          
//          }
               // CC3D_Log(LOG_TRACE) << " new Cell velocity="<<cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr)->getInstantenousVelocity().X()
// //          <<cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr)->getInstantenousVelocity().Y()
// //          <<cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr)->getInstantenousVelocity().Z();
// //          <<endl;
//    }
// 
//    
//               
// }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellVelocityPlugin::readXML(XMLPullParser &in){
  in.skip(TEXT);
  unsigned int size=2;
  unsigned int enoughData=2;
  
  while (in.check(START_ELEMENT)) {
    if (in.getName() == "VelocityDataHistorySize") {
      size = BasicString::parseUInteger(in.matchSimple());

    } else if (in.getName() == "EnoughDataThreshold") {
       enoughData= BasicString::parseUInteger(in.matchSimple());

    } else {
      throw BasicException(string("Unexpected element '") + in.getName() + 
            "'!", in.getLocation());
    }

    in.skip(TEXT);
  }  

  cldequeCapacity=size;
  enoughDataThreshold=enoughData;

  ASSERT_OR_THROW("capacity must be at least 2 " , cldequeCapacity >= 2 );
  ASSERT_OR_THROW("capacity must be >= enoughDataThreshold " , cldequeCapacity >= enoughDataThreshold );
  
  
}

};
