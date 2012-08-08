#include "VelocityPlugin.h"
#include <PublicUtilities/NumericalUtils.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/plugins/CenterOfMass/CenterOfMassPlugin.h>
#include <Utils/cldeque.h>
#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <iostream>

using namespace std;

namespace CompuCell3D {
double VelocityPlugin::localEnergy(const Point3D & pt){
   return 0.0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VelocityPlugin::VelocityPlugin()
 : CellGChangeWatcher(), EnergyFunction()
{
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VelocityPlugin::~VelocityPlugin()
{

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void VelocityPlugin::init(Simulator *_simulator){
   sim=_simulator;
   potts = sim->getPotts();
   fieldDim=potts->getCellFieldG()->getDim();
   Simulator::pluginManager.get("CenterOfMass");//making sure CenterOfMass Plugin is registered before this one COM plugin
   // also loads volume tracker
   

   potts->getBoundaryXName()=="Periodic" ? boundaryConditionIndicator.x=1 : boundaryConditionIndicator.x=0;
   potts->getBoundaryYName()=="Periodic" ? boundaryConditionIndicator.y=1 : boundaryConditionIndicator.y=0;
   potts->getBoundaryZName()=="Periodic" ? boundaryConditionIndicator.z=1 : boundaryConditionIndicator.z=0;


}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void VelocityPlugin::extraInit(Simulator *_simulator){

   potts->getCellFactoryGroupPtr()->registerClass(&velocityDataAccessor);
   potts->registerCellGChangeWatcher(this);
   potts->registerEnergyFunctionWithName(this,"Velocity");

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double VelocityPlugin::changeEnergy(const Point3D &pt, const CellG *newCell,const CellG *oldCell){
   
   if(sim->getStep() < 1) //protect from updating cell velocity during initialization
      return 0.0;

  // this function always returns zero as it is not really an energy function. It anly precalculates center of mass and velocity 
  // of cells after spin flip
   CenterOfMassPair_t centerOfMassPair;

   centerOfMassPair=precalculateAfterFlipCM(pt, newCell, oldCell,fieldDim,  boundaryConditionIndicator);

   if(oldCell){
      Coordinates3D<float> & beforeFlipCM = velocityDataAccessor.get(oldCell->extraAttribPtr)->beforeFlipCM;
      Coordinates3D<float> & afterFlipCM = velocityDataAccessor.get(oldCell->extraAttribPtr)->afterFlipCM;
      Coordinates3D<float> & velocity = velocityDataAccessor.get(oldCell->extraAttribPtr)->velocity;
      Coordinates3D<float> & afterFlipVelocity = velocityDataAccessor.get(oldCell->extraAttribPtr)->afterFlipVelocity;
      
      afterFlipCM=centerOfMassPair.second;

      if(oldCell->volume  > 1 ){

         afterFlipVelocity.XRef()=findMin(afterFlipCM.X()-beforeFlipCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         afterFlipVelocity.YRef()=findMin(afterFlipCM.Y()-beforeFlipCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         afterFlipVelocity.ZRef()=findMin(afterFlipCM.Z()-beforeFlipCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

      }else{// if cell is about to disappear due to a spin flip
         afterFlipCM=Coordinates3D<float>(0.,0.,0.);
         afterFlipVelocity=Coordinates3D<float>(0.,0.,0.);
      }
   }

   if (newCell){
      Coordinates3D<float> & beforeFlipCM = velocityDataAccessor.get(newCell->extraAttribPtr)->beforeFlipCM;
      Coordinates3D<float> & afterFlipCM = velocityDataAccessor.get(newCell->extraAttribPtr)->afterFlipCM;
      Coordinates3D<float> & velocity = velocityDataAccessor.get(newCell->extraAttribPtr)->velocity;
      Coordinates3D<float> & afterFlipVelocity = velocityDataAccessor.get(newCell->extraAttribPtr)->afterFlipVelocity;
      
      afterFlipCM=centerOfMassPair.first;

      afterFlipVelocity.XRef()=findMin(afterFlipCM.X()-beforeFlipCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
      afterFlipVelocity.YRef()=findMin(afterFlipCM.Y()-beforeFlipCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
      afterFlipVelocity.ZRef()=findMin(afterFlipCM.Z()-beforeFlipCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );


   }

   return 0.0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VelocityPlugin::field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell) {

   calculateVelocityData(pt, newCell,oldCell);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VelocityPlugin::calculateVelocityData(const Point3D &pt, const CellG *newCell, const CellG *oldCell){
   if(!sim)
      return;

   //may have to update CM history here
      
   if(sim->getStep() < 1) //protect from updating cell velocity during initialization
      return;


   Coordinates3D<float> currentCM;
   

   
   if(oldCell){

       if(oldCell->volume  > 0 ){
         Coordinates3D<float> & beforeFlipCM = velocityDataAccessor.get(oldCell->extraAttribPtr)->beforeFlipCM;
         Coordinates3D<float> & velocity = velocityDataAccessor.get(oldCell->extraAttribPtr)->velocity;


         currentCM.XRef()=(oldCell->xCM)/(float)(oldCell->volume);
         currentCM.YRef()=(oldCell->yCM)/(float)(oldCell->volume);
         currentCM.ZRef()=(oldCell->zCM)/(float)(oldCell->volume);

         velocity.XRef()=findMin(currentCM.X()-beforeFlipCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         velocity.YRef()=findMin(currentCM.Y()-beforeFlipCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         velocity.ZRef()=findMin(currentCM.Z()-beforeFlipCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

         //here I update beforeFlipCM - it will be valid untill next spin flip actually occurs
         beforeFlipCM=currentCM;
         
       }

    }
   
   if(newCell){

         if(newCell->volume  == 1){

            Coordinates3D<float> & beforeFlipCM = velocityDataAccessor.get(newCell->extraAttribPtr)->beforeFlipCM;
            Coordinates3D<float> & velocity = velocityDataAccessor.get(newCell->extraAttribPtr)->velocity;

            velocity.XRef()=0.0;
            velocity.YRef()=0.0;
            velocity.ZRef()=0.0;

            beforeFlipCM.XRef()=newCell->xCM;
            beforeFlipCM.YRef()=newCell->yCM;
            beforeFlipCM.ZRef()=newCell->zCM;
         }
         
         if(newCell->volume  > 1 ){
            Coordinates3D<float> & beforeFlipCM = velocityDataAccessor.get(newCell->extraAttribPtr)->beforeFlipCM;
            Coordinates3D<float> & velocity = velocityDataAccessor.get(newCell->extraAttribPtr)->velocity;
   
   
            currentCM.XRef()=(newCell->xCM)/(float)(newCell->volume);
            currentCM.YRef()=(newCell->yCM)/(float)(newCell->volume);
            currentCM.ZRef()=(newCell->zCM)/(float)(newCell->volume);
   
            velocity.XRef()=findMin(currentCM.X()-beforeFlipCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
            velocity.YRef()=findMin(currentCM.Y()-beforeFlipCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
            velocity.ZRef()=findMin(currentCM.Z()-beforeFlipCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

            //here I update beforeFlipCM - it will be valid untill next spin flip actually occurs
            beforeFlipCM=currentCM;

         }

   }


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void VelocityPlugin::readXML(XMLPullParser &in) {
  in.skip(TEXT);
}

void VelocityPlugin::writeXML(XMLSerializer &out) {
}

};

