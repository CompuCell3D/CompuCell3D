/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#include "ViscosityEnergy.h"

#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Simulator.h>
using namespace CompuCell3D;

#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>

#include <CompuCell3D/Potts3D/Cell.h>

#include <CompuCell3D/plugins/CellVelocity/CellVelocityData.h>
#include <CompuCell3D/plugins/CenterOfMass/CenterOfMassPlugin.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTracker.h>

#include <Utils/Coordinates3D.h>
#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h>

#include <CompuCell3D/plugins/CellVelocity/InstantVelocityData.h>
#include <CompuCell3D/plugins/CellVelocity/CellInstantVelocityPlugin.h>

#include <PublicUtilities/NumericalUtils.h>
#include <string>
using namespace std;

double ViscosityEnergy::localEnergy(const Point3D &pt) {
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ViscosityEnergy::initializeViscosityEnergy(){
   potts = simulator->getPotts();
   fieldDim=potts->getCellFieldG()->getDim();
   
   potts->getBoundaryXName()=="Periodic" ? boundaryConditionIndicator.x=1 : boundaryConditionIndicator.x=0 ;
   potts->getBoundaryYName()=="Periodic" ? boundaryConditionIndicator.y=1 : boundaryConditionIndicator.y=0;
   potts->getBoundaryZName()=="Periodic" ? boundaryConditionIndicator.z=1 : boundaryConditionIndicator.z=0;
   cerr<<"boundaryConditionIndicator="<<boundaryConditionIndicator<<endl;

   velPlug = (CellInstantVelocityPlugin*)(Simulator::pluginManager.get("CellInstantVelocity"));
   ASSERT_OR_THROW("CellInstantVelocity plugin not initialized!", velPlug);

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ViscosityEnergy::readXML(XMLPullParser &in) {
  in.skip(TEXT);

  automaton = potts->getAutomaton();
  while (in.check(START_ELEMENT)) {
    if (in.getName() == "LambdaViscosity") {
      lambdaViscosity = BasicString::parseDouble(in.matchSimple());
      

    } 
    else {
      throw BasicException(string("Unexpected element '") + in.getName() +
                           "'!", in.getLocation());
    }

    in.skip(TEXT);
  }

}

void ViscosityEnergy::writeXML(XMLSerializer &out) {
}



double ViscosityEnergy::changeEnergy(const Point3D &pt,
                                  const CellG *newCell,
                                  const CellG *oldCell) {
   if(simulator->getStep()<2){
      return 0;
   }
   
  double energy = 0;
  unsigned int token = 0;
  double distance = 0;
  Point3D n;
  double cellDistance=0.0;
  double commonArea=0.0;
  float x0,y0,z0,x1,y1,z1;
  double velocityDiffX=0;
  double velocityDiffY=0;
  double velocityDiffZ=0;
  Coordinates3D<float> nCellCom0,nCellCom1,cellCom0,cellCom1;

  Coordinates3D<float> oldCellCMBefore, oldCellCMAfter, newCellCMBefore, newCellCMAfter;
  Coordinates3D<float> nCellCMBefore,nCellCMAfter;

  CellG *nCell=0;
  Field3D<CellG *> *fieldG = potts->getCellFieldG();

  std::set<NeighborSurfaceData > *oldCellNeighborsPtr=0;
  std::set<NeighborSurfaceData > *newCellNeighborsPtr=0;

  std::set<NeighborSurfaceData >::iterator sitr;

  set<NeighborSurfaceData> oldCellPixelNeighborSurfaceData;
  set<NeighborSurfaceData>::iterator sitrNSD;
  set<NeighborSurfaceData>::iterator sitrNSDTmp;

  bool printFlag=false;


   precalculateAfterFlipInstantVelocityData(pt,newCell,oldCell);

  if(oldCell){

   //new code
      oldCellNeighborsPtr = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);
      oldCellCMAfter  = ivd.oldCellCM;

      oldCellCMBefore=Coordinates3D<float>(
               oldCell->xCM/(float)oldCell->volume ,
               oldCell->yCM/(float)oldCell->volume ,
               oldCell->zCM/(float)oldCell->volume

               );


   //new code


  }
  if(newCell){

   //new code
   newCellNeighborsPtr = &(neighborTrackerAccessorPtr->get(newCell->extraAttribPtr)->cellNeighbors);
   newCellCMAfter  = ivd.newCellCM;

   newCellCMBefore = Coordinates3D<float>(newCell->xCM/(float)newCell->volume ,
                                          newCell->yCM/(float)newCell->volume ,
                                          newCell->zCM/(float)newCell->volume
                                          );



  }


   //will compute here common surface area of old cell pixel with its all nearest neighbors
  while (true) {
    n = fieldG->getNeighbor(pt, token, distance, false);
    if (distance > depth) break;

    nCell = fieldG->get(n);
    if(!nCell) continue;

    sitrNSD=oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));
    if(sitrNSD != oldCellPixelNeighborSurfaceData.end()){
      sitrNSD->incrementCommonSurfaceArea(*sitrNSD);
    }else{
      oldCellPixelNeighborSurfaceData.insert(NeighborSurfaceData(nCell,1));
    }
  }

   //NOTE: There is a double counting issue count energy between old and new cell - as it is written in the paper

   //energy before flip from old cell
   if(oldCell){


      for(sitr=oldCellNeighborsPtr->begin() ; sitr != oldCellNeighborsPtr->end() ; ++sitr){

         nCell= sitr -> neighborAddress;

         if (!nCell) continue; //in case medium is a nieighbor
         
         commonArea = sitr -> commonSurfaceArea;
         
         
         
//          if(nCell->type==2){
//             printFlag=true;
//          }



         //cell velocity data -  difference!

         velocityDiffX = (cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr))->getInstantenousVelocity().X()
                         -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().X();

         velocityDiffY = (cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr))->getInstantenousVelocity().Y()
                          -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Y();

         velocityDiffZ = (cellVelocityDataAccessorPtr->get(oldCell->extraAttribPtr))->getInstantenousVelocity().Z()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Z();

         nCellCMBefore=Coordinates3D<float>(
               nCell->xCM/(float)nCell->volume ,
               nCell->yCM/(float)nCell->volume ,
               nCell->zCM/(float)nCell->volume

               );
                           
         x0=findMin(oldCellCMBefore.X()-nCellCMBefore.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         y0=findMin(oldCellCMBefore.Y()-nCellCMBefore.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         z0=findMin(oldCellCMBefore.Z()-nCellCMBefore.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

         cellDistance = dist(x0,y0,z0);

         energy-=commonArea*(
                     velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
                     +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
                     +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
                     )
                     /(cellDistance*cellDistance*cellDistance);


      }

   }


 
   //energy before flip from new cell
   if(newCell){

      for(sitr=newCellNeighborsPtr->begin() ; sitr != newCellNeighborsPtr->end() ; ++sitr){

         nCell= sitr -> neighborAddress;
         
         
         
//          if (!nCell){
//             cerr<<"PROBLEM:  neighbor of new cell before flip has zero ptr"<<endl;
//             cerr<<"number of neighbors newCellNeighborsPtr->size()="<<newCellNeighborsPtr->size()<<endl;
//             cerr<<"pt="<<pt<<" newCell->type="<<(int)newCell->type<<endl;
//             
//          }

         if (!nCell) continue; //in case medium is a nieighbor
         ///DOUBLE COUNTING PROTECTION *******************************************************************

         if(nCell==oldCell) continue; //to avoid double counting of newCell-oldCell eenrgy

         commonArea = sitr -> commonSurfaceArea;




         velocityDiffX = (cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr))->getInstantenousVelocity().X()
                         -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().X();

         velocityDiffY = (cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr))->getInstantenousVelocity().Y()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Y();

         velocityDiffZ = (cellVelocityDataAccessorPtr->get(newCell->extraAttribPtr))->getInstantenousVelocity().Z()
                         -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Z();;

         nCellCMBefore=Coordinates3D<float>(
               nCell->xCM/(float)nCell->volume ,
               nCell->yCM/(float)nCell->volume ,
               nCell->zCM/(float)nCell->volume

               );

         x0=findMin(newCellCMBefore.X()-nCellCMBefore.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         y0=findMin(newCellCMBefore.Y()-nCellCMBefore.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         z0=findMin(newCellCMBefore.Z()-nCellCMBefore.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

         cellDistance = dist(x0,y0,z0);


         energy-=commonArea*(
                     velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))

                     +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
                     +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
                     )
                     /(cellDistance*cellDistance*cellDistance);


      }
   }

   

   //energy after flip from old cell
   if(oldCell){

      for(sitr=oldCellNeighborsPtr->begin() ; sitr != oldCellNeighborsPtr->end() ; ++sitr){
         nCell= sitr -> neighborAddress;
         if (!nCell) continue; //in case medium is a nieighbor
         //will need to adjust commonArea for after flip case
         commonArea = sitr -> commonSurfaceArea;
         sitrNSD = oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));

         if(sitrNSD != oldCellPixelNeighborSurfaceData.end() ){
            if(sitrNSD->neighborAddress != newCell){ // if neighbor pixel is not a newCell we decrement commonArea
               commonArea-=sitrNSD->commonSurfaceArea;
            }
            else{//otherwise we do the following
               sitrNSDTmp=oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(const_cast<CellG*>(oldCell)));
               commonArea-=sitrNSD->commonSurfaceArea;//we subtract common area of pixel with newCell
               if(sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()){// in case old cell is not
                                                                        //on the list of oldPixelNeighbors
                  commonArea+=sitrNSDTmp->commonSurfaceArea;//we add common area of pixel with oldCell
               }

            }
         }

//                if(sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()){
//                   ;
//                }else{
//                   cerr<<"sitrNSDTmp is poiting to end of the set PROBLEM!!! commonArea="<<sitrNSDTmp->commonSurfaceArea<<endl;
//                   cerr<<"OLD CELL="<<oldCell<<" NEW CELL="<<newCell<<endl;
//                   for(set<NeighborSurfaceData>::iterator itr=oldCellPixelNeighborSurfaceData.begin();
//                   itr!=oldCellPixelNeighborSurfaceData.end();
//                   ++itr
//                   ){
//                      cerr<<"neighborAddress="<<itr->neighborAddress<<endl;
//                      cerr<<"commonSurfaceArea="<<itr->commonSurfaceArea<<endl;
//                   }
//                   exit(0);
//                }

         
         if(commonArea<0.0){ //just in case
            commonArea=0.0;
//             cerr<<"reached below zero old after"<<endl;
            }
         if(nCell!=newCell){

            velocityDiffX = ivd.oldCellV.X()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().X();

            velocityDiffY = ivd.oldCellV.Y()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Y();

            velocityDiffZ = ivd.oldCellV.Z()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Z();

            nCellCMAfter=Coordinates3D<float>(
                  nCell->xCM/(float)nCell->volume ,
                  nCell->yCM/(float)nCell->volume ,
                  nCell->zCM/(float)nCell->volume

                  );

         }else{
            velocityDiffX = ivd.oldCellV.X()
                           -ivd.newCellV.X();

            velocityDiffY = ivd.oldCellV.Y()
                           -ivd.newCellV.Y();

            velocityDiffZ = ivd.oldCellV.Z()
                           -ivd.newCellV.Z();
            nCellCMAfter = ivd.newCellCM;
         }


         x0=findMin(oldCellCMAfter.X()-nCellCMAfter.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         y0=findMin(oldCellCMAfter.Y()-nCellCMAfter.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         z0=findMin(oldCellCMAfter.Z()-nCellCMAfter.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );


         cellDistance = dist(x0,y0,z0);




         energy+=commonArea*(
                     velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
                     +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
                     +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
                     )
                     /(cellDistance*cellDistance*cellDistance);

      }
   }
   


   //energy after flip from new cell

   if(newCell){

      for( sitr = newCellNeighborsPtr->begin() ; sitr != newCellNeighborsPtr->end() ; ++sitr ){
         nCell= sitr -> neighborAddress;
         if (!nCell) continue; //in case medium is a nieighbor

         ///DOUBLE COUNTING PROTECTION *******************************************************************
         if(nCell==oldCell) continue; //to avoid double counting of newCell-oldCell eenrgy
         //will need to adjust commonArea for after flip case
         commonArea = sitr -> commonSurfaceArea;

         sitrNSD = oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(nCell));
         if(sitrNSD != oldCellPixelNeighborSurfaceData.end() ){
            if(sitrNSD->neighborAddress != oldCell){ // if neighbor is not a oldCell we increment commonArea
               commonArea+=sitrNSD->commonSurfaceArea;
            }
            else{//otherwise we do the following
               sitrNSDTmp=oldCellPixelNeighborSurfaceData.find(NeighborSurfaceData(const_cast<CellG*>(newCell)));
               if(sitrNSDTmp != oldCellPixelNeighborSurfaceData.end()){// in case new cell is not
                                                                        //on the list of oldPixelNeighbors
                  commonArea-=sitrNSDTmp->commonSurfaceArea;//we subtract common area of pixel with newCell
               }
               commonArea+=sitrNSD->commonSurfaceArea;//we add common area of pixel with oldCell

            }
         }
         if(commonArea<0.0){ //just in case
            commonArea=0.0;
//             cerr<<"reached below zero new after"<<endl;
         }


         if(nCell!=oldCell){

            velocityDiffX = ivd.newCellV.X()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().X();

            velocityDiffY = ivd.newCellV.Y()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Y();

            velocityDiffZ = ivd.newCellV.Z()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Z();


            nCellCMAfter=Coordinates3D<float>(
                  nCell->xCM/(float)nCell->volume ,
                  nCell->yCM/(float)nCell->volume ,
                  nCell->zCM/(float)nCell->volume

                  );

         }else{
            velocityDiffX = ivd.newCellV.X()
                           -ivd.oldCellV.X();

            velocityDiffY = ivd.newCellV.Y()
                           -ivd.oldCellV.Y();

            velocityDiffZ = ivd.newCellV.Z()
                           -ivd.oldCellV.Z();
            nCellCMAfter = ivd.oldCellCM;
         }


         x0=findMin(newCellCMAfter.X()-nCellCMAfter.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         y0=findMin(newCellCMAfter.Y()-nCellCMAfter.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         z0=findMin(newCellCMAfter.Z()-nCellCMAfter.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

         cellDistance = dist(x0,y0,z0);




         energy+=commonArea*(
                     velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))
                     +velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))
                     +velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
                     )
                     /(cellDistance*cellDistance*cellDistance);


      }

 
    }
    //exclusive case for new neighbors which may occur after the flip
    // examining neighbors of the oldpixel
   if(newCell){


      for(sitrNSD = oldCellPixelNeighborSurfaceData.begin() ; sitrNSD != oldCellPixelNeighborSurfaceData.end() ;++sitrNSD ){
         sitr = newCellNeighborsPtr->find(NeighborSurfaceData(sitrNSD->neighborAddress));
         if(sitr==newCellNeighborsPtr->end()){//pixel neighbor does not show up in newCellNeighbors - we have found new neighbor of newCell

            nCell = sitrNSD->neighborAddress;
            if (!nCell) continue; //in case medium is a nieighbor

            if(nCell==newCell || nCell==oldCell) continue;

            commonArea = sitrNSD->commonSurfaceArea;

            velocityDiffX = ivd.newCellV.X()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().X();
   
            velocityDiffY = ivd.newCellV.Y()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Y();
   
            velocityDiffZ = ivd.newCellV.Z()
                           -(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr))->getInstantenousVelocity().Z();
   
//             nCellCMAfter = (*(cellVelocityDataAccessorPtr->get(nCell->extraAttribPtr)))[0];

            nCellCMAfter=Coordinates3D<float>(
                  nCell->xCM/(float)nCell->volume ,
                  nCell->yCM/(float)nCell->volume ,
                  nCell->zCM/(float)nCell->volume

                  );


            x0=findMin(newCellCMAfter.X()-nCellCMAfter.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
            y0=findMin(newCellCMAfter.Y()-nCellCMAfter.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
            z0=findMin(newCellCMAfter.Z()-nCellCMAfter.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );
   
            cellDistance = dist(x0,y0,z0);
   
   

            energy+=commonArea*(
                        velocityDiffX*velocityDiffX*sqrt((y0)*(y0)+(z0)*(z0))+
                        velocityDiffY*velocityDiffY*sqrt((z0)*(z0)+(x0)*(x0))+
                        velocityDiffZ*velocityDiffZ*sqrt((x0)*(x0)+(y0)*(y0))
                        )
                        /(cellDistance*cellDistance*cellDistance);


         }
      }
   }


  return lambdaViscosity*energy;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ViscosityEnergy::precalculateAfterFlipInstantVelocityData(const Point3D &pt, const CellG *newCell, const CellG *oldCell){
   if(!simulator)
      return;

   //may have to update CM history here

   if(simulator->getStep() < 1) //protect from updating cell velocity during initialization
      return;

   Coordinates3D<float> v;
   Coordinates3D<float> prevV;
   Coordinates3D<float> oldCM;
   Coordinates3D<float> newCM;

   ivd.zeroAll();
   precalculateAfterFlipCM(pt,newCell,oldCell);


   if(oldCell){

       if(oldCell->volume  > 0 ){

         oldCM = Coordinates3D<float>(
            oldCell->xCM/(float)oldCell->volume,
            oldCell->yCM/(float)oldCell->volume,
            oldCell->zCM/(float)oldCell->volume
         );
         newCM = ivd.oldCellCM;


         v.XRef()=findMin(newCM.X()-oldCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
         v.YRef()=findMin(newCM.Y()-oldCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
         v.ZRef()=findMin(newCM.Z()-oldCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

         ivd.oldCellV=v;//most up2date instant velocity for oldCell


       }


    }

   if(newCell){



         if(newCell->volume  == 1){

            newCM=ivd.newCellCM;

            ivd.newCellV  = Coordinates3D<float>(0,0,0);//most up2date instant velocity for newCell



         }

         if(newCell->volume  > 1 ){


            oldCM = Coordinates3D<float>(
               newCell->xCM/(float)newCell->volume,
               newCell->yCM/(float)newCell->volume,
               newCell->zCM/(float)newCell->volume
            );

            newCM = ivd.newCellCM;


            v.XRef()=findMin(newCM.X()-oldCM.X(), boundaryConditionIndicator.x ? fieldDim.x : 0 );
            v.YRef()=findMin(newCM.Y()-oldCM.Y(), boundaryConditionIndicator.y ? fieldDim.y : 0 );
            v.ZRef()=findMin(newCM.Z()-oldCM.Z(), boundaryConditionIndicator.z ? fieldDim.z : 0 );

            ivd.newCellV  = v;//most up2date instant velocity for newCell


         }

   }




}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ViscosityEnergy::precalculateAfterFlipCM(const Point3D &pt, const CellG *newCell, const CellG *oldCell){
   
   //if no boundary conditions are present
   if ( !boundaryConditionIndicator.x && !boundaryConditionIndicator.y && !boundaryConditionIndicator.z ){

      
      if (oldCell) {
         ivd.oldCellCM.XRef()=oldCell->xCM-pt.x;
         ivd.oldCellCM.YRef()=oldCell->xCM-pt.y;
         ivd.oldCellCM.ZRef()=oldCell->xCM-pt.z;

         if(oldCell->volume>1){
            ivd.oldCellCM.XRef()=ivd.oldCellCM.XRef()/((float)oldCell->volume-1);
            ivd.oldCellCM.YRef()=ivd.oldCellCM.YRef()/((float)oldCell->volume-1);
            ivd.oldCellCM.ZRef()=ivd.oldCellCM.ZRef()/((float)oldCell->volume-1);
         }else{
         
            ivd.oldCellCM.XRef()=oldCell->xCM;
            ivd.oldCellCM.YRef()=oldCell->xCM;
            ivd.oldCellCM.ZRef()=oldCell->xCM;
         
            

         }

      }

      if (newCell) {
      
         ivd.newCellCM.XRef()=(newCell->xCM+pt.x)/((float)newCell->volume+1);
         ivd.newCellCM.YRef()=(newCell->xCM+pt.y)/((float)newCell->volume+1);
         ivd.newCellCM.ZRef()=(newCell->xCM+pt.z)/((float)newCell->volume+1);
      

      }

      return;
   }

   //if there are boundary conditions defined that we have to do some shifts to correctly calculate center of mass
   //This approach will work only for cells whose span is much smaller that lattice dimension in the "periodic "direction
   //e.g. cell that is very long and "wraps lattice" will have miscalculated CM using this algorithm. On the other hand, you do not real expect
   //cells to have dimensions comparable to lattice...
   

   
   Point3D shiftVec;
   Point3D shiftedPt;
   int xCM,yCM,zCM; //temp centroids

   int x,y,z;
   int xo,yo,zo;
//     cerr<<"CM PLUGIN"<<endl;
    
  if (oldCell) {

   xo=oldCell->xCM;
   yo=oldCell->yCM;
   zo=oldCell->zCM;

        

      x=oldCell->xCM-pt.x;
      y=oldCell->yCM-pt.y;
      z=oldCell->zCM-pt.z;
    //calculating shiftVec - to translate CM


    //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
    shiftVec.x= (short)((oldCell->xCM/(float)(oldCell->volume)-fieldDim.x/2)*boundaryConditionIndicator.x);
    shiftVec.y= (short)((oldCell->yCM/(float)(oldCell->volume)-fieldDim.y/2)*boundaryConditionIndicator.y);
    shiftVec.z= (short)((oldCell->zCM/(float)(oldCell->volume)-fieldDim.z/2)*boundaryConditionIndicator.z);

    //shift CM to approximately center of lattice, new centroids are:
    xCM = oldCell->xCM - shiftVec.x*(oldCell->volume);
    yCM = oldCell->yCM - shiftVec.y*(oldCell->volume);
    zCM = oldCell->zCM - shiftVec.z*(oldCell->volume);
    //Now shift pt
    shiftedPt=pt;
    shiftedPt-=shiftVec;
    
    //making sure that shifterd point is in the lattice
    if(shiftedPt.x < 0){
      shiftedPt.x += fieldDim.x;
    }else if (shiftedPt.x > fieldDim.x-1){
      shiftedPt.x -= fieldDim.x;
    }  

    if(shiftedPt.y < 0){
      shiftedPt.y += fieldDim.y;
    }else if (shiftedPt.y > fieldDim.y-1){
      shiftedPt.y -= fieldDim.y;
    }  

    if(shiftedPt.z < 0){
      shiftedPt.z += fieldDim.z;
    }else if (shiftedPt.z > fieldDim.z-1){
      shiftedPt.z -= fieldDim.z;
    }
    //update shifted centroids
    xCM -= shiftedPt.x;
    yCM -= shiftedPt.y;
    zCM -= shiftedPt.z;

    //shift back centroids
    xCM += shiftVec.x * (oldCell->volume-1);
    yCM += shiftVec.y * (oldCell->volume-1);
    zCM += shiftVec.z * (oldCell->volume-1);

    //Check if CM is in the lattice
    if( xCM/((float)oldCell->volume-1) < 0){
      xCM += fieldDim.x*(oldCell->volume-1);
    }else if ( xCM/((float)oldCell->volume -1)> fieldDim.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
                                                         //to avoid rollovers for unsigned int from oldCell->xCM
                                                         

       xCM -= fieldDim.x*(oldCell->volume-1);

     
    }

    if( yCM/((float)oldCell->volume-1) < 0){
      yCM += fieldDim.y*(oldCell->volume-1);
    }else if ( yCM/((float)oldCell->volume-1) > fieldDim.y){
      yCM -= fieldDim.y*(oldCell->volume-1);
    }

    if( zCM/((float)oldCell->volume-1) < 0){
      zCM += fieldDim.z*(oldCell->volume-1);
    }else if ( zCM/((float)oldCell->volume-1) > fieldDim.z){
      zCM -= fieldDim.z*(oldCell->volume-1);
    }

   
   if(oldCell->volume>1){
      ivd.oldCellCM.XRef()=xCM/((float)oldCell->volume-1);
      ivd.oldCellCM.YRef()=yCM/((float)oldCell->volume-1);
      ivd.oldCellCM.ZRef()=zCM/((float)oldCell->volume-1);
   }else{
   
      ivd.oldCellCM.XRef()=zCM;
      ivd.oldCellCM.YRef()=yCM;
      ivd.oldCellCM.ZRef()=zCM;
   
      

   }


  }

  if (newCell) {

    xo=newCell->xCM;
    yo=newCell->yCM;
    zo=newCell->zCM;


      x=newCell->xCM+pt.x;
      y=newCell->yCM+pt.y;
      z=newCell->zCM+pt.z;

  
    if(newCell->volume==1){
      shiftVec.x=0;
      shiftVec.y=0;
      shiftVec.z=0;
    }else{
      shiftVec.x= (short)((newCell->xCM/(float)(newCell->volume)-fieldDim.x/2)*boundaryConditionIndicator.x);
      shiftVec.y= (short)((newCell->yCM/(float)(newCell->volume)-fieldDim.y/2)*boundaryConditionIndicator.y);
      shiftVec.z= (short)((newCell->zCM/(float)(newCell->volume)-fieldDim.z/2)*boundaryConditionIndicator.z);
    
      
    }
    
    //if CM of the cell is too close to the "middle" of the lattice correct shift vector

    
    //shift CM to approximately center of lattice , new centroids are:
    xCM = newCell->xCM - shiftVec.x*(newCell->volume);
    yCM = newCell->yCM - shiftVec.y*(newCell->volume);
    zCM = newCell->zCM - shiftVec.z*(newCell->volume);
    //Now shift pt
    shiftedPt=pt;
    shiftedPt-=shiftVec;

    //making sure that shifted point is in the lattice
    if(shiftedPt.x < 0){
      shiftedPt.x += fieldDim.x;
    }else if (shiftedPt.x > fieldDim.x-1){
//       cerr<<"shifted pt="<<shiftedPt<<endl;
      shiftedPt.x -= fieldDim.x;
    }  

    if(shiftedPt.y < 0){
      shiftedPt.y += fieldDim.y;
    }else if (shiftedPt.y > fieldDim.y-1){
      shiftedPt.y -= fieldDim.y;
    }  

    if(shiftedPt.z < 0){
      shiftedPt.z += fieldDim.z;
    }else if (shiftedPt.z > fieldDim.z-1){
      shiftedPt.z -= fieldDim.z;
    }    

    //update shifted centroids
    xCM += shiftedPt.x;
    yCM += shiftedPt.y;
    zCM += shiftedPt.z;
    
    //shift back centroids
    xCM += shiftVec.x * (newCell->volume+1);
    yCM += shiftVec.y * (newCell->volume+1);
    zCM += shiftVec.z * (newCell->volume+1);
    
    //Check if CM is in the lattice
    if( xCM/((float)newCell->volume+1) < 0){
      xCM += fieldDim.x*(newCell->volume+1);
    }else if ( xCM/((float)newCell->volume+1) > fieldDim.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
                                                         //to avoid rollovers for unsigned int from oldCell->xCM
      xCM -= fieldDim.x*(newCell->volume+1);
    }

    if( yCM/((float)newCell->volume+1) < 0){
      yCM += fieldDim.y*(newCell->volume+1);
    }else if ( yCM/((float)newCell->volume+1) > fieldDim.y){
      yCM -= fieldDim.y*(newCell->volume+1);
    }

    if( zCM/((float)newCell->volume+1) < 0){
      zCM += fieldDim.z*(newCell->volume+1);
    }else if ( zCM/((float)newCell->volume+1) > fieldDim.z){
      zCM -= fieldDim.z*(newCell->volume+1);
    }

   
   ivd.newCellCM.XRef()=xCM/((float)newCell->volume+1);
   ivd.newCellCM.YRef()=yCM/((float)newCell->volume+1);
   ivd.newCellCM.ZRef()=zCM/((float)newCell->volume+1);

               
    

  }

}