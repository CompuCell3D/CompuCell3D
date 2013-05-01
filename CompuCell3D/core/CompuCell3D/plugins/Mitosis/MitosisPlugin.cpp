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

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>
// // // #include <PublicUtilities/ParallelUtilsOpenMP.h>

using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>

// // // #include <iostream>
using namespace std;


#include "MitosisPlugin.h"

MitosisPlugin::MitosisPlugin() {potts=0;}

MitosisPlugin::~MitosisPlugin() {}

void MitosisPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {


  potts = simulator->getPotts();


     bool pluginAlreadyRegisteredFlag;
   Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
	cerr<<"GOT HERE BEFORE CALLING INIT"<<endl;
	if(!pluginAlreadyRegisteredFlag)
      plugin->init(simulator);

   simulator->getPotts()->registerCellGChangeWatcher(this);
   simulator->getPotts()->registerStepper(this);
   


   boundaryStrategy=BoundaryStrategy::getInstance();
   maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(5);

   simulator->registerSteerableObject(this);

   pUtils=simulator->getParallelUtils();
   unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
   childCellVec.assign(maxNumberOfWorkNodes,0);
   parentCellVec.assign(maxNumberOfWorkNodes,0);
   splitPtVec.assign(maxNumberOfWorkNodes,Point3D());
   splitVec.assign(maxNumberOfWorkNodes,false);
   onVec.assign(maxNumberOfWorkNodes,false);
   mitosisFlagVec.assign(maxNumberOfWorkNodes,false);

   turnOn(); //this can be called only after vectors have been allocated

	cerr<<"maxNumberOfWorkNodes="<<maxNumberOfWorkNodes<<endl;
	update(_xmlData,true);

}

void MitosisPlugin::handleEvent(CC3DEvent & _event){
    if (_event.id==CHANGE_NUMBER_OF_WORK_NODES){    
       unsigned int maxNumberOfWorkNodes=pUtils->getMaxNumberOfWorkNodesPotts();
       childCellVec.assign(maxNumberOfWorkNodes,0);
       parentCellVec.assign(maxNumberOfWorkNodes,0);
       splitPtVec.assign(maxNumberOfWorkNodes,Point3D());
       splitVec.assign(maxNumberOfWorkNodes,false);
       onVec.assign(maxNumberOfWorkNodes,false);
       mitosisFlagVec.assign(maxNumberOfWorkNodes,false);
       turnOn();

    }

}


void MitosisPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
     
	doublingVolume=_xmlData->getFirstElement("DoublingVolume")->getUInt();

}


void MitosisPlugin::field3DChange(const Point3D &pt, CellG *newCell,
				  CellG *oldCell) {

  // Note: We cannot mitosis the cell here because  updating the field
  // from this function would cause further field3DChange calls which
  // would be out of order for any change listeners who  happened to
  // be after this one in the list.


  if (newCell){
//       cerr<<"this is the mitosis newCell "<<newCell<<endl;
//       cerr<<" DoublingVolume="<<doublingVolume<<endl;
      if(newCell->volume>= doublingVolume){
		 int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();
         splitVec[currentWorkNodeNumber] = true;
         splitPtVec[currentWorkNodeNumber] = pt;

      }
  }

}

void MitosisPlugin::turnOn() {
	onVec.assign(onVec.size(),true);
	//cerr<<"pUtils->getCurrentWorkNodeNumber()="<<pUtils->getCurrentWorkNodeNumber()<<" onVec.size()="<<onVec.size()<<endl;
	//onVec[pUtils->getCurrentWorkNodeNumber()] = true;

}
void MitosisPlugin::turnOff() {
	onVec.assign(onVec.size(),false);
	//onVec[pUtils->getCurrentWorkNodeNumber()] = false;
}

//void MitosisPlugin::turnOnAll(){
//	onVec.assign(onVec.size(),true);
//}
//void MitosisPlugin::turnOffAll(){
//	onVec.assign(onVec.size(),false);
//}


void MitosisPlugin::step() {
   bool didMitosis=doMitosis();
   if(didMitosis){
      updateAttributes();
    
   }
   
}

CellG * MitosisPlugin::getChildCell(){
	//cerr<<" getting CHILD CELL WORKNODE="<<pUtils->getCurrentWorkNodeNumber()<<" cell = "<<childCellVec[pUtils->getCurrentWorkNodeNumber()]<<endl;
	return childCellVec[pUtils->getCurrentWorkNodeNumber()];
}
CellG * MitosisPlugin::getParentCell(){
	return parentCellVec[pUtils->getCurrentWorkNodeNumber()];
}


void MitosisPlugin::updateAttributes(){

   ///copying type and target volume of the parent cell to the new cell
   int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();
   childCellVec[currentWorkNodeNumber]->type = parentCellVec[currentWorkNodeNumber]->type;
   childCellVec[currentWorkNodeNumber]->targetVolume = parentCellVec[currentWorkNodeNumber]->targetVolume;
}

bool MitosisPlugin::doMitosis(){
   
  bool didMitosis=false;
  int currentWorkNodeNumber=pUtils->getCurrentWorkNodeNumber();	

   // simplifying access to vectorized class variables
  short & split = splitVec[currentWorkNodeNumber];
  short & on = onVec[currentWorkNodeNumber];
  CellG * & childCell = childCellVec[currentWorkNodeNumber];
  CellG * & parentCell = parentCellVec[currentWorkNodeNumber];
  Point3D & splitPt=splitPtVec[currentWorkNodeNumber];

  if (split && on) {

    split= false;

    WatchableField3D<CellG *> *cellField =(WatchableField3D<CellG *> *) potts->getCellFieldG();
    //reseting poiters to parent and child cell - neessary otherwise may get some strange side effects when mitosis is aborted
    childCell=0;
    parentCell=0;

    CellG *cell = cellField->get(splitPtVec[currentWorkNodeNumber]);
    parentCell=cell;

    ASSERT_OR_THROW("Cell should not be NULL at mitosis point!", cell);

    int volume = cell->volume;
    int newVol = 0;
    int targetVol = volume / 2;



    vector<Point3D> ary0Vec;
    vector<Point3D> ary1Vec;

    vector<Point3D> *tmp;;
    vector<Point3D> *ary0=&ary0Vec;
    vector<Point3D> *ary1=&ary1Vec;
    
    ary0->clear();
    ary1->clear();

    // Put the first point in the array
    ary0->push_back(splitPt);

    CellG *splitCell = 0;

    // Do a breadth first search for approximately half of the  cell's
    // volume.
    Neighbor neighbor;    
    while (ary0->size() > 0 && newVol < targetVol) {
      Point3D n;
      CellG *nCell;
      // Loop over all the points from the last round.
      for (unsigned int i = 0; i < ary0->size(); i++) {
         unsigned int token = 0;
         double distance = 0;

      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect((*ary0)[i],nIdx);
         if(!neighbor.distance){
         //if distance is 0 then the neighbor returned is invalid
         continue;
         }

      
         // Add to split cell
          nCell=cellField->get(neighbor.pt);

         
         // Add to split cell
         if (nCell/*cellField->get(n)*/ == cell) {

            ary1->push_back(neighbor.pt);

            newVol++;
            if (splitCell){
               cellField->set(neighbor.pt, splitCell);
            }
            else{
               splitCell = potts->createCellG(neighbor.pt);
               childCell=splitCell;
            }
            if (newVol >= targetVol) break;
         }
         }

         if (newVol >= targetVol) break;
      }

      // Swap arrays;
      tmp = ary0;
      ary0 = ary1;
      ary1 = tmp;
      ary1->clear();
    }
   if(!childCell){
      cerr<<"Fragmented Cell - mitosis aborted"<<endl;
      didMitosis=false;
      return didMitosis;
   }
   
   if(childCell && fabs((float)childCell->volume-parentCell->volume)>2.0){
      cerr<<"cell was fragmented before mitosis, volumes of parent and child cells might significantly differ"<<endl;
//       cerr<<"C++ childCell.volume="<<childCell->volume<<" parentCell.volume="<<parentCell->volume<<endl;
      didMitosis=true;
      return didMitosis;
   }

  }else{//this means that mitosis loop was not called
      didMitosis=false;
      return didMitosis;
  }

//Everything went fine, mitosis suceeded witho no side effects
didMitosis=true;
return didMitosis;


}



std::string MitosisPlugin::toString(){
   return "Mitosis";
}


std::string MitosisPlugin::steerableName(){
   return toString();
}


















