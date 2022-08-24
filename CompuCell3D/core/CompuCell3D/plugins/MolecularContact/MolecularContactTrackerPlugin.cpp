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
 

#include <CompuCell3D/Automaton/Automaton.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
//#include <CompuCell3D/plugins/Volume/VolumePlugin.h>
//#include <CompuCell3D/plugins/Volume/VolumeEnergy.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <ctime>
using namespace CompuCell3D;


#include <iostream>
#include <cmath>
using namespace std;

#define EXP_STL
#include "MolecularContactTrackerPlugin.h"
#include<core/CompuCell3D/CC3DLogger.h>


MolecularContactTrackerPlugin::MolecularContactTrackerPlugin() :
   cellFieldG(0),
   initialized(false),
   maxNeighborIndex(0),
   boundaryStrategy(0),
	xmlData(0)

   {}

MolecularContactTrackerPlugin::~MolecularContactTrackerPlugin() {
}


void MolecularContactTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  simulator=_simulator;
  Potts3D *potts = simulator->getPotts();
  cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

  ///getting cell inventory
  cellInventoryPtr=& potts->getCellInventory(); 



  ///will register MolecularContactTracker here
  BasicClassAccessorBase * molecularcontactTrackerAccessorPtr=&molecularcontactTrackerAccessor;
   ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
   ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(molecularcontactTrackerAccessorPtr);

  
  
  fieldDim=cellFieldG->getDim();

  boundaryStrategy=BoundaryStrategy::getInstance();
  maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//1st nearest neighbor

   bool pluginAlreadyRegisteredFlag;
   Plugin *plugin=Simulator::pluginManager.get("CenterOfMass",&pluginAlreadyRegisteredFlag); //this will load COM plugin if it is not already loaded
  if(!pluginAlreadyRegisteredFlag)
      plugin->init(simulator);
   
  
  potts->registerCellGChangeWatcher(this);//register molecularcontactTracker after CenterOfMass and after VolumeTracker - implicitely called from CenterOfmass
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void MolecularContactTrackerPlugin::extraInit(Simulator *_simulator) {
	molecularcontactTypesNames.clear();
	molecularcontactTypes.clear();
	CC3DXMLElementList includeTypeNamesXMLVec=xmlData->getElements("IncludeType");
	for(int i = 0 ; i < includeTypeNamesXMLVec.size() ; ++i){
		molecularcontactTypesNames.insert(includeTypeNamesXMLVec[i]->getText());			
	}

   Automaton * automaton=simulator->getPotts()->getAutomaton();
   // Initializing set of molecularcontacttypes
   for (set<string>::iterator sitr = molecularcontactTypesNames.begin() ; sitr != molecularcontactTypesNames.end() ; ++sitr){
      molecularcontactTypes.insert(automaton->getTypeId( *sitr));
   }
   
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void MolecularContactTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,
				  CellG *oldCell) {  

   //do not do any updates until the lattice is fully initialized
/*   if(simulator->getStep()<0){
      return;
   }*/
   Potts3D *potts = simulator->getPotts();
  
   WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
   Neighbor neighbor;
   set<unsigned char>::iterator endSitr=molecularcontactTypes.end();

   
   
	
   if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;
   if(simulator->getStep()>=0 && ! initialized){
      initializeMolecularContactNeighborList();
      initialized=true;
   }


   std::set<NeighborSurfaceData> * neighborData;
   std::set<NeighborSurfaceData >::iterator sitr;
   CellG* nCell;
   bool pluginAlreadyRegisteredFlag;
   NeighborTrackerPlugin * neighborTrackerPluginPtr=(NeighborTrackerPlugin*)(Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag));
   neighborTrackerAccessorPtr=neighborTrackerPluginPtr->getNeighborTrackerAccessorPtr();

   if(oldCell){
      Log(LOG_DEBUG) << "ID: " << oldCell->id << " Type: " << (int)oldCell->type << " Address: " << oldCell;
//          sleep(5);
      neighborData = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);
//          neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr);
      set<MolecularContactTrackerData>::iterator PlasSetitr;
      set<MolecularContactTrackerData>::iterator tmpPlasSetitr;
      set<MolecularContactTrackerData> * molecularcontactNeighborsPtr=&molecularcontactTrackerAccessor.get(oldCell->extraAttribPtr)->molecularcontactNeighbors;
      set<MolecularContactTrackerData> OGmolecularcontactNeighborsPtr=molecularcontactTrackerAccessor.get(oldCell->extraAttribPtr)->molecularcontactNeighbors;
      set<MolecularContactTrackerData> * molecularcontactNeighborsTmpPtr;
      molecularcontactNeighborsPtr->clear();
      for(sitr=neighborData->begin() ; sitr != neighborData->end() ; ++sitr){
         nCell= sitr->neighborAddress;
         Log(LOG_DEBUG) << "\t NeigbhorID: " << sitr->neighborAddress;
         if(nCell) {
            if(molecularcontactTypes.find(nCell->type)==endSitr){
               Log(LOG_DEBUG) << "\t Type not inlucded in MolecularContact\n";
               Log(LOG_DEBUG) << "\tID: " << nCell->id << " Type: " << (int)nCell->type;
            }
            else{
                Log(LOG_DEBUG) << "\t Inserting NeigbhorID: " << sitr->neighborAddress << " Type: " << (int)nCell->type;
                Log(LOG_DEBUG) << "\t Before Inserting Size of Set: " << molecularcontactNeighborsPtr->size();
               molecularcontactNeighborsPtr->insert(MolecularContactTrackerData(nCell));
               molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(nCell->extraAttribPtr)->molecularcontactNeighbors;
               molecularcontactNeighborsTmpPtr->insert(MolecularContactTrackerData(oldCell));
                Log(LOG_DEBUG) << "\t After Inserting";
            }
         }
      }
      /*for(PlasSetitr=molecularcontactNeighborsPtr->begin() ; PlasSetitr != molecularcontactNeighborsPtr->end() ; ++PlasSetitr){
         molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(oldCell->extraAttribPtr)->molecularcontactNeighbors;
      }
      
      for(PlasSetitr=OGmolecularcontactNeighborsPtr.begin() ; PlasSetitr != OGmolecularcontactNeighborsPtr.end() ; ++PlasSetitr){
          molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(PlasSetitr->neighborAddress->extraAttribPtr)->molecularcontactNeighbors;
          for(tmpPlasSetitr=molecularcontactNeighborsTmpPtr->begin() ; tmpPlasSetitr != molecularcontactNeighborsTmpPtr->end() ; ++tmpPlasSetitr){
          }
                
//             molecularcontactNeighborsTmpPtr->insert(MolecularContactTrackerData(oldCell));
      }*/
//       std::set<unsigned char>::iterator typesit; 
//       for(typesit=molecularcontactTypes.begin() ; typesit != molecularcontactTypes.end();++typesit) {
//       }
   }

   
   

}


void MolecularContactTrackerPlugin::initializeMolecularContactNeighborList(){

   Point3D pt;
   CellG* cell;
   CellG* nCell;
   Neighbor neighbor;
   set<MolecularContactTrackerData> * molecularcontactNeighborsTmpPtr;
   set<unsigned char>::iterator endSitr=molecularcontactTypes.end();

   
   for( unsigned int x =0 ; x< fieldDim.x ; ++x)
      for( unsigned int y =0 ; y< fieldDim.y ; ++y)
         for( unsigned int z =0 ; z< fieldDim.z ; ++z){
            pt=Point3D(x,y,z);
            cell=cellFieldG->get(pt);
            for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
               neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
               if(!neighbor.distance){
                  //if distance is 0 then the neighbor returned is invalid
                  continue;
               }
               nCell = cellFieldG->get(neighbor.pt);
               if(nCell!=cell){
                  if(nCell && cell){//only cells which are of certain types are considered as molecularcontactNeighbors
                     if(molecularcontactTypes.find(nCell->type)!=endSitr && molecularcontactTypes.find(cell->type)!=endSitr){
                        molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(nCell->extraAttribPtr)->molecularcontactNeighbors;
                        molecularcontactNeighborsTmpPtr->insert(MolecularContactTrackerData(cell));
                        molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(cell->extraAttribPtr)->molecularcontactNeighbors;
                        molecularcontactNeighborsTmpPtr->insert(MolecularContactTrackerData(nCell));
                     }
                  }
               }
            }
         }
}

void MolecularContactTrackerPlugin::addMolecularContactNeighborList(){}
