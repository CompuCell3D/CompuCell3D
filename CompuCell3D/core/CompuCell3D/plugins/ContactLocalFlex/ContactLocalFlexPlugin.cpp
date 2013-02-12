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
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>
// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
// // // #include <algorithm>
// // // #include <set>

using namespace std;


#include "ContactLocalFlexPlugin.h"





ContactLocalFlexPlugin::ContactLocalFlexPlugin():
   pUtils(0),
   lockPtr(0),
   depth(1),
   weightDistance(false),
   boundaryStrategy(0),
   xmlData(0)  
{
   initializadContactData=false;
}

ContactLocalFlexPlugin::~ContactLocalFlexPlugin() {
  	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr=0;
}

void ContactLocalFlexPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {
  xmlData=_xmlData;
  sim=simulator;
  potts=simulator->getPotts();
  
   pUtils=sim->getParallelUtils();
   lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
   pUtils->initLock(lockPtr);   
  
  potts->getCellFactoryGroupPtr()->registerClass(&contactDataContainerAccessor);
  
  
   bool pluginAlreadyRegisteredFlag;
   Plugin *plugin=Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag); //this will load SurfaceTracker plugin if it is not already loaded
  if(!pluginAlreadyRegisteredFlag)
      plugin->init(sim);

  potts->registerEnergyFunction(this);
  potts->registerCellGChangeWatcher(this);
  simulator->registerSteerableObject(this);
}

void ContactLocalFlexPlugin::extraInit(Simulator *simulator){
	update(xmlData, true);
}

double ContactLocalFlexPlugin::changeEnergy(const Point3D &pt,
                                  const CellG *newCell,
                                  const CellG *oldCell) {
   //cerr<<"ChangeEnergy"<<endl;
   
   
  double energy = 0;
  Point3D n;
  
  CellG *nCell=0;
  WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
  Neighbor neighbor;


   if(weightDistance){
      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
         //if distance is 0 then the neighbor returned is invalid
         continue;
         }
         nCell = fieldG->get(neighbor.pt);
         if(nCell!=oldCell){
 				if((nCell != 0) && (oldCell != 0)) {
				   if((nCell->clusterId) != (oldCell->clusterId)) {
					  energy -= contactEnergy(oldCell,nCell)/ neighbor.distance;
				   }
				}else{
				   energy -= contactEnergy(oldCell, nCell)/ neighbor.distance;
			   }
         }
         if(nCell!=newCell){
				if((newCell != 0) && (nCell != 0)) {
				   if((newCell->clusterId) != (nCell->clusterId)) {
					  energy += contactEnergy(newCell,nCell)/ neighbor.distance;
				   }
				}
				else{
				   energy += contactEnergy(newCell, nCell)/ neighbor.distance;

				}
         }
   
   
      }

  }else{
         //default behaviour  no energy weighting 
      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
         //if distance is 0 then the neighbor returned is invalid
         continue;
         }
         
         nCell = fieldG->get(neighbor.pt);
         if(nCell!=oldCell){
				if((nCell != 0) && (oldCell != 0)) {
				   if((nCell->clusterId) != (oldCell->clusterId)) {
					  energy -= contactEnergy(oldCell,nCell);
				   }
				}else{
				   energy -= contactEnergy(oldCell, nCell);
			   }
/*            if(pt.x==25 && pt.y==74 && pt.z==0)
            cerr<<"!=oldCell neighbor.pt="<<neighbor.pt<<" contactEnergy(oldCell, nCell)="<<contactEnergy(oldCell, nCell)<<endl;*/
         }

         if(nCell!=newCell){
				if((newCell != 0) && (nCell != 0)) {
				   if((newCell->clusterId) != (nCell->clusterId)) {
					  energy += contactEnergy(newCell,nCell);
				   }
				}
				else{
				   energy += contactEnergy(newCell, nCell);

				}
//             if(pt.x==25 && pt.y==74 && pt.z==0)
//             cerr<<"!=newCell neighbor.pt="<<neighbor.pt<<" contactEnergy(oldCell, nCell)="<<contactEnergy(newCell, nCell)<<endl;

//             cerr<<"!=newCell neighbor.pt="<<neighbor.pt<<" energyTmp="<<energy<<endl;
         }
   
   
      }

   }

//   cerr<<"energy="<<energy<<endl;
  return energy;
}

void ContactLocalFlexPlugin::setContactEnergy(const string typeName1,
				     const string typeName2,
				     const double energy) {
                    
  char type1 = automaton->getTypeId(typeName1);
  char type2 = automaton->getTypeId(typeName2);
    
  int index = getIndex(type1, type2);

  contactEnergies_t::iterator it = contactEnergies.find(index);
  ASSERT_OR_THROW(string("Contact energy for ") + typeName1 + " " + typeName2 +
		  " already set!", it == contactEnergies.end());

  contactEnergies[index] = energy;
}

int ContactLocalFlexPlugin::getIndex(const int type1, const int type2) const {
  if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
  else return ((type2 + 1) | ((type1 + 1) << 16));
}



double ContactLocalFlexPlugin::contactEnergy(const CellG *cell1, const CellG *cell2) {
   ContactLocalFlexData clfdObj;
   CellG *cell;
   CellG *neighbor;

   if(cell1){
      cell=const_cast<CellG *>(cell1);
      neighbor=const_cast<CellG *>(cell2);
   }else{
      cell=const_cast<CellG *>(cell2);
      neighbor=const_cast<CellG *>(cell1);
   }

   set<ContactLocalFlexData> & clfdSet = contactDataContainerAccessor.get(cell->extraAttribPtr)->contactDataContainer;
   
   
   clfdObj.neighborAddress=neighbor;
 
		set<ContactLocalFlexData>::iterator sitrCD=clfdSet.find(clfdObj);

   if( sitrCD != clfdSet.end() ){
       //cerr<<"\t retrieving cell->type="<<(int)cell->type<<" neighbor->type="<<(neighbor? (int)neighbor->type:0)<<" energy="<<sitrCD->J<<endl;
      return sitrCD->J;
   }else{
//       cerr<<"\t\t default energy="<<defaultContactEnergy(cell1,cell2)<<endl;
      return defaultContactEnergy(cell1,cell2);
   }

}

double ContactLocalFlexPlugin::defaultContactEnergy(const CellG *cell1, const CellG *cell2){
   //implementing only referring to the local defaultContactEnergy
   return contactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];
}




//this function is called once per simulation after cells have been assigned types (some initializers postpone type initialization)
void ContactLocalFlexPlugin::initializeContactLocalFlexData(){

    if (initializadContactData) //we double-check this flag to makes sure this function does not get called multiple times by different threads
        return;
        
   CellInventory * cellInventoryPtr=&potts->getCellInventory();
   CellInventory::cellInventoryIterator cInvItr;
   CellG * cell;
   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
		cell=cellInventoryPtr->getCell(cInvItr);
      //cell=*cInvItr;
      ContactLocalFlexDataContainer *dataContainer = contactDataContainerAccessor.get(cell->extraAttribPtr);
      dataContainer->localDefaultContactEnergies = contactEnergyArray;
   }
   


   for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
		  cell=cellInventoryPtr->getCell(cInvItr);
        //cell=*cInvItr;
        set<ContactLocalFlexData> & clfdSet=contactDataContainerAccessor.get(cell->extraAttribPtr)->contactDataContainer;
        clfdSet.clear();
        updateContactEnergyData(cell);
   }

    initializadContactData=true;
}

void ContactLocalFlexPlugin::updateContactEnergyData(CellG *_cell){
   //this function syncs neighbor list for _cell and contac data for _cell so that they contain same number of corresponding 
   //entries

   NeighborTrackerPlugin *neighborTrackerPlugin=(NeighborTrackerPlugin *)Simulator::pluginManager.get("NeighborTracker");
   BasicClassAccessor<NeighborTracker> *neighborTrackerAccessorPtr = neighborTrackerPlugin->getNeighborTrackerAccessorPtr();
   unsigned int size1=0, size2=0;   
//neighborTrackerAccessor.get(newCell->extraAttribPtr)->cellNeighbors


      set<ContactLocalFlexData> & clfdSet=contactDataContainerAccessor.get(_cell->extraAttribPtr)->contactDataContainer;
      set<NeighborSurfaceData> & nsdSet = neighborTrackerAccessorPtr->get(_cell->extraAttribPtr)->cellNeighbors;

      size1=clfdSet.size();
      size2=nsdSet.size();
      

     //if sizes of sets are different then we add any new neighbors from nsdSet and remove those neighbors from clfdSet
     //that do not show up in nsdSet anymore
     //This way we avoid all sorts of problems associated with various configuration of neighbors after spin flip.
     // Although it is not the fastest algorithm , it is very simple and self explanatory and given the fact that in most
     // cases number of neighbors is fairly small all those inefficiencies do not matter too much.

//     if(size1!=size2){
//          clfdSet.clear();
         ContactLocalFlexData clfdObj;
         NeighborSurfaceData nfdObj;
         set<NeighborSurfaceData>::iterator sitrND;
         set<ContactLocalFlexData>::iterator sitrCD;

         //here we insert neighbors from nsdSet into clfdSet that do not show up in clfdSet
         for(sitrND=nsdSet.begin() ; sitrND!=nsdSet.end() ; ++sitrND){
            clfdObj.neighborAddress=sitrND->neighborAddress;
            clfdObj.J=defaultContactEnergy(clfdObj.neighborAddress,_cell);
//             cerr<<"INSERTING _cell->type="<<(_cell? (int)_cell->type:0)<<" neighbor->type="<<(clfdObj.neighborAddress? (int)clfdObj.neighborAddress->type:0)<<" energy="<<clfdObj.J<<endl;

            clfdSet.insert(clfdObj); //the element will be inserted only if it is not there
         }

          //here we remove neighbors from clfd if they do not show up in nsdSet
         for(sitrCD=clfdSet.begin() ; sitrCD!=clfdSet.end() ; ){ //notice that incrementing takes place in the loop because we are erasing elements
            nfdObj.neighborAddress=sitrCD->neighborAddress;
            sitrND=nsdSet.find(nfdObj);
            if(sitrND==nsdSet.end()){ //did not find nfdObj.neighborAddress in nsdSet  - need to remove it from clfdSet 
               clfdSet.erase(sitrCD++);
            }else{
               ++sitrCD;
            }
         }
     //}

   

//    if(clfdSet.size()!=nsdSet.size()){
//       cerr<<"problem with syncing neighbors and contact energies"<<endl;
//       exit(0);
//    }
}

void ContactLocalFlexPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell){
   if(!initializadContactData && sim->getStep()==0){
      pUtils->setLock(lockPtr);
      initializeContactLocalFlexData();      
      pUtils->unsetLock(lockPtr);
   }

//    cerr<<"INSIDE field3DChange"<<endl
   if(newCell){
      updateContactEnergyData(newCell);
   }
   if(oldCell){
      updateContactEnergyData(oldCell);
   }
}


void ContactLocalFlexPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){
	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
		set<unsigned char> cellTypesSet;
	contactEnergies.clear();

	CC3DXMLElementList energyVec=_xmlData->getElements("Energy");

	for (int i = 0 ; i<energyVec.size(); ++i){

		setContactEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"), energyVec[i]->getDouble());

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

	}

	//Now that we know all the types used in the simulation we will find size of the contactEnergyArray
	vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

	int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
	size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated

	int index ;
	contactEnergyArray.clear();
	contactEnergyArray.assign(size,vector<double>(size,0.0));

	for(int i = 0 ; i < size ; ++i)
		for(int j = 0 ; j < size ; ++j){

			index = getIndex(cellTypesVector[i],cellTypesVector[j]);

			contactEnergyArray[i][j] = contactEnergies[index];

		}
		cerr<<"size="<<size<<endl;
		for(int i = 0 ; i < size ; ++i)
			for(int j = 0 ; j < size ; ++j){

				cerr<<"contact["<<i<<"]["<<j<<"]="<<contactEnergyArray[i][j]<<endl;

			}
		
			//Here I initialize max neighbor index for direct acces to the list of neighbors 
			boundaryStrategy=BoundaryStrategy::getInstance();
			maxNeighborIndex=0;

			if(_xmlData->getFirstElement("Depth")){
				maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
				//cerr<<"got here will do depth"<<endl;
			}else{
				//cerr<<"got here will do neighbor order"<<endl;
				if(_xmlData->getFirstElement("NeighborOrder")){

					maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());	
				}else{
					maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

				}

			}

			cerr<<"Contact maxNeighborIndex="<<maxNeighborIndex<<endl;
	
}




std::string ContactLocalFlexPlugin::toString(){
   return "ContactLocalFlex";
}


std::string ContactLocalFlexPlugin::steerableName(){
   return toString();
}
