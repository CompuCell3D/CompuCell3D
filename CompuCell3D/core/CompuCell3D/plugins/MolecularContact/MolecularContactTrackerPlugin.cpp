
 

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
      cerr << "ID: " << oldCell->id << " Type: " << (int)oldCell->type << " Address: " << oldCell << endl;
//          sleep(5);
      neighborData = &(neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr)->cellNeighbors);
//          neighborTrackerAccessorPtr->get(oldCell->extraAttribPtr);
      set<MolecularContactTrackerData>::iterator PlasSetitr;
      set<MolecularContactTrackerData>::iterator tmpPlasSetitr;
      set<MolecularContactTrackerData> * molecularcontactNeighborsPtr=&molecularcontactTrackerAccessor.get(oldCell->extraAttribPtr)->molecularcontactNeighbors;
      set<MolecularContactTrackerData> OGmolecularcontactNeighborsPtr=molecularcontactTrackerAccessor.get(oldCell->extraAttribPtr)->molecularcontactNeighbors;
      set<MolecularContactTrackerData> * molecularcontactNeighborsTmpPtr;
//       cerr << "Before Size of Set: " << molecularcontactNeighborsPtr->size() << endl;
      molecularcontactNeighborsPtr->clear();
      for(sitr=neighborData->begin() ; sitr != neighborData->end() ; ++sitr){
         nCell= sitr->neighborAddress;
         cerr << "\t NeigbhorID: " << sitr->neighborAddress    << endl;
         if(nCell) {
            if(molecularcontactTypes.find(nCell->type)==endSitr){
               cerr << "\t Type not inlucded in MolecularContact\n";
               cerr << "\tID: " << nCell->id << " Type: " << (int)nCell->type << endl;
            }
            else{
            cerr << "\t Inserting NeigbhorID: " << sitr->neighborAddress << " Type: " << (int)nCell->type << endl;
            cerr << "\t Before Inserting Size of Set: " << molecularcontactNeighborsPtr->size() << endl;
               molecularcontactNeighborsPtr->insert(MolecularContactTrackerData(nCell));
               molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(nCell->extraAttribPtr)->molecularcontactNeighbors;
               molecularcontactNeighborsTmpPtr->insert(MolecularContactTrackerData(oldCell));
               cerr << "\t After Inserting" << endl;
            }
         }
      }
      /*for(PlasSetitr=molecularcontactNeighborsPtr->begin() ; PlasSetitr != molecularcontactNeighborsPtr->end() ; ++PlasSetitr){
         molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(oldCell->extraAttribPtr)->molecularcontactNeighbors;
         cerr << "Set NeigbhorID: " << PlasSetitr->neighborAddress << " Type: " << (int)PlasSetitr->neighborAddress->type << endl;
//             molecularcontactNeighborsTmpPtr->insert(MolecularContactTrackerData(oldCell));
      }
//       cerr << "After Size of Set: " << molecularcontactNeighborsPtr->size() << endl;
//       cerr << "OG Size of Set: " << OGmolecularcontactNeighborsPtr.size() << endl;
      
      for(PlasSetitr=OGmolecularcontactNeighborsPtr.begin() ; PlasSetitr != OGmolecularcontactNeighborsPtr.end() ; ++PlasSetitr){
         cerr << "OGSet NeigbhorID: " << PlasSetitr->neighborAddress << " Type: " << (int)PlasSetitr->neighborAddress->type << " TargetLength: " << PlasSetitr->targetLength << endl;
          molecularcontactNeighborsTmpPtr=&molecularcontactTrackerAccessor.get(PlasSetitr->neighborAddress->extraAttribPtr)->molecularcontactNeighbors;
          for(tmpPlasSetitr=molecularcontactNeighborsTmpPtr->begin() ; tmpPlasSetitr != molecularcontactNeighborsTmpPtr->end() ; ++tmpPlasSetitr){
             cerr << "\t tmp NeigbhorID: " << tmpPlasSetitr->neighborAddress << " Type: " << (int)tmpPlasSetitr->neighborAddress->type << endl;
          }
                
//             molecularcontactNeighborsTmpPtr->insert(MolecularContactTrackerData(oldCell));
      }*/
//       std::set<unsigned char>::iterator typesit; 
//       for(typesit=molecularcontactTypes.begin() ; typesit != molecularcontactTypes.end();++typesit) {
//          cerr << "Type: " << (int)*typesit << endl;
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
