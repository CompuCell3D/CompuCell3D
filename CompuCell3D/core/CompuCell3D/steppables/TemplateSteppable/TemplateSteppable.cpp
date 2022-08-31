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
#include<core/CompuCell3D/CC3DLogger.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
// // // #include <CompuCell3D/Potts3D/TypeTransition.h>
// // // #include <CompuCell3D/Field3D/Point3D.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>

using namespace CompuCell3D;



// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>

// // // #include <string>
// // // #include <map>
// // // #include <sstream>
// // // #include <iostream>

using namespace std;


#include <CompuCell3D/plugins/NeighborTracker/NeighborTrackerPlugin.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include "TemplateSteppable.h"

TemplateSteppable::TemplateSteppable() :
potts(0), pifname("") {}

TemplateSteppable::TemplateSteppable(string filename) :
potts(0), pifname(filename) {}

void TemplateSteppable::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

   bool pluginAlreadyRegisteredFlag;
   Plugin *plugin=Simulator::pluginManager.get("VolumeTracker",&pluginAlreadyRegisteredFlag); //this will load VolumeTracker plugin if it is not already loaded
  if(!pluginAlreadyRegisteredFlag)
      plugin->init(simulator);

   NeighborTrackerPlugin * neighborTrackerPluginPtr=(NeighborTrackerPlugin*)(Simulator::pluginManager.get("NeighborTracker",&pluginAlreadyRegisteredFlag));
   if (!pluginAlreadyRegisteredFlag){
      neighborTrackerPluginPtr->init(simulator);
      ASSERT_OR_THROW("NeighborTracker plugin not initialized!", neighborTrackerPluginPtr);
      neighborTrackerAccessorPtr=neighborTrackerPluginPtr->getNeighborTrackerAccessorPtr();
      ASSERT_OR_THROW("neighborAccessorPtr  not initialized!", neighborTrackerAccessorPtr);
   }

	pifname=_xmlData->getFirstElement("PIFName")->getText();
    Log(LOG_DEBUG) << "PIFNAME: " << pifname;
	potts = simulator->getPotts();\
	cellInventoryPtr = & potts->getCellInventory();

	boundaryStrategy=BoundaryStrategy::getInstance();


}
void TemplateSteppable::start(){}

void TemplateSteppable::step(const unsigned int currentStep) {
    CellInventory::cellInventoryIterator cInvItr;
	CellG * cell;
	CellG * nCell;

	Neighbor neighbor;

    std::set<NeighborSurfaceData> * neighborData;
    std::set<NeighborSurfaceData >::iterator sitr;

	for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr )
	{
		cell=cellInventoryPtr->getCell(cInvItr);
        Log(LOG_DEBUG) << "Cell Volume: " << cell->volume;
		neighborData = &(neighborTrackerAccessorPtr->get(cell->extraAttribPtr)->cellNeighbors);
		      for(sitr=neighborData->begin() ; sitr != neighborData->end() ; ++sitr) {

		          nCell= sitr->neighborAddress;
		          if(nCell){
		              int nType = (int)nCell->type;
                      Log(LOG_DEBUG) << "Neighbor: " << (int)nCell->id << " Type: " << nType;
		          }
		      }
	}

   WatchableField3D<CellG *> *cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
   Dim3D dim = cellFieldG->getDim();
   Point3D pt;

   int maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(2);
   Log(LOG_DEBUG) << "Number of neighbors: " << maxNeighborIndex;
   for (int x = 0 ; x < dim.x ; ++x) {
      for (int y = 0 ; y < dim.y ; ++y) {
         for (int z = 0 ; z < dim.z ; ++z){
            pt.x=x;
            pt.y=y;
            pt.z=z;
            cell=cellFieldG->get(pt);
            Log(LOG_DEBUG) << "point: (" << x << ", " << y << ", " << z << ") \n";
            for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
                neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
                Log(LOG_DEBUG) << "neighbor point: (" << neighbor.pt.x << ", " << neighbor.pt.y << ", " << neighbor.pt.z << ") \n";
                nCell = cellFieldG->get(neighbor.pt);
                if(nCell){
                    Log(LOG_DEBUG) << "Neighbor point: " << nCell->id;
                }
            }
         }
      }
   }

}



