

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>

using namespace CompuCell3D;

#include <iostream>
#include <cmath>
using namespace std;


#include "BoundaryPixelTrackerPlugin.h"

BoundaryPixelTrackerPlugin::BoundaryPixelTrackerPlugin():
simulator(0),potts(0),boundaryStrategy(0),xmlData(0)    
{}

BoundaryPixelTrackerPlugin::~BoundaryPixelTrackerPlugin() {}




void BoundaryPixelTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  simulator=_simulator;
  potts = simulator->getPotts();



  ///will register BoundaryPixelTracker here
  BasicClassAccessorBase * cellBoundaryPixelTrackerAccessorPtr=&boundaryPixelTrackerAccessor;
   ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
   ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(cellBoundaryPixelTrackerAccessorPtr);

  potts->registerCellGChangeWatcher(this);

  boundaryStrategy=BoundaryStrategy::getInstance();
  maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BoundaryPixelTrackerPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BoundaryPixelTrackerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){


	//Here I initialize max neighbor index for direct acces to the list of neighbors 
	boundaryStrategy=BoundaryStrategy::getInstance();
	maxNeighborIndex=0;

	if (!_xmlData){ //this happens if plugin is loaded directly as a dependency from another plugin
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
		return;
	}


	if(_xmlData->getFirstElement("Depth")){
		maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
	}else{
		if(_xmlData->getFirstElement("NeighborOrder")){

			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());	
		}else{
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

		}

	}


}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void BoundaryPixelTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell) {
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;
	
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();

	Neighbor neighbor;
	Neighbor neighborOfNeighbor;
	CellG * nCell;
	CellG * nnCell;
	
	
	

	if(newCell){
		std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelSet;
		bool ptInsertedAsBoundary=false;
		//new pixel is NOT automatically inserted into set of boundary pixels  - it will be inserted after we determine that indeed it belongs to the boundary
		//pixelSetRef.insert(BoundaryPixelTrackerData(pt));



		//we visit all neighbors of the new pixel (pt) and check if the neighboring pixels are still in the boundary
		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			nCell=fieldG->get(neighbor.pt);
			if(nCell!=newCell){
                if (!ptInsertedAsBoundary){
                    pixelSetRef.insert(BoundaryPixelTrackerData(pt)); // when we determine that new pixel touches cell different than newCell only then we will insert it as a boundary pixel
                    ptInsertedAsBoundary=true;
                }
				continue; //we visit only neighbors of pt that belong to newCell
            }
            
			bool keepNeighborInBoundary=false;
			//to check if neighboring pixel is still in the boundary we visit its neighbors and make sure at least
			//one of the pixel neighbors belongs to cell different than newCell
			for(unsigned int nnIdx=0 ; nnIdx <= maxNeighborIndex ; ++nnIdx ){
				neighborOfNeighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(neighbor.pt),nnIdx);
				if(!neighborOfNeighbor.distance){
					continue;
				}
				nnCell=fieldG->get(neighborOfNeighbor.pt);
				if( neighborOfNeighbor.pt!=pt && nnCell!=newCell){
					keepNeighborInBoundary=true;
					break;
				}
			}
			if(!keepNeighborInBoundary){
				std::set<BoundaryPixelTrackerData >::iterator sitr=pixelSetRef.find(BoundaryPixelTrackerData(neighbor.pt));	
				ASSERT_OR_THROW("Could not find point:"+neighbor.pt+" in the boundary of cell id: "+BasicString(newCell->id)+" type: "+BasicString((int)newCell->type),
				sitr!=pixelSetRef.end());
				pixelSetRef.erase(sitr);
			}

		}

	}

	if(oldCell){
		//first erase pt from set of boundary pixels
		std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelSet;
		std::set<BoundaryPixelTrackerData >::iterator sitr;
		sitr=pixelSetRef.find(BoundaryPixelTrackerData(pt));
		//ASSERT_OR_THROW("Could not find point:"+pt+" inside cell of id: "+BasicString(oldCell->id)+" type: "+BasicString((int)oldCell->type),
		//sitr!=pixelSetRef.end());

		if(sitr!=pixelSetRef.end()){//means that pt belongs to oldCell border
			pixelSetRef.erase(sitr);
		}
		//handling global boundary pixel set
		//if(boundaryPixelSetPtr ){//always insert pt to the global set of boundary pixels
		//	boundaryPixelSetPtr->insert(pt);
		//}

		//add all the neighboring pixels of pt to the boundary provided they belong to oldCell
		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}
			nCell=fieldG->get(neighbor.pt);

			if(nCell==oldCell){
				pixelSetRef.insert(BoundaryPixelTrackerData(neighbor.pt));	
			}

		}
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string BoundaryPixelTrackerPlugin::toString(){
	return "BoundaryPixelTracker";
}

std::string BoundaryPixelTrackerPlugin::steerableName(){
	return toString();
}

