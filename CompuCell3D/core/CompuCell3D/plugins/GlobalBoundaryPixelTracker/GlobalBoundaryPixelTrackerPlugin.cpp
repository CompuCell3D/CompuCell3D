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
using namespace CompuCell3D;

using namespace std;


#include "GlobalBoundaryPixelTrackerPlugin.h"

GlobalBoundaryPixelTrackerPlugin::GlobalBoundaryPixelTrackerPlugin():
pUtils(0),lockPtr(0),simulator(0),potts(0),boundaryStrategy(0),xmlData(0),boundaryPixelSetPtr(0)    
{}

GlobalBoundaryPixelTrackerPlugin::~GlobalBoundaryPixelTrackerPlugin() {
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr=0;
}




void GlobalBoundaryPixelTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  simulator=_simulator;
  potts = simulator->getPotts();

  	pUtils=simulator->getParallelUtils();
	lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);
    
  potts->registerCellGChangeWatcher(this);
  


}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GlobalBoundaryPixelTrackerPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GlobalBoundaryPixelTrackerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){


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
    boundaryPixelSetPtr=potts->getBoundaryPixelSetPtr();

}

void GlobalBoundaryPixelTrackerPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id!=LATTICE_RESIZE){
		return;
	}

	CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize&>(_event);
	Dim3D newDim = ev.newDim;
	Dim3D oldDim = ev.oldDim;
	Dim3D shiftVec = ev.shiftVec;

	
	for(std::set<Point3D>::iterator sitr =  boundaryPixelSetPtr->begin() ; sitr != boundaryPixelSetPtr->end() ; ++sitr){
            
                Point3D & pixel=const_cast<Point3D&>(*sitr);
                pixel.x+=shiftVec.x;
                pixel.y+=shiftVec.y;
                pixel.z+=shiftVec.z;
            
// 		sitr->x+=ev.shiftVec.x;
// 		sitr->y+=ev.shiftVec.y;
// 		sitr->z+=ev.shiftVec.z;
	}

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void GlobalBoundaryPixelTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell) {
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;
	
    pUtils->setLock(lockPtr); //have to lock this fcn to prevent parallel access 
    
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();
    
	Neighbor neighbor;
	Neighbor neighborOfNeighbor;
	CellG * nCell;
	CellG * nnCell;
	//handling global boundary pixel set
	//newCell section
    
    
	boundaryPixelSetPtr->insert(pt);//always insert pt to the global set of boundary pixels
    
	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
		if(!neighbor.distance){
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
        
		nCell=fieldG->get(neighbor.pt);
		if(nCell!=newCell)
			continue; //we visit only neighbors of pt that belong to newCell
        
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
				//boundaryPixelSetPtr->insert(neighbor.pt);
				break;
			}
		}
		if(!keepNeighborInBoundary){
			//handling global boundary pixel set					
			std::set<Point3D>::iterator sitr_pt=boundaryPixelSetPtr->find(neighbor.pt);
			ASSERT_OR_THROW("Could not find point:"+neighbor.pt+" in the set of all boundary pixels stored in Potts3D.cpp",sitr_pt!=boundaryPixelSetPtr->end());
            
			boundaryPixelSetPtr->erase(sitr_pt);
		}
	}	
    
	//oldCell section
    
	//handling global boundary pixel set
	//add all the neighboring pixels of pt to the boundary provided they belong to oldCell
	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
		if(!neighbor.distance){
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
		nCell=fieldG->get(neighbor.pt);
        
		if(nCell==oldCell){
			boundaryPixelSetPtr->insert(neighbor.pt);	
		}
	}
    
    pUtils->unsetLock(lockPtr); //have to lock this fcn to prevent parallel access 
    
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string GlobalBoundaryPixelTrackerPlugin::toString(){
	return "GlobalBoundaryPixelTracker";
}

std::string GlobalBoundaryPixelTrackerPlugin::steerableName(){
	return toString();
}

