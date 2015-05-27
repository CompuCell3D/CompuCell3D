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
// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>

using namespace CompuCell3D;

// // // #include <iostream>
// // // #include <cmath>
using namespace std;


#include "BoundaryPixelTrackerPlugin.h"

BoundaryPixelTrackerPlugin::BoundaryPixelTrackerPlugin():
simulator(0),potts(0),boundaryStrategy(0),xmlData(0),maxNeighborIndex(0),neighborOrder(1)
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
void BoundaryPixelTrackerPlugin::handleEvent(CC3DEvent & _event){
	if (_event.id!=LATTICE_RESIZE){
		return;
	}
	cerr<<"INSIDE BOUNDARY PIXEL TRACKER EVEN HANDLER"<<endl;
	CC3DEventLatticeResize ev = static_cast<CC3DEventLatticeResize&>(_event);

	Dim3D shiftVec=ev.shiftVec;

    CellInventory &cellInventory = potts->getCellInventory();
    CellInventory::cellInventoryIterator cInvItr;
    CellG * cell;
        
    for(cInvItr=cellInventory.cellInventoryBegin() ; cInvItr !=cellInventory.cellInventoryEnd() ;++cInvItr )
    {
		cell=cInvItr->second;
		set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessor.get(cell->extraAttribPtr)->pixelSet;
		for (set<BoundaryPixelTrackerData >::iterator sitr=pixelSetRef.begin() ; sitr != pixelSetRef.end() ; ++sitr ){
                    
                        Point3D & pixel=const_cast<Point3D&>(sitr->pixel);
                        pixel.x+=shiftVec.x;
                        pixel.y+=shiftVec.y;
                        pixel.z+=shiftVec.z;
                    
// 			sitr->pixel.x+=shiftVec.x;
// 			sitr->pixel.y+=shiftVec.y;
// 			sitr->pixel.z+=shiftVec.z;
		}



    }

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

    

        // when user specifies depth , fetching of boundary for neighbor order might not work properly - not a big deal because almost nobody is using the Depth tag
        neighborOrder = 0;

		//cerr<<"got here will do depth"<<endl;
	}else{
		//cerr<<"got here will do neighbor order"<<endl;
		if(_xmlData->getFirstElement("NeighborOrder")){

            neighborOrder =  _xmlData->getFirstElement("NeighborOrder")->getUInt();
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);	
            

		}else{
            neighborOrder = 1;
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);
		}
    

        std::set<int>  extraBoundariesNeighborOrderSet;
        CC3DXMLElementList extraBoundariesVec=_xmlData->getElements("ExtraBoundary");

	    for (int i = 0 ; i<extraBoundariesVec.size(); ++i){
            extraBoundariesNeighborOrderSet.insert(extraBoundariesVec[i]->getAttributeAsInt("NeighborOrder"));

	    }
        
        extraBoundariesNeighborOrder.assign(extraBoundariesNeighborOrderSet.begin(),extraBoundariesNeighborOrderSet.end());
        extraBoundariesMaxNeighborIndex.assign(extraBoundariesNeighborOrder.size(),0); //allocating memory in the extraBoundariesMaxNeighborIndex vector

        cerr<<"extraBoundariesNeighborOrder.size()="<<extraBoundariesNeighborOrder.size()<<endl;
        for (unsigned int i = 0 ; i <  extraBoundariesNeighborOrder.size() ; ++i){
            extraBoundariesMaxNeighborIndex [i] = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder( extraBoundariesNeighborOrder[i] );
        }

        for (unsigned int i = 0 ; i <  extraBoundariesMaxNeighborIndex.size() ; ++i){
            cerr<<"i="<<i<<" extraBoundariesMaxNeighborIndex[i]="<<extraBoundariesMaxNeighborIndex[i]<<endl;
        }
                
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BoundaryPixelTrackerPlugin::updateBoundaryPixels(const Point3D &pt, CellG *newCell,CellG *oldCell, int indexOfExtraBoundary) 
{

	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;
	
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();

	Neighbor neighbor;
	Neighbor neighborOfNeighbor;
	CellG * nCell;
	CellG * nnCell;
	
    //for (unsigned int i = 0 ; i <  extraBoundariesMaxNeighborIndex.size() ; ++i){
    //    cerr<<"i="<<i<<" extraBoundariesMaxNeighborIndex[i]="<<extraBoundariesMaxNeighborIndex[i]<<endl;
    //}
	
	

	if(newCell){
        
        std::set<BoundaryPixelTrackerData > * pixelSetPtr =  0;
        int maxNI=-1; //maxNI stands for maxNeighborIndex

        if (indexOfExtraBoundary >= 0 ){
            //pixelSetMap is indexed by neighbor order
            pixelSetPtr = & (boundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelSetMap[extraBoundariesNeighborOrder[indexOfExtraBoundary] ]); 
            
            maxNI = extraBoundariesMaxNeighborIndex [indexOfExtraBoundary] ;     
            
        }else{        
            pixelSetPtr = & boundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelSet;
            maxNI = maxNeighborIndex;
        }

        std::set<BoundaryPixelTrackerData > & pixelSetRef = *pixelSetPtr; 

		//std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelSet;

		bool ptInsertedAsBoundary=false;

		//new pixel is NOT automatically inserted into set of boundary pixels  - it will be inserted after we determine that indeed it belongs to the boundary
		//pixelSetRef.insert(BoundaryPixelTrackerData(pt));

		//we visit all neighbors of the new pixel (pt) and check if the neighboring pixels are still in the boundary
		for(unsigned int nIdx=0 ; nIdx <= maxNI ; ++nIdx ){
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
			for(unsigned int nnIdx=0 ; nnIdx <= maxNI ; ++nnIdx ){
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

        std::set<BoundaryPixelTrackerData > * pixelSetPtr =  0;
        int maxNI=-1; //maxNI stands for maxNeighborIndex

        if (indexOfExtraBoundary >= 0 ){
            //pixelSetMap is indexed by neighbor order
            pixelSetPtr = & (boundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelSetMap[extraBoundariesNeighborOrder[indexOfExtraBoundary] ]); 
            maxNI = extraBoundariesMaxNeighborIndex [indexOfExtraBoundary] ;
        }else{
            pixelSetPtr = & boundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelSet;
            maxNI = maxNeighborIndex;
        }



        std::set<BoundaryPixelTrackerData > & pixelSetRef = *pixelSetPtr; 


		//first erase pt from set of boundary pixels
		//std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelSet;
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
void BoundaryPixelTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell) {
    
    updateBoundaryPixels(pt, newCell,oldCell); //we always call update function for the default case wchic is for NeighborOrder =1  assuming the user does not overwrite it

    
    //WHen user requests tracking of other boundaries with additional neighbor orders we update them here
    for (unsigned int indexOfExtraBoundary  = 0 ; indexOfExtraBoundary < extraBoundariesMaxNeighborIndex.size() ; ++indexOfExtraBoundary){

        updateBoundaryPixels(pt, newCell,oldCell, indexOfExtraBoundary);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<BoundaryPixelTrackerData > * BoundaryPixelTrackerPlugin::getPixelSetForNeighborOrderPtr( CellG * _cell, int _neighborOrder){
    if (_neighborOrder<=0){
        return 0;
    }

    // cerr<<"_neighborOrder="<<_neighborOrder<<" this->neighborOrder="<<this->neighborOrder<<endl;
    if (_neighborOrder == this->neighborOrder){
        //return coundary calculated by default
        return & boundaryPixelTrackerAccessor.get(_cell->extraAttribPtr)->pixelSet;
    }

    //if default pixel set does not match request search pixelSetMap

    std::map<int, std::set<BoundaryPixelTrackerData > > &pixelSetMap = boundaryPixelTrackerAccessor.get(_cell->extraAttribPtr)->pixelSetMap;


    std::map<int, std::set<BoundaryPixelTrackerData > >::iterator mitr = pixelSetMap.find(_neighborOrder);
    if (mitr != pixelSetMap.end() ){
        return & mitr->second;
    }
    return 0;
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string BoundaryPixelTrackerPlugin::toString(){
	return "BoundaryPixelTracker";
}

std::string BoundaryPixelTrackerPlugin::steerableName(){
	return toString();
}

