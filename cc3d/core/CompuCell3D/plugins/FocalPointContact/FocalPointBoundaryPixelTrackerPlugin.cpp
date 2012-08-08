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

#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Potts3D/CellInventory.h>
#include <CompuCell3D/Field3D/Field3D.h>
#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <PublicUtilities/NumericalUtils.h>

using namespace CompuCell3D;

#include <iostream>
#include <cmath>
using namespace std;


#include "FocalPointBoundaryPixelTrackerPlugin.h"

FocalPointBoundaryPixelTrackerPlugin::FocalPointBoundaryPixelTrackerPlugin():
simulator(0),potts(0),boundaryStrategy(0),xmlData(0)    
{}

FocalPointBoundaryPixelTrackerPlugin::~FocalPointBoundaryPixelTrackerPlugin() {}




void FocalPointBoundaryPixelTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  simulator=_simulator;
  potts = simulator->getPotts();



  ///will register FocalPointBoundaryPixelTracker here
  BasicClassAccessorBase * cellFocalPointBoundaryPixelTrackerAccessorPtr=&focalPointBoundaryPixelTrackerAccessor;
   ///************************************************************************************************  
  ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
   ///************************************************************************************************  
  potts->getCellFactoryGroupPtr()->registerClass(cellFocalPointBoundaryPixelTrackerAccessorPtr);

  potts->registerCellGChangeWatcher(this);
  


}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FocalPointBoundaryPixelTrackerPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FocalPointBoundaryPixelTrackerPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){


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


}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void FocalPointBoundaryPixelTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell) {
	if (newCell==oldCell) //this may happen if you are trying to assign same cell to one pixel twice 
		return;
	

	
	WatchableField3D<CellG *> *fieldG =(WatchableField3D<CellG *> *) potts->getCellFieldG();

	Neighbor neighbor;
	Neighbor neighborOfNeighbor;
	CellG * nCell;
	CellG * nnCell;

	//check we need to create new junctions only new cell can initiate junctions
	if(newCell){
		if(focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->junctionPool>0){

		}

	}


	

	if(oldCell){
		list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->focalJunctionList;
		list<FocalPointBoundaryPixelTrackerData >::iterator litr;
		list<FocalPointBoundaryPixelTrackerData >::iterator nextLitr;
		//for (litr=focalJunctionListRef.begin() ; litr!=focalJunctionListRef.end() ; ++litr){
		litr=focalJunctionListRef.begin();
		while(litr!=focalJunctionListRef.end()){
			if(litr->pt1==pt){//pixel containing junction gets removed - need to move junction to a neighbor
				//will need to find closest point
				double minDistance=100.0;
				Point3D newJunctionPoint(0,0,0);
				for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
					neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
					if(!neighbor.distance){
						//if distance is 0 then the neighbor returned is invalid
						continue;
					}
					if(minDistance>neighbor.distance){
						minDistance=neighbor.distance;
						newJunctionPoint=neighbor.pt;
					}
				}
				//will increment iterators here
				if(minDistance==100.0 && newJunctionPoint==Point3D(0,0,0)){//no place to move junction point - move to reservoir
					focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->junctionPool++;
					//remove junction from the list
					nextLitr=focalJunctionListRef.erase(litr);
					litr=nextLitr;
				}else{
					litr->pt1=newJunctionPoint;
					CellG * partnerCell=fieldG->get(litr->pt2);

					list<FocalPointBoundaryPixelTrackerData > & partnerFocalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(partnerCell->extraAttribPtr)->focalJunctionList;
					for(list<FocalPointBoundaryPixelTrackerData >::iterator litrP=partnerFocalJunctionListRef.begin() ; litrP!=partnerFocalJunctionListRef.end() ; ++litrP){
						if(litrP->pt2==pt){
							litrP->pt2=newJunctionPoint;
						}
					}
					++litr;
				}
			}
		}

	}

	if(newCell){
		//in this situation we may move junction point from position before spin flip to pt if pt lies closer to
		//corresponding point in the partner cell
		//otherwise we leave junction point of the newCell untouched
		list<FocalPointBoundaryPixelTrackerData > & focalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->focalJunctionList;
		list<FocalPointBoundaryPixelTrackerData >::iterator litr;
		list<FocalPointBoundaryPixelTrackerData >::iterator nextLitr;
		
		//will check if pt is a point that lies closer to junction point of the partner cell
		//first check 
		double minDistance=100.0;
		Point3D newJunctionPoint(0,0,0);


		litr=focalJunctionListRef.begin();
		while(litr!=focalJunctionListRef.end()){
			
			if(dist(pt.x,pt.y,pt.z,litr->pt2.x,litr->pt2.y,litr->pt2.z)<dist(litr->pt1.x,litr->pt1.y,litr->pt1.z,litr->pt2.x,litr->pt2.y,litr->pt2.z)){
				Point3D originalJunctionPoint=litr->pt1;
				litr->pt1=pt;
				//will need to adjust pt2 for partner cell
				CellG * partnerCell=fieldG->get(litr->pt2);
				list<FocalPointBoundaryPixelTrackerData > & partnerFocalJunctionListRef=focalPointBoundaryPixelTrackerAccessor.get(partnerCell->extraAttribPtr)->focalJunctionList;
				for(list<FocalPointBoundaryPixelTrackerData >::iterator litrP=partnerFocalJunctionListRef.begin() ; litrP!=partnerFocalJunctionListRef.end() ; ++litrP){
					if(litrP->pt2==originalJunctionPoint){
						litrP->pt2=newJunctionPoint;
					}
				}

			}

			litr++;
		}

	}
	

	//if(newCell){
	//	std::map<Point3D,FocalPointBoundaryPixelTrackerData > & pixelCadMapRef=focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelCadMap;
	//	
	//	//new pixel is automatically inserted into set of boundary pixels 
	//	pair<std::map<Point3D,FocalPointBoundaryPixelTrackerData >::iterator,bool> res = pixelCadMapRef.insert(make_pair(pt,FocalPointBoundaryPixelTrackerData()));
	//	if (res.first !=pixelCadMapRef.end()){
	//		//setting new cad level
	//		res.first->second.cadLevel=0.0;
	//	}
	//	
	//	//check if new pixel will be in contact with another cell of same type
	//	bool requiredContact=false;
	//	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
	//		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
	//		if(!neighbor.distance){
	//			//if distance is 0 then the neighbor returned is invalid
	//			continue;
	//		}

	//		nCell=fieldG->get(neighbor.pt);
	//		if(nCell && nCell->type==newcell->type){
	//			requiredContact=true;
	//		}
	//	}
	//	
	//	//we visit all neighbors of the new pixel (pt) and check if the neighboring pixels are still in the boundary
	//	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
	//		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
	//		if(!neighbor.distance){
	//			//if distance is 0 then the neighbor returned is invalid
	//			continue;
	//		}

	//		nCell=fieldG->get(neighbor.pt);
	//		if(nCell!=newCell)
	//			continue; //we visit only neighbors of pt that belong to newCell

	//		bool keepNeighborInBoundary=false;
	//		//to check if neighboring pixel is still in the boundary we visit its neighbors and make sure at least
	//		//one of the pixel neighbors belongs to cell different than newCell
	//		for(unsigned int nnIdx=0 ; nnIdx <= maxNeighborIndex ; ++nnIdx ){
	//			neighborOfNeighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(neighbor.pt),nnIdx);
	//			if(!neighborOfNeighbor.distance){
	//				continue;
	//			}
	//			nnCell=fieldG->get(neighborOfNeighbor.pt);
	//			if( neighborOfNeighbor.pt!=pt && nnCell!=newCell){
	//				keepNeighborInBoundary=true;
	//				break;
	//			}
	//		}
	//		if(!keepNeighborInBoundary){
	//			std::map<Point3D,FocalPointBoundaryPixelTrackerData >::iterator mitr=pixelCadMapRef.find(neighbor.pt);	
	//			ASSERT_OR_THROW("Could not find point:"+neighbor.pt+" in the boundary of cell id: "+BasicString(newCell->id)+" type: "+BasicString((int)newCell->type),
	//			mitr!=pixelCadMapRef.end());

	//			//assigning cadherin from lost boundary pixel to new pixel if reuiredContact flag is 'on'
	//			if(requiredContact){
	//				std::map<Point3D,FocalPointBoundaryPixelTrackerData >::iterator newMitr = focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)
	//				->pixelCadMap.find(pt);
	//				if(newMitr!=focalPointBoundaryPixelTrackerAccessor.get(newCell->extraAttribPtr)->pixelCadMap.end()){	
	//					newMitr->second.cadLevel+=mitr->second.cadLevel;
	//				}
	//			}else{//if required contatc does not take plase return cadherin to reservioir
	//				focalPointBoundaryPixelTrackerAccessor.get(_cell->extraAttribPtr)->reservoir+=mitr->second.cadLevel;
	//			}
	//			pixelCadMapRef.erase(mitr);

	//		}

	//	}

	//}

	//if(oldCell){
	//	//first erase pt from set of boundary pixels
	//	std::map<Point3D,FocalPointBoundaryPixelTrackerData > & pixelCadMapRef=focalPointBoundaryPixelTrackerAccessor.get(oldCell->extraAttribPtr)->pixelCadMap;
	//	std::map<Point3D,FocalPointBoundaryPixelTrackerData >::iterator mitr;
	//	mitr=pixelCadMapRef.find(pt);
	//	//ASSERT_OR_THROW("Could not find point:"+pt+" inside cell of id: "+BasicString(oldCell->id)+" type: "+BasicString((int)oldCell->type),
	//	//mitr!=pixelCadMapRef.end());

	//	//store cad level of the lost pixel
	//	double lostPixelCadLevel=mitr->second.cadLevel;

	//	if(mitr!=pixelCadMapRef.end()){//means that pt belongs to oldCell border
	//		pixelCadMapRef.erase(mitr);
	//	}
	//	//handling global boundary pixel set
	//	//if(boundaryPixelSetPtr ){//always insert pt to the global set of boundary pixels
	//	//	boundaryPixelSetPtr->insert(pt);
	//	//}

	//	//add all the neighboring pixels of pt to the boundary provided they belong to oldCell
	//	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
	//		neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
	//		if(!neighbor.distance){
	//			//if distance is 0 then the neighbor returned is invalid
	//			continue;
	//		}
	//		nCell=fieldG->get(neighbor.pt);

	//		if(nCell==oldCell){
	//			pair<std::map<Point3D,FocalPointBoundaryPixelTrackerData >::iterator,bool> res= pixelCadMapRef.insert(make_pair(neighbor.pt,FocalPointBoundaryPixelTrackerData()));	
	//			if (res.first !=pixelCadMapRef.end()){
	//				//setting new cad level
	//				res.first->second.cadLevel=0.0;
	//			}
	//		}


	//	}
	//}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FocalPointBoundaryPixelTrackerPlugin::distributeReservoir(double _fraction, CellG * _cell){
		if(!_cell || _fraction>1.0 || _fraction<=0.0)
			return;
		std::map<Point3D,FocalPointBoundaryPixelTrackerData > & pixelCadMapRef=focalPointBoundaryPixelTrackerAccessor.get(_cell->extraAttribPtr)->pixelCadMap;
		std::map<Point3D,FocalPointBoundaryPixelTrackerData >::iterator mitr;
		double & reservoirValue=focalPointBoundaryPixelTrackerAccessor.get(_cell->extraAttribPtr)->reservoir;
		double cadLevelShare=_fraction*reservoirValue/(pixelCadMapRef.size());
		
		for(mitr=pixelCadMapRef.begin() ; mitr!=pixelCadMapRef.end() ; ++mitr){
			mitr->second.cadLevel=cadLevelShare;
			reservoirValue-=cadLevelShare;
		}

		
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string FocalPointBoundaryPixelTrackerPlugin::toString(){
	return "FocalPointBoundaryPixelTracker";
}

std::string FocalPointBoundaryPixelTrackerPlugin::steerableName(){
	return toString();
}

