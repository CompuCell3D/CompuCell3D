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

#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/Field3D/Neighbor.h>

#include "SimulationBox.h"
#include <iostream>
#include <cmath>

//indexing macro
#define PT2IDX(pt) (pt.x + ((pt.y + (pt.z * lookupLatticeDim.y)) * lookupLatticeDim.x))

using namespace std;
using namespace CenterModel;






SimulationBox::SimulationBox():lookupLatticePtr(0),maxNeighborOrder(2)
{}


SimulationBox::~SimulationBox(){

	if (!lookupLatticePtr)
		return;
	
	
	CompuCell3D::Point3D pt;
	for (pt.x = 0 ; pt.x < lookupLatticeDim.x ; ++pt.x)
		for (pt.y = 0 ; pt.y < lookupLatticeDim.y ; ++pt.y)
			for (pt.z = 0 ; pt.z < lookupLatticeDim.z ; ++pt.z){

				lookupLatticePtr->set(pt,new CellSorterCM());
			}

	delete lookupLatticePtr;


}


void  SimulationBox::setDim(double _x,double _y,double _z) {
	//dim.fX=(ceil(fabs(_x)));
	//dim.fY=(ceil(fabs(_y)));
	//dim.fZ=(ceil(fabs(_z)));

	dim.fX=_x;
	dim.fY=_y;
	dim.fZ=_z;

}


void SimulationBox::setGridSpacing(double _x,double _y,double _z){
	//gridSpacing.fX=(ceil(fabs(_x)));
	//gridSpacing.fY=(ceil(fabs(_y)));
	//gridSpacing.fZ=(ceil(fabs(_z)));

	gridSpacing.fX=_x;
	gridSpacing.fY=_y;
	gridSpacing.fZ=_z;

}

void SimulationBox::setBoxSpatialProperties(Vector3 & _dim, Vector3 & _gridSpacing){
	setBoxSpatialProperties(_dim.fX,_dim.fY,_dim.fZ,_gridSpacing.fX,_gridSpacing.fY,_gridSpacing.fZ);	
}

void SimulationBox::setBoxSpatialProperties(double _x,double _y,double _z,double _xs,double _ys,double _zs){

	dim.fX=_x;
	dim.fY=_y;
	dim.fZ=_z;

	gridSpacing.fX=_xs;
	gridSpacing.fY=_ys;
	gridSpacing.fZ=_zs;

	inverseGridSpacing.fX=1.0/gridSpacing.fX;
	inverseGridSpacing.fY=1.0/gridSpacing.fY;
	inverseGridSpacing.fZ=1.0/gridSpacing.fZ;

	double xratio=1.0,yratio=1.0,zratio=1.0;

	xratio=dim.fX/gridSpacing.fX;
	yratio=dim.fY/gridSpacing.fY;
	zratio=dim.fZ/gridSpacing.fZ;
	//cerr<<"xratio="<<xratio<<endl;
	//cerr<<"yratio="<<yratio<<endl;
	//cerr<<"zratio="<<zratio<<endl;


	lookupLatticeDim.x=static_cast<short>(floor(fabs(xratio)))+1;
	lookupLatticeDim.y=static_cast<short>(floor(fabs(yratio)))+1;
	lookupLatticeDim.z=static_cast<short>(floor(fabs(zratio)))+1;


	// // // //once we figured out lookupLAtticeDimension we will have to adjust box size to make sure that the size of each lookup box  is the same
	// // // dim.fX=_x;
	// // // dim.fY=_y;
	// // // dim.fZ=_z;



	lookupLatticePtr=new CompuCell3D::Field3DImpl<CellSorterCM*>(lookupLatticeDim,static_cast<CellSorterCM*>(0));


	CompuCell3D::Point3D pt;
	for (pt.x = 0 ; pt.x < lookupLatticeDim.x ; ++pt.x)
		for (pt.y = 0 ; pt.y < lookupLatticeDim.y ; ++pt.y)
			for (pt.z = 0 ; pt.z < lookupLatticeDim.z ; ++pt.z){

				lookupLatticePtr->set(pt,new CellSorterCM());
			}


			//for (int i = 0 ; i < lookupLatticeDim.x ; ++i){
			//	pt.x=x;

			//	lookupLatticePtr->get(pt)
			//}

			//CompuCell3D::BoundaryStrategy::destroy();
			CompuCell3D::BoundaryStrategy::instantiate("noflux","noflux","noflux","regular",0,0,"none",CompuCell3D::LatticeType::SQUARE_LATTICE);
			boundaryStrategy=CompuCell3D::BoundaryStrategy::getInstance();
			//cerr<<"boundaryStrategy="<<boundaryStrategy<<endl;
			//cerr<<"lookupLatticeDim="<<lookupLatticeDim<<endl;
			//cerr<<"maxOffset="<<boundaryStrategy->getMaxOffset()<<endl;
			//cerr<<"getMaxDistance="<<boundaryStrategy->getMaxDistance()<<endl;
			
			//BoundaryStrategy is sensitive to mimimum dimension value -if the value is too small it will not work properly...
			//CompuCell3D::Dim3D newDim(10,10,10);
			boundaryStrategy->setMaxDistance(2.0);
			boundaryStrategy->setDim(lookupLatticeDim);
			//boundaryStrategy->setDim(newDim);


			//cerr<<"maxNeighborOrder="<<maxNeighborOrder<<endl;
            //in 3D maxNeighborOrder=3 - we need to cover all "corners" of central pixel
            //in2D maxNeighborOrder=2
            maxNeighborOrder=3;
			maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(maxNeighborOrder);	//maxNeighborOrder is 2 here because we will search for interaction pairs in grid locations which are up to 2nd nearest neighbors
			//cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;

			neighborsVec.clear();
			neighborsVec.assign(maxNeighborIndex+2,CompuCell3D::Point3D());


}

std::pair<std::vector<CompuCell3D::Point3D>,unsigned int> SimulationBox::getLatticeLocationsWithinInteractingRange(CellCM *_cell){

	CompuCell3D::Point3D pt=getCellLatticeLocation(_cell);

	std::pair<std::vector<CompuCell3D::Point3D>,unsigned int> neighborListCountPair;
	//cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;
	neighborListCountPair.first=std::vector<CompuCell3D::Point3D>(maxNeighborIndex+2);
	neighborListCountPair.second=0;

	//cerr<<"neighborListCountPair.first.size()="<<neighborListCountPair.first.size()<<endl;
	int validNeighbors=0;
	neighborListCountPair.first[validNeighbors++]=pt; //lattice location with current cell has to be searched as well

	CompuCell3D::Neighbor neighbor;

	
	//cerr<<"maxNeighborIndex="<<maxNeighborIndex<<endl;
	for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
		neighbor=boundaryStrategy->getNeighborDirect(pt,nIdx);
		if(!neighbor.distance){
			//if distance is 0 then the neighbor returned is invalid
			continue;
		}
		
		neighborListCountPair.first[validNeighbors++]=neighbor.pt;
		
	}
	neighborListCountPair.second=neighborListCountPair.first.size();
	return neighborListCountPair;


}

void SimulationBox::setLookupLatticeDim(short _x,short _y, short _z){

	lookupLatticeDim.x=_x;
	lookupLatticeDim.y=_y;
	lookupLatticeDim.z=_z;

}

void SimulationBox::updateCellLookup(CellCM * _cell){
	
	//Vector3 lookupPosition=_cell->position*inverseGridSpacing;

	CompuCell3D::Point3D pt=getCellLatticeLocation(_cell);

	//cerr<<"_cell->position="<<_cell->position<<endl;
	//cerr<<"inverseGridSpacing="<<inverseGridSpacing<<endl;
	//cerr<<"lookupPosition="<<lookupPosition<<endl;

	//cerr<<"pt="<<pt<<endl;	
	long newLookupIndex=PT2IDX(pt);
	long oldLookupIndex=_cell->lookupIdx;
	if (oldLookupIndex!=newLookupIndex){
		//fetch lookup set  corresponding to _cell->lookupIdx
		
		CellSorterCM * csPtr=lookupLatticePtr->getByIndex(oldLookupIndex);
		if (csPtr){
			//making sure that cell's lookup index is sane 
			//cerr<<"csPtr="<<csPtr<<endl;

			set<CellSorterDataCM> & oldSorterSetRef=lookupLatticePtr->getByIndex(oldLookupIndex)->sorterSet;
			
			//cerr<<"oldSorterSetRef.size()="<<oldSorterSetRef.size()<<endl;		

			set<CellSorterDataCM>::iterator oldSitr=oldSorterSetRef.find(CellSorterDataCM(_cell));

			

			if(oldSitr!=oldSorterSetRef.end()){//cell is in the location pointed by lookupIdx and we remove it
				oldSorterSetRef.erase(oldSitr);
			}

		}

		//now insert cell with updated lookup index ()
		_cell->lookupIdx=newLookupIndex;
		//cerr<<"newLookupIndex="<<newLookupIndex<<endl;
		set<CellSorterDataCM> & newSorterSetRef=lookupLatticePtr->getByIndex(newLookupIndex)->sorterSet;
		set<CellSorterDataCM>::iterator newSitr;
		newSorterSetRef.insert(CellSorterDataCM(_cell));
		//cerr<<"newSorterSetRef.size()="<<newSorterSetRef.size()<<endl;

	}

	
	
}

InteractionRangeIterator SimulationBox::getInteractionRangeIterator(CellCM *_cell){
	InteractionRangeIterator itr;
	itr.sbPtr=this;
	itr.cell=_cell;
	itr.initialize();
	return itr;	
}



InteractionRangeIterator::InteractionRangeIterator():sbPtr(0),counter(0),lookupFieldPtr(0),currentSorterSetPtr(0){
	
}


void InteractionRangeIterator::initialize(){
	neighborListPair=sbPtr->getLatticeLocationsWithinInteractingRange(cell);


	lookupFieldPtr=const_cast<SimulationBox::LookupField_t*>(&sbPtr->getLookupFieldRef());

	

	//counter=0;

	//currentSorterSetPtr=&lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet;
	//sitrBegin = lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet.begin();
	//sitrCurrent= sitrBegin;
	//sitrEnd=lookupFieldPtr->get(neighborListPair.first[neighborListPair.second-1])->sorterSet.end();
	
}
InteractionRangeIterator& InteractionRangeIterator::begin(){

	counter=0;

	currentSorterSetPtr=&lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet;
	//sitrBegin = lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet.begin();
	sitrCurrent= lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet.begin();
	currentEnd=lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet.end();

	//cerr<<"begin sitrCurrent->cell="<<sitrCurrent->cell<<endl;
	
	//sitrEnd=lookupFieldPtr->get(neighborListPair.first[neighborListPair.second-1])->sorterSet.end();
	//bool flag= (sitrCurrent==sitrEnd);
	//cerr<<"begin flag="<<flag<<endl;
	return *this;

}

InteractionRangeIterator& InteractionRangeIterator::end(){

	counter=neighborListPair.second-1;

	//currentSorterSetPtr=&lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet;
	//sitrBegin = lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet.end();
	
	//sitrEnd=lookupFieldPtr->get(neighborListPair.first[neighborListPair.second-1])->sorterSet.end();
	//sitrCurrent= sitrEnd;
	sitrCurrent= lookupFieldPtr->get(neighborListPair.first[neighborListPair.second-1])->sorterSet.end();
	return *this;
}




InteractionRangeIterator & InteractionRangeIterator::operator ++(){

	currentSorterSetPtr=&lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet;
	
	//cerr<<"ENTRY currentSorterSetPtr="<<currentSorterSetPtr<<endl;

	if (++sitrCurrent != currentSorterSetPtr->end()){
		
	}else{
		//cerr<<"\n\n\n\n HAS TO LOOK FOR NEW SET"<<endl;
		//find next non empty set
		currentSorterSetPtr=0;
		//cerr<<"counter="<<counter<<endl;
		std::set<CellSorterDataCM> *tmpSorterSetPtr;

		for ( ++counter; counter < neighborListPair.second ;++counter){
			tmpSorterSetPtr=&lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet;
			//cerr<<"loopCOunter="<<counter<<endl;
			if (tmpSorterSetPtr->size()){
				currentSorterSetPtr=tmpSorterSetPtr;
				//cerr<<"FOUND NEW SET WITH "<<tmpSorterSetPtr->size()<<endl;
				break;

			}
		}
		if (counter>=neighborListPair.second){
			--counter;
		}
		//cerr<<"AFTER LOOP COUNTER="<<counter<<endl;
		if (!currentSorterSetPtr){
			counter=neighborListPair.second-1;
			sitrCurrent=lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet.end();
			currentEnd=sitrCurrent;
			currentSorterSetPtr=&lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet;
			//cerr<<"DID NOT FIND SORTER SET SETTING TO LAST AVAILABLE="<<currentSorterSetPtr<<endl;
		}else{
			//cerr<<"NEW SET SIZE="<<currentSorterSetPtr->size()<<endl;
			sitrCurrent=currentSorterSetPtr->begin();
			currentEnd=currentSorterSetPtr->end();
			//cerr<<"currentSorterSetPtr="<<currentSorterSetPtr<<endl;
		}

	}

	//cerr<<"returning iterator currentSorterSetPtr="<<currentSorterSetPtr<<endl;
	//cerr<<"previous cellsortrerSet="<<&lookupFieldPtr->get(neighborListPair.first[counter])->sorterSet<<endl;
	return *this;
}

CellCM * InteractionRangeIterator::operator *() const{	
	return sitrCurrent != currentEnd ? sitrCurrent->cell:0;
}

bool  InteractionRangeIterator::operator ==(const InteractionRangeIterator & _rhs){
	//cerr<<"this->sitrCurrent="<<this->sitrCurrent->cell<<endl;
	//cerr<<"_rhs.sitrCurrent="<<_rhs.counter<<endl;		

	return _rhs.counter==this->counter ? _rhs.sitrCurrent==this->sitrCurrent:false;
}

bool  InteractionRangeIterator::operator !=(const InteractionRangeIterator & _rhs){
	return !(*this==_rhs);
}
