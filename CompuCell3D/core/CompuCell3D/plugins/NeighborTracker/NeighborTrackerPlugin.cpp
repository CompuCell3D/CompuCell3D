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
// // // //#include <CompuCell3D/plugins/Volume/VolumePlugin.h>
// // // //#include <CompuCell3D/plugins/Volume/VolumeEnergy.h>
// // // #include <CompuCell3D/Potts3D/CellInventory.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Boundary/BoundaryStrategy.h>

using namespace CompuCell3D;

// // // #include <iostream>
// // // #include <cmath>
using namespace std;


#include "NeighborTrackerPlugin.h"

NeighborTrackerPlugin::NeighborTrackerPlugin() :
pUtils(0),
cellFieldG(0),
checkSanity(false),
checkFreq(1),
changeCounter(0),
periodicX(false),
periodicY(false),
periodicZ(false),

maxNeighborIndex(0),
boundaryStrategy(0)

{

}

NeighborTrackerPlugin::~NeighborTrackerPlugin() {
	pUtils->destroyLock(lockPtr);
	delete lockPtr;
	lockPtr=0;
	//cerr<<"\n\n\n DELETING NeighborTrackerPlugin"<<endl;
}

void NeighborTrackerPlugin::init(Simulator *_simulator, CC3DXMLElement *_xmlData) {


	if (_xmlData && _xmlData->findElement("CheckLatticeSanityFrequency"))  {
		checkSanity=true;
		checkFreq=_xmlData->getFirstElement("CheckLatticeSanityFrequency")->getUInt();
	}
	cerr<<"INITIALIZING CELL BOUNDARYTRACKER PLUGIN"<<endl;
	simulator=_simulator;
	Potts3D *potts = simulator->getPotts();
	cellFieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();

	pUtils=simulator->getParallelUtils();
	lockPtr=new ParallelUtilsOpenMP::OpenMPLock_t;
	pUtils->initLock(lockPtr);
    
	///getting cell inventory
	cellInventoryPtr=& potts->getCellInventory(); 



	///will register NeighborTracker here
	BasicClassAccessorBase * cellBTAPtr=&neighborTrackerAccessor;
	///************************************************************************************************  
	///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
	///************************************************************************************************  
	potts->getCellFactoryGroupPtr()->registerClass(cellBTAPtr);

	potts->registerCellGChangeWatcher(this);

	fieldDim=cellFieldG->getDim();

	adjNeighbor = AdjacentNeighbor(fieldDim);
	if(potts->getBoundaryXName()=="Periodic"){ 
		adjNeighbor.setPeriodicX();
		periodicX=true;
	}
	if(potts->getBoundaryYName()=="Periodic"){
		adjNeighbor.setPeriodicY();
		periodicY=true;
	}
	if(potts->getBoundaryZName()=="Periodic"){
		adjNeighbor.setPeriodicZ();
		periodicZ=true;
	}


	boundaryStrategy=BoundaryStrategy::getInstance();
	maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);//1st nearest neighbor


}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void NeighborTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell) {


	if(newCell==oldCell){//happens during multiple calls to se fcn on the same pixel woth current cell - should be avoided
		return;
	}
	//cerr<<"RUNNING NEIGHBOR TRACKER"<<endl;
	const Field3DIndex & field3DIndex=adjNeighbor.getField3DIndex();

	long currentPtIndex=0;
	long adjNeighborIndex=0;
	long adjFace2FaceNeighborIndex=0;

	CellG * currentCellPtr=0;
	CellG * adjCellPtr=0;

	unsigned int token = 0;
	double distance;
	int oldDiff = 0;
	int newDiff = 0;
	Point3D n;
	Point3D ptAdj;
	CellG *nCell=0;
	Neighbor neighbor;

	set<NeighborSurfaceData> * oldCellNeighborSurfaceDataSetPtr=0;
	set<NeighborSurfaceData> * newCellNeighborSurfaceDataSetPtr=0;
	pair<set<NeighborSurfaceData>::iterator,bool > set_NSD_itr_OK_Pair;
	set<NeighborSurfaceData>::iterator set_NSD_itr;


	if(newCell){

		newCellNeighborSurfaceDataSetPtr =  &neighborTrackerAccessor.get(newCell->extraAttribPtr)->cellNeighbors;
	}

	if(oldCell){

		oldCellNeighborSurfaceDataSetPtr =  &neighborTrackerAccessor.get(oldCell->extraAttribPtr)->cellNeighbors;
	}


	currentPtIndex=field3DIndex.index(pt);
	currentCellPtr=cellFieldG->getByIndex(currentPtIndex);


	if(oldCell){

		/// Now will adjust common surface area with cell neighbors
		long temp_index;
		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}
			adjCellPtr=cellFieldG->get(neighbor.pt);

			if( adjCellPtr != oldCell ){ /// will decrement commSurfArea with all face 2 face neighbors
				/*                  cerr<<"adjCellPtr="<<adjCellPtr<<" oldCell="<<oldCell<<endl;
				cerr<<"ptAdj="<<ptAdj<<" pt="<<pt<<endl;*/
				set_NSD_itr = oldCellNeighborSurfaceDataSetPtr->find(NeighborSurfaceData(adjCellPtr));
				if( set_NSD_itr != oldCellNeighborSurfaceDataSetPtr->end() ){
					set_NSD_itr->decrementCommonSurfaceArea(*set_NSD_itr); ///decrement commonSurfArea with adj cell
					if(set_NSD_itr->OKToRemove()) ///if commSurfArea reaches 0 I remove this entry from cell neighbor set
						oldCellNeighborSurfaceDataSetPtr->erase(set_NSD_itr);
						//cerr<<"erasing "<<adjCellPtr<<" from "<< oldCell<<endl;

				}else{

					//cerr<<"adjCellPtr="<<adjCellPtr<<" oldCell="<<oldCell<<" oldCell.type="<<(int)oldCell->type<<" oldCell.id="<<oldCell->id<<endl;
					//cerr<<"oldCell->volume="<<oldCell->volume<<endl;
					//cerr<<"oldCell->surface="<<oldCell->surface<<endl;
					//cerr<<"newCell="<<newCell<<endl;

					//if(newCell){
					//	cerr<<"newCell.type="<<newCell->type<<" id="<<newCell->id<<endl;
					//}
					//if (adjCellPtr){
					//	cerr<<"adjCellPtr.type="<<(int)adjCellPtr->type<<endl;
					//	cerr<<"adjCellPtr->volume="<<adjCellPtr->volume<<endl;
					//	cerr<<"adjCellPtr->surface="<<adjCellPtr->surface<<endl;
					//}
					//cerr<<"neighbor.pt="<<neighbor.pt<<" pt="<<pt<<endl;


					//Neighbor nbr;
					//cerr<<"NEIGHBORS OF pt="<<pt<<" ************************* "<<endl;
					//for(unsigned int ndx=0 ; ndx <= maxNeighborIndex ; ++ndx ){
					//	nbr=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),ndx);
					//	if(!nbr.distance)
					//		continue;
					//	cerr<<nbr.pt<<endl;
					//}
					//cerr<<"********************************************************"<<endl;
					//cerr<<"NEIGHBORS OF neighbor.pt="<<neighbor.pt<<" ************************* "<<endl;
					//for(unsigned int ndx=0 ; ndx <= maxNeighborIndex ; ++ndx ){
					//	nbr=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(neighbor.pt),ndx);
					//	if(!nbr.distance)
					//		continue;
					//	cerr<<nbr.pt<<endl;
					//}

					//cerr<<"********************************************************"<<endl;
					//cerr<<"oldcell neighbors ******************************************"<<endl;
					//for(set<NeighborSurfaceData>::iterator sitr=oldCellNeighborSurfaceDataSetPtr->begin(); sitr!=oldCellNeighborSurfaceDataSetPtr->end(); ++sitr){
					//	if(sitr->neighborAddress)
					//		cerr<<"neighbor.id="<<sitr->neighborAddress->id<<endl;
					//	else
					//		cerr<<"neighbor.id"<<0<<endl;

					//	cerr<<"neighborAddress="<<sitr->neighborAddress<<" commonSurfaceArea="<<sitr->commonSurfaceArea<<endl;

					//}
					testLatticeSanityFull();
					cerr<<"Could not find cell address in the boundary - set of cellNeighbors is corrupted. Exiting ..."<<endl;
					ASSERT_OR_THROW("Could not find cell address in the boundary - set of cellNeighbors is corrupted. Exiting ...",0);
				}


				if(adjCellPtr){ ///now process common area for adj cell provided it is not the oldCell
					set<NeighborSurfaceData> &set_NSD_ref = neighborTrackerAccessor.get(adjCellPtr->extraAttribPtr)->cellNeighbors;
					set<NeighborSurfaceData>::iterator sitr;
					sitr=set_NSD_ref.find(oldCell);
					if(sitr!=set_NSD_ref.end()){
						sitr->decrementCommonSurfaceArea(*sitr); ///decrement common area
						if(sitr->OKToRemove()) ///if commSurfArea reaches 0 I remove this entry from cell neighbor set
							set_NSD_ref.erase(sitr);

					}
				}
			}



		}
	}

	if(newCell){


		/// Now will adjust common surface area with cell neighbors      
		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}
			adjCellPtr=cellFieldG->get(neighbor.pt);


			if( adjCellPtr != newCell ){ ///if adjCellPtr denotes foreign cell we increase common area and insert set entry if necessary
				//                cerr<<"inserting adjCellPtr="<<adjCellPtr <<" ptAdj="<<ptAdj<<" into newCell="<<newCell<<" pt="<<pt<<endl;
				set_NSD_itr_OK_Pair=newCellNeighborSurfaceDataSetPtr->insert(NeighborSurfaceData(adjCellPtr));/// OK to insert even if
				///duplicate, in such a case an iterator to existing NeighborSurfaceData(adjCellPtr) obj is returned

				set_NSD_itr=set_NSD_itr_OK_Pair.first;
				set_NSD_itr->incrementCommonSurfaceArea(*set_NSD_itr); ///increment commonSurfArea with adj cell

				if(adjCellPtr){ ///now process common area for adj cell
					set<NeighborSurfaceData> &set_NSD_ref  = neighborTrackerAccessor.get(adjCellPtr->extraAttribPtr)->cellNeighbors;
					pair<set<NeighborSurfaceData>::iterator,bool> sitr_OK_pair=set_NSD_ref.insert(NeighborSurfaceData(newCell));
					set<NeighborSurfaceData>::iterator sitr=sitr_OK_pair.first;
					sitr->incrementCommonSurfaceArea(*sitr); ///increment commonSurfArea of adj cell with current cell
				}

			}




		}
	}

	token = 0;
	distance = 0;



	if(!oldCell){ ///this special case is required in updating common Surface Area with medium
		///in this case we update surface of adjCell only (we do not update medium's neighbors list or its contact surfaces)

		/// Now will adjust common surface area with cell neighbors
		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}
			adjCellPtr=cellFieldG->get(neighbor.pt);


			if( adjCellPtr != oldCell /*&& !(ptAdj == pt)*/){ /// will decrement commSurfArea with all face 2 face neighbors
				//                cerr<<"!old cell section  adjCellPtr="<<adjCellPtr <<" ptAdj="<<ptAdj<<"  oldCell="<<oldCell<<" pt="<<pt<<endl;
				if(adjCellPtr){ ///now process common area for adj cell provided it is not the oldCell
					set<NeighborSurfaceData> &set_NSD_ref = neighborTrackerAccessor.get(adjCellPtr->extraAttribPtr)->cellNeighbors;
					set<NeighborSurfaceData>::iterator sitr;
					sitr=set_NSD_ref.find(oldCell);
					if(sitr!=set_NSD_ref.end()){
						sitr->decrementCommonSurfaceArea(*sitr); ///decrement common area
						if(sitr->OKToRemove()){ ///if commSurfArea reaches 0 I remove this entry from cell neighbor set
							set_NSD_ref.erase(sitr);
							//                         cerr<<"removing from boundary"<<endl;

						}

					}
				}
			}



		}


	}

	token = 0;
	distance = 0;


	if(!newCell){  ///this special case is required in updating common Surface Area with medium
		///in this case we update surface of adjCell only (we do not update medium's neighbors list or its contact surfaces)

		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			if(cellFieldG->isValid(neighbor.pt)){

				adjCellPtr=cellFieldG->get(neighbor.pt);

				if( adjCellPtr != newCell ){ ///if adjCellPtr denotes foreign cell we increase common area and insert set entry if necessary

					if(adjCellPtr){ ///now process common area of adj cell with medium in this case
						set<NeighborSurfaceData> &set_NSD_ref  = neighborTrackerAccessor.get(adjCellPtr->extraAttribPtr)->cellNeighbors;
						pair<set<NeighborSurfaceData>::iterator,bool> sitr_OK_pair=set_NSD_ref.insert(NeighborSurfaceData(newCell));
						set<NeighborSurfaceData>::iterator sitr=sitr_OK_pair.first;
						sitr->incrementCommonSurfaceArea(*sitr); ///increment commonSurfArea of adj cell with current cell
					}

				}

			}


		}

	}


	///temporarily for testing purposes I set 

	if(checkSanity){
        pUtils->setLock(lockPtr);
		++changeCounter;

		if(!(changeCounter % checkFreq)){
			//cerr<<"OLD CELL ADR: "<<oldCell<<" NEW CELL ADR: "<<newCell<<endl;
			cerr<<"ChangeCounter:"<<changeCounter<<endl;
			testLatticeSanityFull();

		}
        pUtils->unsetLock(lockPtr);
	}


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double distance(double x1,double y1,double z1,double x2,double y2,double z2){
	return sqrt (
		(x1-x2)*(x1-x2)+
		(y1-y2)*(y1-y2)+
		(z1-z2)*(z1-z2)
		);
}


void NeighborTrackerPlugin::testLatticeSanityFull(){



	Dim3D fieldDim=cellFieldG->getDim();

	Point3D pt(0,0,0);
	Point3D ptAdj;
	Neighbor neighbor;



	///Now will have to get access to the pointers stored in cellFieldG from Potts3D


	CellG * currentCellPtr;
	CellG * adjCellPtr;

	map<CellG*,set<NeighborSurfaceData> > mapCellNeighborSurfaceData;
	map<CellG*,set<NeighborSurfaceData> >::iterator mitr;

	/// check neighbors of each cell - will loop over each lattice point and check if point belongs to boundary, then will examine neighbors

	unsigned int token = 0;
	double distance;

	set<NeighborSurfaceData> * set_NSD_ptr;

	pair<set<NeighborSurfaceData>::iterator,bool > set_NSD_itr_OK_Pair;
	set<NeighborSurfaceData>::iterator set_NSD_itr;

	//set<CellG*> cellPointersSet;

	for(int z=0 ; z < fieldDim.z ; ++z)
		for(int y=0 ; y < fieldDim.y ; ++y)
			for(int x=0 ; x < fieldDim.x ; ++x){
				pt.x=x;
				pt.y=y;
				pt.z=z;

				token=0;
				distance=0;

				//             currentPtIndex=field3DIndex.index(pt);
				//currentCellPtr=cellFieldG->getByIndex(currentPtIndex);
				currentCellPtr=cellFieldG->get(pt);
				//if(currentCellPtr)
				//	cellPointersSet.insert(currentCellPtr);

				if(!currentCellPtr)
					continue; //skip the loop if the current latice site does not belong to any cell
				if(!isBoundaryPixel(pt))
					continue; //inner pixel does not bring new neighbors

				mitr=mapCellNeighborSurfaceData.find(currentCellPtr);
				if(mitr != mapCellNeighborSurfaceData.end() ){
					set_NSD_ptr = & (mitr->second) ;
				}else{
					mapCellNeighborSurfaceData.insert(make_pair(currentCellPtr,set<NeighborSurfaceData>()));
					mitr=mapCellNeighborSurfaceData.find(currentCellPtr);
					set_NSD_ptr = &(mitr->second);
				}

				for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
					neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
					if(!neighbor.distance){
						//if distance is 0 then the neighbor returned is invalid
						continue;
					}
					adjCellPtr=cellFieldG->get(neighbor.pt);

					if(adjCellPtr != currentCellPtr){
						set_NSD_itr_OK_Pair = set_NSD_ptr->insert(NeighborSurfaceData(adjCellPtr));/// OK to insert even if
						///duplicate, in such a case an iterator to existing NeighborSurfaceData(adjCellPtr) obj is returned

						set_NSD_itr=set_NSD_itr_OK_Pair.first;
						set_NSD_itr->incrementCommonSurfaceArea(*set_NSD_itr); ///increment commonSurfArea with adj cell

					}

				}

			}


			//Now do lattice sanity checks
			if(mapCellNeighborSurfaceData.size() != cellInventoryPtr->getCellInventorySize()){
				cerr<<"Number of cells in the mapCellNeighborSurfaceData = "<<mapCellNeighborSurfaceData.size()
					<<" is different than in cell inventory:  "<< cellInventoryPtr->getCellInventorySize()<<endl;
				//cerr<<"cellPointersSet.size()="<<cellPointersSet.size()<<endl;
				exit(0);
			}
			CellInventory::cellInventoryIterator cInvItr;
			CellG * cell;
			int counter=0;

			for(cInvItr=cellInventoryPtr->cellInventoryBegin() ; cInvItr !=cellInventoryPtr->cellInventoryEnd() ;++cInvItr ){
				cell=cellInventoryPtr->getCell(cInvItr);
				//cell=*cInvItr;
				mitr = mapCellNeighborSurfaceData.find(cell);
				if(mitr==mapCellNeighborSurfaceData.end()){
					cerr<<"Cell "<<cell<<" does not appear in the just initialized mapCellNeighborSurfaceData"<<endl;
					exit(0);

				}
				set_NSD_ptr=&(mitr->second);

				if(! (*set_NSD_ptr == neighborTrackerAccessor.get(cell->extraAttribPtr)->cellNeighbors)){
					cerr<<"Have checked "<<counter<<" cells"<<endl;
					cerr<<"set of NeighborSurfaceData do not match for cell: "<<cell<<endl;
					cerr<<"cell->id="<<cell->id<<" cell->type="<<(int)cell->type<<endl;
					exit(0);
				}

				++counter;
			}

			cerr<<"FULL TEST: LATTICE IS SANE!!!!!"<<endl;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool NeighborTrackerPlugin::isBoundaryPixel(Point3D pt){

	//     const vector<Point3D> & adjNeighborOffsetsVec=adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
	CellG * currentCellPtr=cellFieldG->get(pt);;
	//     CellG * adjCellPtr;
	CellG * nCell;
	unsigned int token = 0;
	double distance = 0;
	Point3D n;

	Neighbor nbr;
	for(unsigned int ndx=0 ; ndx <= maxNeighborIndex ; ++ndx ){
		nbr=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),ndx);
		if(!nbr.distance)
			continue;
		nCell = cellFieldG->get(nbr.pt);
		if(nCell != currentCellPtr)
			return true;


	}


	//while (true) {
	//   n = cellFieldG->getNeighbor(pt, token, distance, false);
	//   if (distance > 1) break;//only nearest neighbors
	//   nCell = cellFieldG->get(n);
	//   if(nCell != currentCellPtr)
	//      return true;
	//}

	return false;

}


std::string NeighborTrackerPlugin::toString(){
	return "NeighborTracker";
}
